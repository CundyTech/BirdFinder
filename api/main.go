package main

import (
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/gin-gonic/gin"
)

func main() {
	router := gin.New()
	router.Use(gin.Logger(), gin.Recovery(), corsMiddleware())

	router.GET("/", func(c *gin.Context) {
		c.String(200, "BirdFinder API: POST /predict (multipart form field 'image')")
	})

	router.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"status":  "healthy",
			"service": "BirdFinder API",
			"version": "1.0.0",
		})
	})

	router.POST("/predict", predictHandler)

	addr := "0.0.0.0:8080"
	log.Printf("Starting API on %s (accessible from local network)\n", addr)
	log.Fatal(router.Run(addr))
}

func corsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		c.Writer.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(200)
			return
		}
		c.Next()
	}
}

func predictHandler(c *gin.Context) {
	log.Printf("Received predict request")

	// Try to parse multipart form
	form, err := c.MultipartForm()
	if err != nil {
		log.Printf("MultipartForm error: %v", err)
		c.JSON(400, gin.H{"error": "invalid multipart form"})
		return
	}

	files := form.File["image"]
	if len(files) == 0 {
		log.Printf("No image files found in form")
		c.JSON(400, gin.H{"error": "missing 'image' form file"})
		return
	}

	file := files[0]
	log.Printf("Received file: %s, size: %d", file.Filename, file.Size)

	// Get file extension from original filename
	ext := filepath.Ext(file.Filename)
	if ext == "" {
		ext = ".jpg" // default extension
	}

	tmp, err := os.CreateTemp("", "upload-*"+ext)
	if err != nil {
		c.JSON(500, gin.H{"error": "failed to create temp file"})
		return
	}
	defer os.Remove(tmp.Name())

	log.Printf("Created temp file: %s", tmp.Name())

	src, err := file.Open()
	if err != nil {
		c.JSON(500, gin.H{"error": "failed to read uploaded file"})
		return
	}
	defer src.Close()

	if _, err := io.Copy(tmp, src); err != nil {
		c.JSON(500, gin.H{"error": "failed to save uploaded file"})
		return
	}

	log.Printf("Saved uploaded file to temp: %s", tmp.Name())

	// Check file size
	if stat, err := tmp.Stat(); err == nil {
		log.Printf("Uploaded file size: %d bytes", stat.Size())
	}
	tmp.Close()

	// Assume API is run from the `api` directory. Use relative path to the Python wrapper.
	scriptPath := filepath.Join("..", "model", "build", "predict_cli.py")

	cmd := exec.Command("C:\\Users\\DanCu\\AppData\\Local\\Programs\\Python\\Python311-arm64\\python.exe", scriptPath, "--image", tmp.Name())
	done := make(chan struct{})
	var out []byte
	var cmdErr error
	go func() {
		out, cmdErr = cmd.Output() // Use Output() instead of CombinedOutput() to avoid stderr
		close(done)
	}()

	select {
	case <-done:
		if cmdErr != nil {
			msg := fmt.Sprintf("predictor failed: %v", cmdErr)
			log.Print(msg)
			c.JSON(500, gin.H{"error": msg})
			return
		}
	case <-time.After(30 * time.Second):
		if cmd.Process != nil {
			cmd.Process.Kill()
		}
		c.JSON(504, gin.H{"error": "prediction timed out"})
		return
	}

	c.Data(200, "application/json", out)

	log.Printf("Prediction completed successfully")
}
