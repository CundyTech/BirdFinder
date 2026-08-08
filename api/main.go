package main

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"birdfinder/api/middleware"
)

// maxUploadSize caps the /predict request body. Phone camera photos are
// typically a few MB; 10MB gives headroom without leaving the endpoint open
// to unbounded uploads. Declared as a var (not const) so tests can shrink it
// rather than uploading real multi-megabyte payloads.
var maxUploadSize int64 = 10 << 20 // 10 MB

// Per-IP rate limit for /predict, the endpoint that spawns a Python
// subprocess per request. predictRateBurst allows a short run of requests
// (e.g. someone taking a few photos in a row) before the sustained
// predictRateRPS refill rate kicks in.
const (
	predictRateRPS   = 1
	predictRateBurst = 5
)

// maxConcurrentPredictions caps how many predict_cli.py subprocesses can run
// at once. Per-IP rate limiting alone doesn't bound aggregate load across
// many distinct clients; this protects the host itself from being
// overwhelmed regardless of how many different IPs are involved.
// predictionSlotWait is how long a request will queue for a free slot
// before giving up.
const (
	maxConcurrentPredictions = 3
	predictionSlotWait       = 10 * time.Second
)

func newRouter() *gin.Engine {
	router := gin.New()
	router.Use(gin.Logger(), gin.Recovery(), middleware.CORS())

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

	predictLimiter := middleware.NewIPRateLimiter(predictRateRPS, predictRateBurst)
	predictSemaphore := middleware.NewSemaphore(maxConcurrentPredictions, predictionSlotWait)
	router.POST("/predict",
		predictLimiter.Middleware(),
		predictSemaphore.Middleware(),
		middleware.MaxUploadSize(maxUploadSize),
		predictHandler,
	)

	return router
}

func main() {
	router := newRouter()
	addr := "0.0.0.0:8080"
	log.Printf("Starting API on %s (accessible from local network)\n", addr)
	log.Fatal(router.Run(addr))
}

// detectImageContentType sniffs r's content type from its first bytes
// (http.DetectContentType only needs up to 512), without losing those bytes
// for the caller — it returns a reader that replays them followed by the
// rest of r. This rejects obviously-non-image uploads (garbage, scripts,
// arbitrary files) before spending a subprocess spawn on them; it doesn't
// guarantee the file is a *valid* image — that's still predict_cli.py's job.
func detectImageContentType(r io.Reader) (io.Reader, string, error) {
	sniff := make([]byte, 512)
	n, err := io.ReadFull(r, sniff)
	if err != nil && !errors.Is(err, io.ErrUnexpectedEOF) && !errors.Is(err, io.EOF) {
		return nil, "", err
	}
	sniff = sniff[:n]
	contentType := http.DetectContentType(sniff)
	return io.MultiReader(bytes.NewReader(sniff), r), contentType, nil
}

func predictHandler(c *gin.Context) {
	log.Printf("Received predict request")

	// Try to parse multipart form
	form, err := c.MultipartForm()
	if err != nil {
		var maxBytesErr *http.MaxBytesError
		if errors.As(err, &maxBytesErr) {
			log.Printf("Upload exceeded max size: %v", err)
			c.JSON(413, gin.H{"error": fmt.Sprintf("image exceeds maximum size of %d bytes", maxUploadSize)})
			return
		}
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

	imgReader, contentType, err := detectImageContentType(src)
	if err != nil {
		c.JSON(500, gin.H{"error": "failed to read uploaded file"})
		return
	}
	if !strings.HasPrefix(contentType, "image/") {
		log.Printf("Rejected upload: detected content type %q, not an image", contentType)
		c.JSON(400, gin.H{"error": fmt.Sprintf("uploaded file is not a recognized image type (detected %s)", contentType)})
		return
	}

	if _, err := io.Copy(tmp, imgReader); err != nil {
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
	log.Printf("Prediction output: %s", string(out))
	log.Printf("Prediction completed successfully")
}
