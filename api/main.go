package main

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

func main() {
	http.HandleFunc("/predict", corsMiddleware(predictHandler))
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("BirdFinder API: POST /predict (multipart form field 'image')"))
	})

	addr := ":8080"
	log.Printf("Starting API on %s\n", addr)
	log.Fatal(http.ListenAndServe(addr, nil))
}

// corsMiddleware adds simple CORS headers and handles OPTIONS preflight.
func corsMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}
		next(w, r)
	}
}

func predictHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	err := r.ParseMultipartForm(20 << 20) // 20 MB
	if err != nil {
		http.Error(w, "failed to parse multipart form", http.StatusBadRequest)
		return
	}

	file, _, err := r.FormFile("image")
	if err != nil {
		http.Error(w, "missing 'image' form file", http.StatusBadRequest)
		return
	}
	defer file.Close()

	tmp, err := os.CreateTemp("", "upload-*.jpg")
	if err != nil {
		http.Error(w, "failed to create temp file", http.StatusInternalServerError)
		return
	}
	defer os.Remove(tmp.Name())

	if _, err := io.Copy(tmp, file); err != nil {
		http.Error(w, "failed to save uploaded file", http.StatusInternalServerError)
		return
	}
	tmp.Close()

	// Assume API is run from the `api` directory. Use relative path to the Python wrapper.
	scriptPath := filepath.Join("..", "model", "build", "predict_cli.py")

	// Call the Python prediction wrapper
	cmd := exec.Command("python", scriptPath, "--image", tmp.Name())
	// set a reasonable timeout by using a goroutine + channel
	done := make(chan struct{})
	var out []byte
	var cmdErr error
	go func() {
		out, cmdErr = cmd.CombinedOutput()
		close(done)
	}()

	select {
	case <-done:
		if cmdErr != nil {
			msg := fmt.Sprintf("predictor failed: %v\n%s", cmdErr, string(out))
			log.Print(msg)
			http.Error(w, msg, http.StatusInternalServerError)
			return
		}
	case <-time.After(30 * time.Second):
		cmd.Process.Kill()
		http.Error(w, "prediction timed out", http.StatusGatewayTimeout)
		return
	}

	// The Python wrapper prints a JSON object to stdout. Forward it as-is.
	w.Header().Set("Content-Type", "application/json")
	w.Write(out)
}
