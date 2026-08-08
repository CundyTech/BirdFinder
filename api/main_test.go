package main

import (
	"bytes"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
)

func init() {
	gin.SetMode(gin.TestMode)
}

// multipartBody builds a multipart/form-data body with an optional file
// field named "image" containing size bytes of content.
func multipartBody(t *testing.T, includeImage bool, size int) (*bytes.Buffer, string) {
	t.Helper()
	body := &bytes.Buffer{}
	w := multipart.NewWriter(body)
	if includeImage {
		fw, err := w.CreateFormFile("image", "test.jpg")
		if err != nil {
			t.Fatalf("CreateFormFile: %v", err)
		}
		if _, err := fw.Write(bytes.Repeat([]byte{0xFF}, size)); err != nil {
			t.Fatalf("write form file: %v", err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("close multipart writer: %v", err)
	}
	return body, w.FormDataContentType()
}

func TestRootEndpoint(t *testing.T) {
	router := newRouter()
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	if rec.Body.Len() == 0 {
		t.Fatal("expected a non-empty body")
	}
}

func TestHealthEndpoint(t *testing.T) {
	router := newRouter()
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}

	var payload map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("invalid JSON response: %v", err)
	}
	if payload["status"] != "healthy" {
		t.Errorf("expected status=healthy, got %q", payload["status"])
	}
}

func TestPredictHandler_MissingImageField(t *testing.T) {
	router := newRouter()
	body, contentType := multipartBody(t, false, 0)
	req := httptest.NewRequest(http.MethodPost, "/predict", body)
	req.Header.Set("Content-Type", contentType)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", rec.Code, rec.Body.String())
	}

	var payload map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("invalid JSON response: %v", err)
	}
	if payload["error"] != "missing 'image' form file" {
		t.Errorf("unexpected error message: %q", payload["error"])
	}
}

func TestPredictHandler_InvalidMultipartForm(t *testing.T) {
	router := newRouter()
	req := httptest.NewRequest(http.MethodPost, "/predict", bytes.NewBufferString("not a multipart body"))
	req.Header.Set("Content-Type", "text/plain")
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", rec.Code, rec.Body.String())
	}

	var payload map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("invalid JSON response: %v", err)
	}
	if payload["error"] != "invalid multipart form" {
		t.Errorf("unexpected error message: %q", payload["error"])
	}
}

func TestPredictHandler_UploadTooLarge(t *testing.T) {
	// Shrink the cap for this test so we don't need a real multi-megabyte
	// payload to exercise the same http.MaxBytesReader code path. Must be
	// set before newRouter() runs — the limit is baked into the middleware
	// at router-construction time, not read live per-request.
	original := maxUploadSize
	maxUploadSize = 16
	defer func() { maxUploadSize = original }()

	router := newRouter()

	body, contentType := multipartBody(t, true, 1024)
	req := httptest.NewRequest(http.MethodPost, "/predict", body)
	req.Header.Set("Content-Type", contentType)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("expected 413, got %d: %s", rec.Code, rec.Body.String())
	}

	var payload map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("invalid JSON response: %v", err)
	}
	if payload["error"] == "" {
		t.Error("expected a non-empty error message")
	}
}

func TestPredictHandler_UnderSizeLimitReachesPredictor(t *testing.T) {
	// A well-formed, under-the-cap upload should sail past validation and
	// reach the predictor step. We can't run the real Python predictor in a
	// unit test, so this just confirms we don't get rejected at 400/413 —
	// i.e. validation correctly let a legitimate-shaped request through.
	router := newRouter()
	body, contentType := multipartBody(t, true, 1024)
	req := httptest.NewRequest(http.MethodPost, "/predict", body)
	req.Header.Set("Content-Type", contentType)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if rec.Code == http.StatusBadRequest || rec.Code == http.StatusRequestEntityTooLarge {
		t.Fatalf("valid-shaped upload was rejected at validation: %d: %s", rec.Code, rec.Body.String())
	}
}
