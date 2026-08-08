package main

import (
	"bytes"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func init() {
	gin.SetMode(gin.TestMode)
}

// multipartBody builds a multipart/form-data body with an optional file
// field named "image" containing size bytes of content that starts with a
// real JPEG magic-byte prefix — enough to pass content-type sniffing, same
// as a corrupted-but-genuinely-a-photo upload would, without being a
// structurally valid JPEG (predict_cli.py is still expected to reject it,
// just later in the pipeline than the sniff check).
func multipartBody(t *testing.T, includeImage bool, size int) (*bytes.Buffer, string) {
	t.Helper()
	body := &bytes.Buffer{}
	w := multipart.NewWriter(body)
	if includeImage {
		fw, err := w.CreateFormFile("image", "test.jpg")
		if err != nil {
			t.Fatalf("CreateFormFile: %v", err)
		}
		if _, err := fw.Write(jpegLikeContent(size)); err != nil {
			t.Fatalf("write form file: %v", err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("close multipart writer: %v", err)
	}
	return body, w.FormDataContentType()
}

func jpegLikeContent(size int) []byte {
	magic := []byte{0xFF, 0xD8, 0xFF}
	if size <= len(magic) {
		return magic[:size]
	}
	return append(magic, bytes.Repeat([]byte{0xFF}, size-len(magic))...)
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

func TestPredictHandler_RejectsNonImageUpload(t *testing.T) {
	router := newRouter()

	body := &bytes.Buffer{}
	w := multipart.NewWriter(body)
	fw, err := w.CreateFormFile("image", "not-an-image.jpg")
	if err != nil {
		t.Fatalf("CreateFormFile: %v", err)
	}
	if _, err := fw.Write([]byte("just some plain text pretending to be a photo")); err != nil {
		t.Fatalf("write form file: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("close multipart writer: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/predict", body)
	req.Header.Set("Content-Type", w.FormDataContentType())
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for a non-image upload, got %d: %s", rec.Code, rec.Body.String())
	}
	var payload map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("invalid JSON response: %v", err)
	}
	if payload["error"] == "" {
		t.Error("expected a non-empty error message")
	}
}

func TestDetectImageContentType(t *testing.T) {
	cases := []struct {
		name      string
		content   []byte
		wantImage bool
	}{
		{"jpeg magic bytes", []byte{0xFF, 0xD8, 0xFF, 0x01, 0x02, 0x03}, true},
		{"png magic bytes", []byte{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A, 0x00}, true},
		{"plain text", []byte("hello, this is not an image"), false},
		{"empty", []byte{}, false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			reader, contentType, err := detectImageContentType(bytes.NewReader(tc.content))
			if err != nil {
				t.Fatalf("detectImageContentType: %v", err)
			}

			isImage := strings.HasPrefix(contentType, "image/")
			if isImage != tc.wantImage {
				t.Errorf("content type %q: got isImage=%v, want %v", contentType, isImage, tc.wantImage)
			}

			replayed, err := io.ReadAll(reader)
			if err != nil {
				t.Fatalf("reading returned reader: %v", err)
			}
			if !bytes.Equal(replayed, tc.content) {
				t.Errorf("returned reader yielded %v, want original content %v", replayed, tc.content)
			}
		})
	}
}

func TestDetectImageContentType_PreservesContentLongerThanSniffWindow(t *testing.T) {
	content := jpegLikeContent(2000) // longer than the 512-byte sniff window
	reader, contentType, err := detectImageContentType(bytes.NewReader(content))
	if err != nil {
		t.Fatalf("detectImageContentType: %v", err)
	}
	if !strings.HasPrefix(contentType, "image/") {
		t.Errorf("expected an image content type, got %q", contentType)
	}

	replayed, err := io.ReadAll(reader)
	if err != nil {
		t.Fatalf("reading returned reader: %v", err)
	}
	if !bytes.Equal(replayed, content) {
		t.Errorf("returned reader lost or altered content beyond the sniff window (got %d bytes, want %d)", len(replayed), len(content))
	}
}
