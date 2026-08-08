package main

import (
	"bytes"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"golang.org/x/time/rate"
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

func TestCorsMiddleware_PreflightRequest(t *testing.T) {
	router := newRouter()
	req := httptest.NewRequest(http.MethodOptions, "/predict", nil)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 for OPTIONS preflight, got %d", rec.Code)
	}
	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "*" {
		t.Errorf("expected Access-Control-Allow-Origin=*, got %q", got)
	}
	if got := rec.Header().Get("Access-Control-Allow-Methods"); got != "POST, OPTIONS" {
		t.Errorf("expected Access-Control-Allow-Methods='POST, OPTIONS', got %q", got)
	}
}

func TestCorsMiddleware_HeadersSetOnNormalRequest(t *testing.T) {
	router := newRouter()
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "*" {
		t.Errorf("expected Access-Control-Allow-Origin=*, got %q", got)
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
	router := newRouter()

	// Shrink the cap for this test so we don't need a real multi-megabyte
	// payload to exercise the same http.MaxBytesReader code path.
	original := maxUploadSize
	maxUploadSize = 16
	defer func() { maxUploadSize = original }()

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

// predictRequest sends a POST /predict with a body that fails validation
// fast (missing image field) so tests exercise the rate limiter without
// spawning the real predictor subprocess. remoteAddr overrides the request's
// source IP when non-empty; httptest.NewRequest otherwise defaults every
// call to the same fixed address, which is exactly what per-IP tests want.
func predictRequest(t *testing.T, router *gin.Engine, remoteAddr string) int {
	t.Helper()
	body, contentType := multipartBody(t, false, 0)
	req := httptest.NewRequest(http.MethodPost, "/predict", body)
	req.Header.Set("Content-Type", contentType)
	if remoteAddr != "" {
		req.RemoteAddr = remoteAddr
	}
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	return rec.Code
}

// withRateLimit temporarily overrides the global rate limit settings for a
// test, restoring the originals on cleanup. rps is set near-zero so no
// meaningful token refill happens during the test, keeping it deterministic.
func withRateLimit(t *testing.T, rps rate.Limit, burst int) {
	t.Helper()
	origRPS, origBurst := rateLimitRPS, rateLimitBurst
	rateLimitRPS, rateLimitBurst = rps, burst
	t.Cleanup(func() { rateLimitRPS, rateLimitBurst = origRPS, origBurst })
}

func TestRateLimiter_BlocksAfterBurst(t *testing.T) {
	withRateLimit(t, rate.Limit(0.0001), 2)
	router := newRouter()

	for i := 1; i <= 2; i++ {
		if code := predictRequest(t, router, ""); code == http.StatusTooManyRequests {
			t.Fatalf("request %d unexpectedly rate limited", i)
		}
	}

	if code := predictRequest(t, router, ""); code != http.StatusTooManyRequests {
		t.Fatalf("expected 429 after exhausting burst, got %d", code)
	}
}

func TestRateLimiter_PerIPIsolation(t *testing.T) {
	withRateLimit(t, rate.Limit(0.0001), 1)
	router := newRouter()

	if code := predictRequest(t, router, "203.0.113.1:1111"); code == http.StatusTooManyRequests {
		t.Fatal("first request from IP A unexpectedly rate limited")
	}
	if code := predictRequest(t, router, "203.0.113.1:1111"); code != http.StatusTooManyRequests {
		t.Fatalf("expected IP A's second request to be rate limited, got %d", code)
	}
	if code := predictRequest(t, router, "203.0.113.2:2222"); code == http.StatusTooManyRequests {
		t.Fatal("IP B was rate limited by IP A's usage — limiter is not per-IP")
	}
}

func TestRateLimiter_DoesNotApplyToHealthEndpoint(t *testing.T) {
	withRateLimit(t, rate.Limit(0.0001), 1)
	router := newRouter()

	// Exhaust the /predict limiter for this IP.
	predictRequest(t, router, "")
	predictRequest(t, router, "")

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected /health to be unaffected by /predict rate limiting, got %d", rec.Code)
	}
}

func TestRateLimiter_ErrorResponseBody(t *testing.T) {
	withRateLimit(t, rate.Limit(0.0001), 1)
	router := newRouter()

	predictRequest(t, router, "")
	body, contentType := multipartBody(t, false, 0)
	req := httptest.NewRequest(http.MethodPost, "/predict", body)
	req.Header.Set("Content-Type", contentType)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusTooManyRequests {
		t.Fatalf("expected 429, got %d", rec.Code)
	}
	var payload map[string]string
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("invalid JSON response: %v", err)
	}
	if payload["error"] == "" {
		t.Error("expected a non-empty error message")
	}
}
