package middleware

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
)

func init() {
	gin.SetMode(gin.TestMode)
}

func newCORSTestRouter(allowedOrigins ...string) *gin.Engine {
	router := gin.New()
	router.Use(CORS(allowedOrigins...))
	router.GET("/health", func(c *gin.Context) { c.Status(http.StatusOK) })
	router.POST("/predict", func(c *gin.Context) { c.Status(http.StatusOK) })
	return router
}

func TestCORS_AllowsConfiguredOrigin(t *testing.T) {
	router := newCORSTestRouter("https://example.com")
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	req.Header.Set("Origin", "https://example.com")
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "https://example.com" {
		t.Errorf("expected Access-Control-Allow-Origin=https://example.com, got %q", got)
	}
	if got := rec.Header().Get("Access-Control-Allow-Methods"); got != "POST, OPTIONS" {
		t.Errorf("expected Access-Control-Allow-Methods='POST, OPTIONS', got %q", got)
	}
}

func TestCORS_RejectsUnconfiguredOrigin(t *testing.T) {
	router := newCORSTestRouter("https://example.com")
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	req.Header.Set("Origin", "https://evil.example")
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "" {
		t.Errorf("expected no Access-Control-Allow-Origin header for an unconfigured origin, got %q", got)
	}
}

func TestCORS_NoOriginsConfiguredAllowsNone(t *testing.T) {
	router := newCORSTestRouter() // production default: no web origins allowed
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	req.Header.Set("Origin", "https://example.com")
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "" {
		t.Errorf("expected no Access-Control-Allow-Origin header with no origins configured, got %q", got)
	}
}

func TestCORS_PreflightStillReturns200(t *testing.T) {
	// Even for a disallowed/absent origin, the preflight request itself
	// should still get 200 — the browser enforces CORS based on the
	// (missing) header, not the HTTP status.
	router := newCORSTestRouter("https://example.com")
	req := httptest.NewRequest(http.MethodOptions, "/predict", nil)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 for OPTIONS preflight, got %d", rec.Code)
	}
}
