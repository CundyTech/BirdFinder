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

func newCORSTestRouter() *gin.Engine {
	router := gin.New()
	router.Use(CORS())
	router.GET("/health", func(c *gin.Context) { c.Status(http.StatusOK) })
	router.POST("/predict", func(c *gin.Context) { c.Status(http.StatusOK) })
	return router
}

func TestCORS_PreflightRequest(t *testing.T) {
	router := newCORSTestRouter()
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

func TestCORS_HeadersSetOnNormalRequest(t *testing.T) {
	router := newCORSTestRouter()
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if got := rec.Header().Get("Access-Control-Allow-Origin"); got != "*" {
		t.Errorf("expected Access-Control-Allow-Origin=*, got %q", got)
	}
}
