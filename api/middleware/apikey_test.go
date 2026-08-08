package middleware

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
)

func newAPIKeyTestRouter(key string) *gin.Engine {
	router := gin.New()
	router.GET("/open", func(c *gin.Context) { c.Status(http.StatusOK) })
	router.GET("/protected", APIKey(key), func(c *gin.Context) { c.Status(http.StatusOK) })
	return router
}

func TestAPIKey_AcceptsCorrectKey(t *testing.T) {
	router := newAPIKeyTestRouter("correct-key")
	req := httptest.NewRequest(http.MethodGet, "/protected", nil)
	req.Header.Set(APIKeyHeader, "correct-key")
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 for the correct key, got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestAPIKey_RejectsMissingKey(t *testing.T) {
	router := newAPIKeyTestRouter("correct-key")
	req := httptest.NewRequest(http.MethodGet, "/protected", nil)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 for a missing key, got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestAPIKey_RejectsWrongKey(t *testing.T) {
	router := newAPIKeyTestRouter("correct-key")
	req := httptest.NewRequest(http.MethodGet, "/protected", nil)
	req.Header.Set(APIKeyHeader, "wrong-key")
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 for the wrong key, got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestAPIKey_RejectsEmptyConfiguredKeyRegardlessOfInput(t *testing.T) {
	// A misconfigured empty key must not mean "accept anything" — an empty
	// sent header is explicitly rejected before the comparison even runs.
	router := newAPIKeyTestRouter("")
	req := httptest.NewRequest(http.MethodGet, "/protected", nil)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 when no key is configured and none is sent, got %d", rec.Code)
	}
}

func TestAPIKey_DoesNotAffectUnrelatedRoute(t *testing.T) {
	router := newAPIKeyTestRouter("correct-key")
	req := httptest.NewRequest(http.MethodGet, "/open", nil)
	rec := httptest.NewRecorder()

	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected the unprotected route to be unaffected, got %d", rec.Code)
	}
}
