package middleware

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"golang.org/x/time/rate"
)

// newRateLimitTestRouter puts the limiter on /limited only, so tests can
// confirm it doesn't leak into unrelated routes.
func newRateLimitTestRouter(limiter *IPRateLimiter) *gin.Engine {
	router := gin.New()
	router.GET("/unlimited", func(c *gin.Context) { c.Status(http.StatusOK) })
	router.GET("/limited", limiter.Middleware(), func(c *gin.Context) { c.Status(http.StatusOK) })
	return router
}

func rateLimitedRequest(router *gin.Engine, remoteAddr string) int {
	req := httptest.NewRequest(http.MethodGet, "/limited", nil)
	if remoteAddr != "" {
		req.RemoteAddr = remoteAddr
	}
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	return rec.Code
}

// noRefill is a near-zero rate so no meaningful token refill happens during
// a test, keeping burst-exhaustion tests deterministic.
const noRefill = rate.Limit(0.0001)

func TestIPRateLimiter_BlocksAfterBurst(t *testing.T) {
	router := newRateLimitTestRouter(NewIPRateLimiter(noRefill, 2))

	for i := 1; i <= 2; i++ {
		if code := rateLimitedRequest(router, ""); code == http.StatusTooManyRequests {
			t.Fatalf("request %d unexpectedly rate limited", i)
		}
	}
	if code := rateLimitedRequest(router, ""); code != http.StatusTooManyRequests {
		t.Fatalf("expected 429 after exhausting burst, got %d", code)
	}
}

func TestIPRateLimiter_PerIPIsolation(t *testing.T) {
	router := newRateLimitTestRouter(NewIPRateLimiter(noRefill, 1))

	if code := rateLimitedRequest(router, "203.0.113.1:1111"); code == http.StatusTooManyRequests {
		t.Fatal("first request from IP A unexpectedly rate limited")
	}
	if code := rateLimitedRequest(router, "203.0.113.1:1111"); code != http.StatusTooManyRequests {
		t.Fatalf("expected IP A's second request to be rate limited, got %d", code)
	}
	if code := rateLimitedRequest(router, "203.0.113.2:2222"); code == http.StatusTooManyRequests {
		t.Fatal("IP B was rate limited by IP A's usage — limiter is not per-IP")
	}
}

func TestIPRateLimiter_DoesNotApplyToUnrelatedRoute(t *testing.T) {
	router := newRateLimitTestRouter(NewIPRateLimiter(noRefill, 1))

	rateLimitedRequest(router, "")
	rateLimitedRequest(router, "") // exhausts the burst on /limited

	req := httptest.NewRequest(http.MethodGet, "/unlimited", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected unrelated route to be unaffected, got %d", rec.Code)
	}
}

func TestIPRateLimiter_ErrorResponseBody(t *testing.T) {
	router := newRateLimitTestRouter(NewIPRateLimiter(noRefill, 1))

	rateLimitedRequest(router, "")
	req := httptest.NewRequest(http.MethodGet, "/limited", nil)
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
