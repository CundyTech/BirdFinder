package middleware

import (
	"bytes"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestMaxUploadSize_AllowsBodyAtLimit(t *testing.T) {
	router := gin.New()
	router.POST("/upload", MaxUploadSize(10), func(c *gin.Context) {
		body, err := io.ReadAll(c.Request.Body)
		if err != nil {
			t.Errorf("unexpected read error for a body exactly at the limit: %v", err)
		}
		c.JSON(http.StatusOK, gin.H{"bytes": len(body)})
	})

	req := httptest.NewRequest(http.MethodPost, "/upload", bytes.NewReader(make([]byte, 10)))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 for a body exactly at the limit, got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestMaxUploadSize_HandlerReadHitsCapOverLimit(t *testing.T) {
	router := gin.New()
	router.POST("/upload", MaxUploadSize(10), func(c *gin.Context) {
		_, err := io.ReadAll(c.Request.Body)
		var maxBytesErr *http.MaxBytesError
		if errors.As(err, &maxBytesErr) {
			c.JSON(http.StatusRequestEntityTooLarge, gin.H{"error": "too large"})
			return
		}
		c.Status(http.StatusOK)
	})

	req := httptest.NewRequest(http.MethodPost, "/upload", bytes.NewReader(make([]byte, 11)))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("expected the handler's read to hit the byte cap, got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestMaxUploadSize_DoesNotAffectUnrelatedRoute(t *testing.T) {
	router := gin.New()
	router.POST("/capped", MaxUploadSize(1), func(c *gin.Context) { c.Status(http.StatusOK) })
	router.POST("/uncapped", func(c *gin.Context) {
		if _, err := io.ReadAll(c.Request.Body); err != nil {
			t.Errorf("unexpected read error on uncapped route: %v", err)
		}
		c.Status(http.StatusOK)
	})

	req := httptest.NewRequest(http.MethodPost, "/uncapped", bytes.NewReader(make([]byte, 1000)))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected the uncapped route to accept a large body, got %d", rec.Code)
	}
}
