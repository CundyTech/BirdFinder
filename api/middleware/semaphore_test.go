package middleware

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
)

// These tests use a controllable dummy handler rather than a real slow
// operation, so slot-holding duration is driven by test synchronization —
// deterministic and fast regardless of the machine running the tests.

func TestSemaphore_RejectsWhenFull(t *testing.T) {
	sem := NewSemaphore(1, 100*time.Millisecond)

	holding := make(chan struct{})
	release := make(chan struct{})

	router := gin.New()
	router.GET("/slow", sem.Middleware(), func(c *gin.Context) {
		close(holding)
		<-release
		c.Status(http.StatusOK)
	})

	firstDone := make(chan int, 1)
	go func() {
		req := httptest.NewRequest(http.MethodGet, "/slow", nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		firstDone <- rec.Code
	}()

	<-holding // first request now holds the only slot

	req2 := httptest.NewRequest(http.MethodGet, "/slow", nil)
	rec2 := httptest.NewRecorder()
	router.ServeHTTP(rec2, req2)
	if rec2.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503 while the only slot is held, got %d: %s", rec2.Code, rec2.Body.String())
	}
	var payload map[string]string
	if err := json.Unmarshal(rec2.Body.Bytes(), &payload); err != nil {
		t.Fatalf("invalid JSON response: %v", err)
	}
	if payload["error"] == "" {
		t.Error("expected a non-empty error message")
	}

	close(release)
	if code := <-firstDone; code != http.StatusOK {
		t.Fatalf("expected slot-holding request to succeed, got %d", code)
	}
}

func TestSemaphore_FreedSlotAllowsNextRequest(t *testing.T) {
	sem := NewSemaphore(1, 200*time.Millisecond)

	holding := make(chan struct{})
	release := make(chan struct{})

	router := gin.New()
	router.GET("/slow", sem.Middleware(), func(c *gin.Context) {
		select {
		case <-holding: // already signaled once (see below): pass straight through
		default:
			close(holding)
			<-release
		}
		c.Status(http.StatusOK)
	})

	firstDone := make(chan int, 1)
	go func() {
		req := httptest.NewRequest(http.MethodGet, "/slow", nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		firstDone <- rec.Code
	}()

	<-holding
	close(release)
	if code := <-firstDone; code != http.StatusOK {
		t.Fatalf("expected first request to succeed, got %d", code)
	}

	// Slot is free again — a second request should succeed without waiting
	// out the full timeout.
	req2 := httptest.NewRequest(http.MethodGet, "/slow", nil)
	rec2 := httptest.NewRecorder()
	router.ServeHTTP(rec2, req2)
	if rec2.Code != http.StatusOK {
		t.Fatalf("expected second request to succeed once slot freed, got %d", rec2.Code)
	}
}

func TestSemaphore_AllowsUpToN(t *testing.T) {
	const n = 3
	sem := NewSemaphore(n, 200*time.Millisecond)

	release := make(chan struct{})
	var mu sync.Mutex
	inFlight, maxInFlight := 0, 0

	router := gin.New()
	router.GET("/slow", sem.Middleware(), func(c *gin.Context) {
		mu.Lock()
		inFlight++
		if inFlight > maxInFlight {
			maxInFlight = inFlight
		}
		mu.Unlock()

		<-release

		mu.Lock()
		inFlight--
		mu.Unlock()
		c.Status(http.StatusOK)
	})

	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			req := httptest.NewRequest(http.MethodGet, "/slow", nil)
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)
			if rec.Code != http.StatusOK {
				t.Errorf("expected 200 within capacity, got %d", rec.Code)
			}
		}()
	}

	// Give the n goroutines time to all acquire their slots before releasing.
	time.Sleep(50 * time.Millisecond)
	close(release)
	wg.Wait()

	mu.Lock()
	defer mu.Unlock()
	if maxInFlight != n {
		t.Errorf("expected exactly %d concurrent in-flight requests, saw max %d", n, maxInFlight)
	}
}
