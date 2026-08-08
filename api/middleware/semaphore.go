package middleware

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

// Semaphore bounds concurrent access to a downstream handler to at most n
// callers at a time, rejecting with 503 if a slot doesn't free up within
// wait. Useful for capping concurrency on expensive per-request work (e.g.
// spawning a subprocess) regardless of how many distinct clients are asking.
type Semaphore struct {
	slots chan struct{}
	wait  time.Duration
}

// NewSemaphore builds a semaphore allowing at most n concurrent callers,
// each queueing for up to wait before being rejected.
func NewSemaphore(n int, wait time.Duration) *Semaphore {
	return &Semaphore{slots: make(chan struct{}, n), wait: wait}
}

// Middleware acquires a slot for the duration of the downstream handler,
// responding 503 if none frees up within the configured wait.
func (s *Semaphore) Middleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		select {
		case s.slots <- struct{}{}:
			defer func() { <-s.slots }()
			c.Next()
		case <-time.After(s.wait):
			c.JSON(http.StatusServiceUnavailable, gin.H{"error": "server busy, try again shortly"})
			c.Abort()
		}
	}
}
