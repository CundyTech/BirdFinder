package middleware

import (
	"net/http"
	"sync"

	"github.com/gin-gonic/gin"
	"golang.org/x/time/rate"
)

// IPRateLimiter hands out one token-bucket limiter per client IP, so
// one abusive source can't exhaust the limit for everyone else.
type IPRateLimiter struct {
	mu       sync.Mutex
	limiters map[string]*rate.Limiter
	rps      rate.Limit
	burst    int
}

// NewIPRateLimiter builds a limiter allowing burst requests immediately per
// IP, refilling at rps sustained requests per second thereafter.
func NewIPRateLimiter(rps rate.Limit, burst int) *IPRateLimiter {
	return &IPRateLimiter{
		limiters: make(map[string]*rate.Limiter),
		rps:      rps,
		burst:    burst,
	}
}

func (rl *IPRateLimiter) allow(ip string) bool {
	rl.mu.Lock()
	limiter, ok := rl.limiters[ip]
	if !ok {
		limiter = rate.NewLimiter(rl.rps, rl.burst)
		rl.limiters[ip] = limiter
	}
	rl.mu.Unlock()
	return limiter.Allow()
}

// Middleware rejects requests over the per-IP rate limit with 429.
func (rl *IPRateLimiter) Middleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		if !rl.allow(c.ClientIP()) {
			c.JSON(http.StatusTooManyRequests, gin.H{"error": "rate limit exceeded, try again later"})
			c.Abort()
			return
		}
		c.Next()
	}
}
