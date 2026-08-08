package middleware

import (
	"crypto/subtle"
	"net/http"

	"github.com/gin-gonic/gin"
)

// APIKeyHeader is the header clients must send the configured key in.
const APIKeyHeader = "X-API-Key"

// APIKey rejects requests that don't present the expected key in the
// X-API-Key header. This isn't strong client attestation — any secret
// embedded in a mobile app can be extracted by whoever pulls the app apart
// — it just keeps casual/opportunistic traffic off the endpoint.
//
// Uses a constant-time comparison so a wrong guess can't be narrowed down
// by measuring how long the comparison took to fail.
func APIKey(key string) gin.HandlerFunc {
	return func(c *gin.Context) {
		got := c.GetHeader(APIKeyHeader)
		if got == "" || subtle.ConstantTimeCompare([]byte(got), []byte(key)) != 1 {
			c.JSON(http.StatusUnauthorized, gin.H{"error": "missing or invalid API key"})
			c.Abort()
			return
		}
		c.Next()
	}
}
