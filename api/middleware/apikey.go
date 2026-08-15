package middleware

import (
	"crypto/sha256"
	"crypto/subtle"
	"log"
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

		// Log missing header with request context so we can diagnose 401s.
		if got == "" {
			// Dump incoming headers to help diagnose why the header is missing
			log.Printf("API key missing - remote=%s method=%s path=%s", c.ClientIP(), c.Request.Method, c.Request.URL.Path)
			for name, vals := range c.Request.Header {
				log.Printf("Header: %s=%v", name, vals)
			}
			c.JSON(http.StatusUnauthorized, gin.H{"error": "missing or invalid API key"})
			c.Abort()
			return
		}

		// Compute a non-reversible fingerprint of the provided key for logs.
		gotHash := sha256.Sum256([]byte(got))

		// Constant-time compare to avoid timing attacks.
		if subtle.ConstantTimeCompare([]byte(got), []byte(key)) != 1 {
			// Log the received value length and a fingerprint to help debug truncation or mangling.
			log.Printf("API key invalid - remote=%s method=%s path=%s got_sha256=%x", c.ClientIP(), c.Request.Method, c.Request.URL.Path, gotHash)
			log.Printf("Received API header (raw) length=%d", len(got))
			log.Printf("Received API header (raw) value=%q", got)
			c.JSON(http.StatusUnauthorized, gin.H{"error": "missing or invalid API key"})
			c.Abort()
			return
		}

		// Accepted — log a short acceptance fingerprint for debugging.
		log.Printf("API key accepted - remote=%s method=%s path=%s got_sha256=%x", c.ClientIP(), c.Request.Method, c.Request.URL.Path, gotHash)
		c.Next()
	}
}
