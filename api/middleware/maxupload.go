package middleware

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

// MaxUploadSize caps how much of the request body a downstream handler can
// read, via http.MaxBytesReader. It only installs the limit — detecting
// that a read actually hit the cap (http.MaxBytesError) and turning that
// into a response is the handler's job, since only the handler knows how
// and when it reads the body (multipart form, raw bytes, streaming, etc.).
func MaxUploadSize(maxBytes int64) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxBytes)
		c.Next()
	}
}
