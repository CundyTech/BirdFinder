package middleware

import "github.com/gin-gonic/gin"

// CORS allows cross-origin requests only from the given origins. Passing no
// origins disables cross-origin browser access entirely — appropriate here
// since the real client is a mobile app (not subject to CORS at all) and
// there's currently no legitimate web frontend for this API. Add specific
// origins if one is ever introduced.
func CORS(allowedOrigins ...string) gin.HandlerFunc {
	allowed := make(map[string]bool, len(allowedOrigins))
	for _, o := range allowedOrigins {
		allowed[o] = true
	}

	return func(c *gin.Context) {
		origin := c.GetHeader("Origin")
		if origin != "" && allowed[origin] {
			c.Writer.Header().Set("Access-Control-Allow-Origin", origin)
			c.Writer.Header().Set("Vary", "Origin")
			c.Writer.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
			c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, X-API-Key")
		}
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(200)
			return
		}
		c.Next()
	}
}
