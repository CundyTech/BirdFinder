package main

import (
	"bytes"
	"fmt"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
)

// TestTrustedProxies_IgnoresSpoofedForwardedForHeader proves the rate
// limiter can't be bypassed by sending a different X-Forwarded-For on every
// request — SetTrustedProxies(nil) must make Gin ignore that header
// entirely and key off the real connection address instead.
func TestTrustedProxies_IgnoresSpoofedForwardedForHeader(t *testing.T) {
	router := newRouter()

	makeRequest := func(forwardedFor string) int {
		body, contentType := multipartBody(t, false, 0) // missing image field: fails fast, no subprocess
		req := httptest.NewRequest(http.MethodPost, "/predict", body)
		req.Header.Set("Content-Type", contentType)
		if forwardedFor != "" {
			req.Header.Set("X-Forwarded-For", forwardedFor)
		}
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		return rec.Code
	}

	// All requests share the same real httptest connection address. If the
	// spoofed header were honored, each of these distinct "IPs" would land
	// in its own bucket and never get rate limited.
	for i := 0; i < predictRateBurst; i++ {
		if code := makeRequest(fmt.Sprintf("10.0.0.%d", i)); code == http.StatusTooManyRequests {
			t.Fatalf("request %d unexpectedly rate limited before exhausting the burst", i)
		}
	}

	if code := makeRequest("10.0.0.99"); code != http.StatusTooManyRequests {
		t.Fatalf("expected the spoofed X-Forwarded-For to be ignored (same real IP should share one bucket), got %d", code)
	}
}

func TestSanitizeForLog(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  string
	}{
		{"plain string unchanged", "photo.jpg", "photo.jpg"},
		{"newline stripped", "evil\n2026-01-01 FAKE LOG LINE", "evil2026-01-01 FAKE LOG LINE"},
		{"carriage return stripped", "a\rb", "ab"},
		{"tab stripped", "a\tb", "ab"},
		{"del char stripped", "a\x7fb", "ab"},
		{"unicode preserved", "café🐦.jpg", "café🐦.jpg"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := sanitizeForLog(tc.input); got != tc.want {
				t.Errorf("sanitizeForLog(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}

// TestPredictHandler_RejectsControlCharactersInFilenameUpstream documents a
// discovery made while testing sanitizeForLog: Go's own mime/multipart
// parser already refuses to parse a Content-Disposition header whose
// filename contains a raw control character (tested here with an ANSI
// escape byte) — the request fails at c.MultipartForm() as a "malformed
// MIME header line" 400, never reaching predictHandler's file-handling code
// at all. So file.Filename can never actually contain a control character
// coming from any real HTTP client; the log-injection vector sanitizeForLog
// guards against isn't reachable through this path today. It's kept as
// defense-in-depth regardless (see TestSanitizeForLog for its own direct,
// unit-level coverage) — cheap insurance against this parsing behavior
// changing, or the sanitizer being reused for a less strictly-parsed field
// later.
func TestPredictHandler_RejectsControlCharactersInFilenameUpstream(t *testing.T) {
	router := newRouter()

	body := &bytes.Buffer{}
	w := multipart.NewWriter(body)
	fw, err := w.CreateFormFile("image", "evil\x1b[31mFAKE ALERT\x1b[0m.jpg")
	if err != nil {
		t.Fatalf("CreateFormFile: %v", err)
	}
	if _, err := fw.Write(jpegLikeContent(100)); err != nil {
		t.Fatalf("write form file: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("close multipart writer: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/predict", body)
	req.Header.Set("Content-Type", w.FormDataContentType())
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected a control character in the filename to be rejected as a malformed request, got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestExtensionForContentType(t *testing.T) {
	cases := map[string]string{
		"image/jpeg":           ".jpg",
		"image/png":            ".png",
		"image/gif":            ".gif",
		"image/webp":           ".webp",
		"image/bmp":            ".img",
		"text/plain":           ".img",
		"application/xml; a=b": ".img",
	}
	for contentType, want := range cases {
		if got := extensionForContentType(contentType); got != want {
			t.Errorf("extensionForContentType(%q) = %q, want %q", contentType, got, want)
		}
	}
}

func TestLimitedWriter_AllowsWritesUpToLimit(t *testing.T) {
	w := &limitedWriter{limit: 10}
	n, err := w.Write([]byte("0123456789"))
	if err != nil {
		t.Fatalf("unexpected error writing exactly at the limit: %v", err)
	}
	if n != 10 {
		t.Errorf("expected 10 bytes written, got %d", n)
	}
	if w.buf.String() != "0123456789" {
		t.Errorf("unexpected buffered content: %q", w.buf.String())
	}
}

func TestLimitedWriter_RejectsWriteOverLimit(t *testing.T) {
	w := &limitedWriter{limit: 10}
	if _, err := w.Write([]byte("01234567890")); err == nil {
		t.Fatal("expected an error writing past the limit, got nil")
	}
}

func TestLimitedWriter_RejectsCumulativeWritesOverLimit(t *testing.T) {
	w := &limitedWriter{limit: 10}
	if _, err := w.Write([]byte("12345")); err != nil {
		t.Fatalf("unexpected error on first write: %v", err)
	}
	if _, err := w.Write([]byte("12345")); err != nil {
		t.Fatalf("unexpected error on second write reaching exactly the limit: %v", err)
	}
	if _, err := w.Write([]byte("1")); err == nil {
		t.Fatal("expected an error once cumulative writes exceed the limit, got nil")
	}
}

func TestEnvOrDefault(t *testing.T) {
	const key = "BIRDFINDER_TEST_ENV_OR_DEFAULT"

	os.Unsetenv(key)
	if got := envOrDefault(key, "fallback"); got != "fallback" {
		t.Errorf("expected fallback when unset, got %q", got)
	}

	os.Setenv(key, "override")
	defer os.Unsetenv(key)
	if got := envOrDefault(key, "fallback"); got != "override" {
		t.Errorf("expected env override, got %q", got)
	}
}

// detectImageContentType must treat a fully-consumed reader as a normal,
// non-fatal case (io.ReadFull returns io.EOF for an empty reader), not an
// error — it's already covered by TestDetectImageContentType's "empty" case
// in main_test.go, but this pins the specific error-handling behavior.
func TestDetectImageContentType_TreatsShortReadAsNonFatal(t *testing.T) {
	_, _, err := detectImageContentType(bytes.NewReader(nil))
	if err != nil {
		t.Fatalf("expected no error for an empty reader (io.EOF should be treated as non-fatal), got %v", err)
	}
}
