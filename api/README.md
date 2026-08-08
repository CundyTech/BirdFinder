# BirdFinder API

Simple Go HTTP API that exposes a `/predict` endpoint. It accepts a multipart/form-data POST with a single file field named `image` and forwards the image to the Python prediction wrapper.

Run (from the project root or from the `api` folder):

1. Ensure you have Python and the `birdfinder` environment available (see the project README).
2. Set the required API key (the server refuses to start without it):

```powershell
$env:API_KEY = "choose-a-long-random-value"
```

3. From the `api` folder run:

```
go run .
```

4. Send a POST request to `http://localhost:8080/predict` with form field `image` and header `X-API-Key: <same value as above>`.

Environment variables:
- `API_KEY` (required) — clients must send this in the `X-API-Key` header on `/predict`. Not real client attestation (it's extractable from anyone who unpacks the mobile app) — it just keeps casual/opportunistic traffic off the endpoint. `/health` and `/` don't require it.
- `PYTHON_INTERPRETER` (optional) — path to the Python interpreter used to run `predict_cli.py`. Defaults to this project's own dev/deployment machine path; set this on any other host.

Notes:
- The server calls `../model/build/predict_cli.py` (relative to the `api` folder) — keep that location.
- CORS defaults to denying all cross-origin browser access (the real client is a mobile app, not a browser — see `middleware.CORS`). Pass allowed origins to `middleware.CORS(...)` in `main.go` if a web frontend is ever added.
- `/predict` is also protected by per-IP rate limiting, a concurrent-subprocess cap, an upload size cap, and content-type sniffing — see `middleware/` and `main.go`.
