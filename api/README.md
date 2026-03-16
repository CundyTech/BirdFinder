# BirdFinder API

Simple Go HTTP API that exposes a `/predict` endpoint. It accepts a multipart/form-data POST with a single file field named `image` and forwards the image to the Python prediction wrapper.

Run (from the project root or from the `api` folder):

1. Ensure you have Python and the `birdfinder` environment available (see the project README).
2. From the `api` folder run:

```
go run .
```

3. Send a POST request to `http://localhost:8080/predict` with form field `image`.

Notes:
- The server calls `../model/build/predict_cli.py` (relative to the `api` folder) — keep that location.
- CORS is enabled for quick testing; lock it down for production.
