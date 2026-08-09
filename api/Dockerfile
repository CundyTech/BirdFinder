# syntax=docker/dockerfile:1
#
# Build from the repo root so this can reach both api/ and model/:
#   docker build -f api/Dockerfile -t ghcr.io/cundytech/birdfinder-api:TAG .
#
# predictHandler (api/main.go) shells out to model/build/predict_cli.py for
# ONNX inference, so this image needs a Python + onnxruntime environment
# alongside the compiled Go binary, not just the binary alone. The two
# COPY destinations below deliberately mirror the repo's own relative-path
# layout (api/main.go resolves ../model/build/predict_cli.py from its CWD;
# predict_cli.py resolves ../h5/*.{onnx,json} from its own file location) so
# both existing relative lookups resolve correctly with zero code changes.

# ---- Stage 1: compile the Go API as a static binary ----
FROM golang:1.25.0-alpine AS builder
WORKDIR /src
COPY api/go.mod api/go.sum ./
RUN go mod download
COPY api/ .
RUN CGO_ENABLED=0 GOOS=linux go build -o /out/birdfinder-api .

# ---- Stage 2: runtime — Python + onnxruntime for predict_cli.py ----
FROM python:3.11-slim-bookworm

COPY model/build/requirements-inference.txt /app/model/build/requirements-inference.txt
RUN pip install --no-cache-dir -r /app/model/build/requirements-inference.txt

COPY model/build/predict_cli.py /app/model/build/predict_cli.py
COPY model/h5/bird_classifier_model.onnx /app/model/h5/bird_classifier_model.onnx
COPY model/h5/labels.json /app/model/h5/labels.json

COPY --from=builder /out/birdfinder-api /app/api/birdfinder-api

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV PYTHON_INTERPRETER=/usr/local/bin/python3
WORKDIR /app/api
EXPOSE 8080
ENTRYPOINT ["./birdfinder-api"]
