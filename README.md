# BirdFinder

A full-stack bird species classification application with a React Native mobile client, Go API server, and TensorFlow/Keras machine learning model.

## Architecture

- **Client**: React Native (Expo) mobile app for taking photos and displaying predictions
- **API**: Go HTTP server that accepts image uploads and forwards them to a Python prediction subprocess
- **Model**: TensorFlow/Keras CNN trained on bird species images, exported to ONNX for inference

## Quick Start

### Prerequisites

- Go 1.25+ (for the API server)
- Node.js 18+ and npm (for the React Native client)
- Python 3.10 or 3.11 (for the ML model)
- Conda (recommended for Python environment management)

### 1. Set up the Python environment

```bash
cd model
conda env create -f environment.lock.yml
conda activate birdfinder

# Alternative: pip install
pip install -r build/requirements.txt
```

### 2. Start the API server

```bash
cd api
$env:API_KEY = "choose-a-long-random-value"   # required — the server refuses to start without it
go run .
```

The API server starts on `http://localhost:8080`. See [api/README.md](api/README.md) for the full list of environment variables and the hardening built into `/predict` (rate limiting, upload size caps, content-type validation, etc).

### 3. Start the React Native client

```bash
cd client
npm install
cp .env.example .env   # then edit .env: EXPO_PUBLIC_API_KEY must match the server's API_KEY
npm start
```

Follow the Expo CLI instructions to run on iOS simulator, Android emulator, or a physical device. See [client/README.md](client/README.md) for building a standalone Android APK.

### 4. Test the application

1. Open the mobile app
2. Take a photo of a bird
3. The app uploads the image to the API and displays the predicted species

## Project Structure

```
BirdFinder/
├── api/                        # Go HTTP API server
│   ├── main.go                 # Server, routing, predict handler
│   ├── middleware/             # API key auth, rate limiting, upload size cap, CORS, etc.
│   ├── Dockerfile              # Multi-stage build: Go binary + Python/onnxruntime runtime
│   └── README.md
├── client/                     # React Native (Expo) mobile app
│   ├── App.js
│   ├── src/                    # Screens, components, Redux store, API client, config
│   ├── android/                # Generated native project (expo prebuild)
│   └── README.md
├── model/                      # Machine learning components
│   ├── build/                  # Training and prediction scripts
│   │   ├── model_build.py      # Full training script
│   │   ├── predict_cli.py      # CLI prediction wrapper (what the API actually invokes)
│   │   └── requirements*.txt   # Python dependencies (training vs. inference)
│   ├── h5/                     # Trained model files (.h5, .onnx, labels.json)
│   ├── images/                 # Training dataset
│   └── readme.md
└── .github/workflows/           # CI: test the API, build & push its image to GHCR
```

## API Endpoints

- `GET /health` — unauthenticated health check
- `POST /predict` — multipart/form-data with an `image` field; requires an `X-API-Key` header matching the server's `API_KEY`. Returns JSON with the predicted bird species.

## Configuration

- **API key**: required by the server (`API_KEY` env var) and the client (`EXPO_PUBLIC_API_KEY` in `client/.env`) — see [api/README.md](api/README.md).
- **API base URL**: `API_BASE` in `client/src/config.js` — set this to your machine's IP if running on a physical device.
- **Python interpreter**: `PYTHON_INTERPRETER` env var on the API server, if it isn't at the default dev-machine path — see [api/README.md](api/README.md).

## Notes

- For Android emulator testing, use `10.0.2.2` as the host IP to reach `localhost` on the host machine.
- The model is trained on UK bird species photos pulled from GBIF — see [model/readme.md](model/readme.md).
- `/predict` is rate-limited, capped in concurrency and upload size, and validates uploaded content is actually an image before running inference — see [api/README.md](api/README.md).
