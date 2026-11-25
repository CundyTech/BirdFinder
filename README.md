# BirdFinder

A bird species classification application with a React Native mobile client, a TensorFlow/Keras machine learning model, and an optional Go API server.

## Architecture

- **Client**: React Native (Expo) mobile app that takes a photo and classifies it on the device, using the bundled ONNX model via `onnxruntime-react-native`. It works offline and needs no server.
- **Model**: TensorFlow/Keras CNN trained on bird species images, exported to ONNX for inference
- **API** (optional): Go HTTP server that accepts image uploads and forwards them to a Python prediction subprocess. The mobile client no longer uses it, but it still works as a standalone prediction service.

## Quick Start

### Prerequisites

- Node.js 18+ and npm (for the React Native client)
- Python 3.10 or 3.11 (only for training the model or running the API)
- Conda (recommended for Python environment management)
- Go 1.25+ (only for the optional API server)

### 1. Run the React Native client

```bash
cd client
npm install    # also applies the patches in client/patches via postinstall
npm start
```

The app needs native modules (onnxruntime, ads, in-app purchases), so it runs in a dev client or a locally built APK, not Expo Go. See [client/README.md](client/README.md) for building and installing an Android APK.

### 2. Test the application

1. Open the mobile app
2. Take a photo of a bird
3. The app runs the model on the phone and displays the predicted species

### 3. (Optional) Train the model or run the API

The rest of this section is only needed for retraining the model or running the server-side prediction API.

#### Set up the Python environment

```bash
cd model
conda env create -f environment.lock.yml
conda activate birdfinder

# Alternative: pip install
pip install -r build/requirements.txt
```

#### Start the API server

```bash
cd api
$env:API_KEY = "choose-a-long-random-value"   # required — the server refuses to start without it
go run .
```

The API server starts on `http://localhost:8080`. See [api/README.md](api/README.md) for the full list of environment variables and the hardening built into `/predict` (rate limiting, upload size caps, content-type validation, etc).

#### Updating the model in the app

The client bundles its own copy of the model: `client/assets/bird_classifier_model.onnx` and `client/assets/labels.json`, identical to the files in `model/h5/`. After retraining, copy both files into `client/assets/` and rebuild the app. The two must stay in step, because the app checks that the model outputs one score per label.

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
│   ├── assets/                 # Bundled ONNX model + labels (copied from model/h5)
│   ├── src/                    # Screens, components, Redux store, on-device inference, config
│   ├── android/                # Generated native project (expo prebuild)
│   ├── patches/                # patch-package fixes for node_modules, applied on npm install
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

## Optional API server

The mobile app doesn't use this server. It remains available as a standalone prediction service.

### Endpoints

- `GET /health` — unauthenticated health check
- `POST /predict` — multipart/form-data with an `image` field; requires an `X-API-Key` header matching the server's `API_KEY`. Returns JSON with the predicted bird species.

`/predict` is rate-limited, capped in concurrency and upload size, and validates uploaded content is actually an image before running inference.

### Configuration

- **`API_KEY`**: required; the server refuses to start without it.
- **`PYTHON_INTERPRETER`**: set this if Python isn't at the default dev-machine path.

See [api/README.md](api/README.md) for the full list of environment variables.

## Notes

- The model is trained on UK bird species photos pulled from GBIF — see [model/readme.md](model/readme.md).
