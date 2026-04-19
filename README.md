# BirdFinder

A full-stack bird species classification application with a React Native mobile client, Go API server, and TensorFlow/Keras machine learning model.

## Architecture

- **Client**: React Native (Expo) mobile app for taking photos and displaying predictions
- **API**: Go HTTP server that accepts image uploads and forwards to Python prediction service
- **Model**: TensorFlow/Keras CNN model trained on bird species images

## Quick Start

### Prerequisites

- Go 1.19+ (for API server)
- Node.js 18+ and npm (for React Native client)
- Python 3.10 or 3.11 (for ML model)
- Conda (recommended for Python environment management)

### 1. Set up Python Environment

```bash
# Navigate to model directory
cd model

# Create conda environment (recommended)
conda env create -f environment.lock.yml
conda activate birdfinder

# Alternative: pip install
pip install -r build/requirements.txt
```

### 2. Start the API Server

```bash
# From project root or api folder
cd api
go run .
```

The API server will start on `http://localhost:8080`.

### 3. Start the React Native Client

```bash
# In a new terminal, from project root
cd client
npm install
npm start
```

Follow the Expo CLI instructions to run on iOS simulator, Android emulator, or physical device.

### 4. Test the Application

1. Open the mobile app
2. Take a photo of a bird
3. The app will upload the image to the API and display the predicted bird species

## Project Structure

```
BirdFinder/
├── api/                    # Go HTTP API server
│   ├── main.go            # Server implementation
│   └── README.md          # API documentation
├── client/                 # React Native (Expo) mobile app
│   ├── App.js             # Main app component
│   ├── package.json       # Dependencies and scripts
│   └── README.md          # Client setup instructions
└── model/                  # Machine learning components
    ├── build/             # Training and prediction scripts
    │   ├── model_build.py # Full training script
    │   ├── train_smoke.py # Quick training test
    │   ├── model_predict.py # Prediction logic
    │   ├── predict_cli.py # CLI prediction wrapper
    │   └── requirements.txt # Python dependencies
    ├── h5/                # Trained model files
    ├── images/            # Training dataset
    │   └── segmentations/ # Bird species image folders
    ├── environment.yml    # Conda environment spec
    ├── environment.lock.yml # Locked environment versions
    └── readme.md          # Model setup instructions
```

## API Endpoints

- `POST /predict` - Accepts multipart/form-data with `image` field containing a bird photo
- Returns JSON with predicted bird species

## Development

### Training the Model

For a quick test run:
```bash
conda run -n birdfinder python model/build/train_smoke.py
```

For full training:
```bash
conda run -n birdfinder python model/build/model_build.py
```

### Testing Predictions

```bash
conda run -n birdfinder python model/build/predict_cli.py path/to/bird/image.jpg
```

## Configuration

- **API URL**: Configure in `client/App.js` if running on physical device (use your machine's IP instead of localhost)
- **Python Environment**: Ensure the `birdfinder` conda environment is activated when running the API server

## Notes

- For Android emulator testing, use `10.0.2.2` as the host IP to reach localhost on the host machine
- The model is trained on a subset of bird species from the CUB-200-2011 dataset
- CORS is enabled on the API server for development; restrict in production</content>
<parameter name="filePath">c:\Users\DanCu\source\repos\BirdFinder\README.md