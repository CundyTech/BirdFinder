# BirdFinder React Native Client

Simple Expo-based client to take a photo and send it to the BirdFinder API.

Quick start:

1. Install dependencies:

```bash
cd client
npm install
```

2. Start the app:

```bash
npm start
```

3. Configure `API_URL` in `App.js` if you need to use your machine IP (e.g., `http://192.168.x.y:8080/predict`).

Notes:
- The app uses `expo-image-picker` to capture photos and uploads to the `/predict` endpoint.
- For Android emulator, `10.0.2.2` maps to host machine `localhost`.
