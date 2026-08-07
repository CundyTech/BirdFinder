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

or 

```bash
npx expo start --host lan
```

3. Configure `API_URL` in `App.js` if you need to use your machine IP (e.g., `http://192.168.x.y:8080/predict`).

Notes:
- The app uses `expo-image-picker` to capture photos and uploads to the `/predict` endpoint.
- For Android emulator, `10.0.2.2` maps to host machine `localhost`.

## Building an Android APK locally

This builds a real `.apk` on your machine via `expo prebuild` + Gradle — no Expo account or EAS cloud build needed (`eas build --local` isn't supported on native Windows).

### One-time setup

1. Install the Android SDK command-line tools (from https://developer.android.com/studio#command-tools) into a folder such as `C:\Android\sdk\cmdline-tools\latest`, then install the required packages:
   ```bash
   sdkmanager.bat "platform-tools" "platforms;android-35" "build-tools;35.0.0"
   sdkmanager.bat --licenses
   ```
2. Set `ANDROID_HOME` to the SDK folder and add `platform-tools` and `cmdline-tools\latest\bin` to `PATH`, persistently (e.g. via `[Environment]::SetEnvironmentVariable`).
3. Install a JDK 17. On Windows ARM64, use Microsoft's build (Gradle's automatic toolchain download doesn't support ARM64 Windows):
   ```powershell
   winget install --id Microsoft.OpenJDK.17
   ```

### Generate the native project

```bash
cd client
npx expo prebuild --platform android --non-interactive
```

This creates `client/android/`. Two machine-specific settings are needed that live outside source control:

- `android/local.properties` (gitignored) — since `ANDROID_HOME` isn't always visible to the Gradle process. Create it with:
  ```properties
  sdk.dir=C:/Android/sdk
  ```
- `<user home>/.gradle/gradle.properties` — e.g. `C:\Users\<you>\.gradle\gradle.properties`. Needed if Gradle doesn't already default to a JDK 17. This is Gradle's user-level config file (outside the project), not the project's own `gradle.properties`, so it never needs committing and doesn't need to match another machine's path:
  ```properties
  org.gradle.java.home=C:/Program Files/Microsoft/jdk-17.0.20.8-hotspot
  ```

The Gradle wrapper is pinned to Gradle 8.12 in `android/gradle/wrapper/gradle-wrapper.properties`. Don't change this without checking compatibility first — Gradle 8.7–8.11 has a Windows-specific bug that breaks the build (`Could not move temporary workspace ... to immutable location`), while Gradle 8.13+ is newer than the Android Gradle Plugin version React Native bundles and breaks in a different way (`Could not get unknown property 'release' for SoftwareComponent container`). 8.12 is the version that works with the AGP version currently bundled by React Native.

### Build

```bash
cd android
./gradlew.bat assembleDebug      # needs `npx expo start` running for the JS bundle
./gradlew.bat assembleRelease    # JS bundled in, fully standalone
```

Output APKs land in `android/app/build/outputs/apk/debug/app-debug.apk` and `android/app/build/outputs/apk/release/app-release.apk`. Both are signed with the default debug keystore, so they install directly without setting up a real signing key.

### Install on a phone

1. On the phone: Settings → About phone → tap "Build number" 7 times to enable Developer Options, then Developer options → enable USB debugging.
2. Plug the phone in via USB and accept the "Allow USB debugging?" prompt.
3. From `client/android`:
   ```bash
   adb install -r app/build/outputs/apk/release/app-release.apk
   ```
   (or the debug APK — if using the debug build, also run `adb reverse tcp:8081 tcp:8081` so the phone can reach Metro over the cable).
