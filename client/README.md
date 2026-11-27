# BirdFinder React Native Client

Expo-based client that takes a photo and identifies the bird on the device. No server or network connection is needed.

Quick start:

1. Install dependencies:

```bash
cd client
npm install
```

`npm install` also runs `patch-package` (the `postinstall` script), which applies the fixes in `patches/` to `node_modules`.

2. Start the app:

```bash
npm start
```

or 

```bash
npx expo start --host lan
```

The app uses native modules, so it needs a dev client or a locally built APK (see below). Expo Go can't run it.

## How classification works

`src/services/localInference.js` loads the bundled model (`assets/bird_classifier_model.onnx`) with `onnxruntime-react-native` the first time it's needed. For each photo it:

1. resizes it to 224×224 with `expo-image-manipulator`
2. decodes the JPEG to RGB pixels with `jpeg-js`
3. scales each channel to [-1, 1] and runs the model
4. returns the top 5 species from `assets/labels.json`

`metro.config.js` registers `.onnx` as an asset extension so Metro bundles the model file.

The model and labels are copies of `model/h5/bird_classifier_model.onnx` and `model/h5/labels.json`. After retraining, copy both into `assets/` and rebuild. The label count must match the model's output size, or every prediction throws an error.

The API client (`src/services/api.js`, `API_BASE` and `EXPO_PUBLIC_API_KEY` in `src/config.js`) is still in the codebase, but no screen calls it any more, so `.env` doesn't need setting up to use the app.

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

After re-running prebuild, check `android/app/src/main/AndroidManifest.xml` for duplicated AdMob `meta-data` entries (`OPTIMIZE_AD_LOADING`, `OPTIMIZE_INITIALIZATION`). The ads config plugin can add them a second time, so delete the duplicates.

### onnxruntime patch

`onnxruntime-react-native`'s Android build asks for `com.microsoft.onnxruntime:onnxruntime-android:latest.integration`. Resolving that dynamic version makes Gradle parse Maven metadata, which fails on this setup with `SAXNotRecognizedException: Property 'http://javax.xml.XMLConstants/property/accessExternalSchema' is not recognized`. `patches/onnxruntime-react-native+1.24.3.patch` pins it to `1.24.3`, the npm package's own version. If you upgrade `onnxruntime-react-native`, regenerate the patch with `npx patch-package onnxruntime-react-native` and pin it to the new version.

The release APK is about 230 MB, because it includes onnxruntime's native libraries for every CPU architecture as well as the 10 MB model.

### Build

```bash
cd android
./gradlew.bat assembleDebug      # needs `npx expo start` running for the JS bundle
./gradlew.bat assembleRelease    # JS bundled in, fully standalone
```

Output APKs land in `android/app/build/outputs/apk/debug/app-debug.apk` and `android/app/build/outputs/apk/release/app-release.apk`. Both are signed with the default debug keystore, so they install directly without setting up a real signing key.

### Ads and in-app purchase (Expo Go no longer works once these are wired in)

The app uses `react-native-google-mobile-ads` (rewarded ads for bonus Film) and `react-native-iap` (one-time "unlock forever" purchase). Both are native modules, so from this point on you need the dev client / local APK built above — Expo Go can't load them.

- **AdMob**: `app.json`'s `react-native-google-mobile-ads` plugin entry currently uses Google's public **test** app IDs (`ca-app-pub-3940256099942544~...`), and `src/hooks/useFilmRewardedAd.js` uses `TestIds.REWARDED`. Swap both for real IDs from the AdMob dashboard before release — test ads show real ad UI but never earn real revenue.
  - `react-native-google-mobile-ads` is pinned to `14.7.2` (well behind latest) — versions from 15.x onward pull in Google Play Services/UMP releases built with Kotlin 2.1 metadata, which this project's Kotlin 1.9.25 toolchain can't read. Bumping the whole project's Kotlin Gradle Plugin to 2.1.x isn't an option either — React Native 0.76's own bundled Gradle plugin is hard-pinned to Kotlin 1.9.25 internally (`node_modules/@react-native/gradle-plugin/gradle/libs.versions.toml`) and breaks (`Found interface ... but class was expected`) if a newer Kotlin Gradle Plugin ends up on the same buildscript classpath. Re-check this pin (and re-verify with a real `assembleDebug`, not just `expo export`) next time React Native's bundled Kotlin version moves.
- **IAP**: `react-native-iap` is pinned to `12.16.4` (not the latest major) because newer versions moved to a Nitro-Modules architecture that requires React Native's New Architecture, which this project doesn't have enabled (`newArchEnabled=false` in `android/gradle.properties`). v12 uses the classic bridge and works as-is.
  - `react-native-iap` supports both the Play Store and Amazon Appstore on Android via Gradle product flavors; `client/plugins/withIapStoreFlavor.js` (a local Expo config plugin) pins the app to the Play flavor so the build can resolve. It reapplies automatically on every `expo prebuild`.
  - Create a non-consumable/managed product in Google Play Console (and the App Store equivalent for iOS) with the SKU/product ID in `src/services/purchases.js` (`UNLOCK_SKU`, currently `com.cundytech.birdfinder.unlock_forever`).
  - There's no backend purchase-verification server — `src/hooks/usePurchases.js` trusts the store's purchase-updated event directly once the platform reports it as completed. Fine for a small app; would need server-side receipt verification to be worth hardening against.

### Install on a phone

From the repo root, `make release` builds, installs and opens the release app in one step, and `make debug` does the same for the debug build. Run `make` for the full list, including `prebuild`, which reruns `expo prebuild` and then removes the duplicate AdMob manifest entries for you. The manual steps are below.

1. On the phone: Settings → About phone → tap "Build number" 7 times to enable Developer Options, then Developer options → enable USB debugging.
2. Plug the phone in via USB and accept the "Allow USB debugging?" prompt.
3. From `client/android`:
   ```bash
   adb install -r app/build/outputs/apk/release/app-release.apk
   ```
   (or the debug APK — if using the debug build, also run `adb reverse tcp:8081 tcp:8081` so the phone can reach Metro over the cable).

#### Wireless debugging (Android 11+)

1. Developer options → Wireless debugging → on. Tap **Pair device with pairing code**, then run `adb pair <ip>:<pairing-port> <code>`. You only need to pair once per computer.
2. Run `adb connect <ip>:<port>`, using the **IP address & Port** from the main Wireless debugging screen. This port changes every time wireless debugging restarts, and `adb mdns services` lists the current one.
3. The phone and PC must be able to reach each other directly. Public Wi-Fi such as coffee shops, and networks with client isolation, block this even when both devices show as connected. In that case, connect the laptop to the phone's hotspot. adb usually then finds and connects to the phone automatically.
