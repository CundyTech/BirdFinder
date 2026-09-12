# Wraps the Android build/install steps documented in README.md's
# "Building an Android APK locally" section. Assumes the one-time SDK/JDK
# setup described there is already done (ANDROID_HOME, JAVA_HOME, and the
# android/ native project all present) — this doesn't run `expo prebuild`.

ANDROID_DIR := android
APK_DEBUG   := $(ANDROID_DIR)/app/build/outputs/apk/debug/app-debug.apk
APK_RELEASE := $(ANDROID_DIR)/app/build/outputs/apk/release/app-release.apk

.PHONY: help build-debug build-release install-debug install-release clean

help:
	@echo "build-debug     - assemble the debug APK"
	@echo "build-release   - assemble the release APK"
	@echo "install-debug   - build the debug APK, install it, and reverse Metro's port over USB"
	@echo "install-release - build and install the standalone release APK"
	@echo "clean           - gradle clean"

build-debug:
	cd $(ANDROID_DIR) && ./gradlew.bat assembleDebug

build-release:
	cd $(ANDROID_DIR) && ./gradlew.bat assembleRelease

# Debug builds load JS from Metro rather than bundling it, so this also
# reverses Metro's port over the USB cable (harmless if Metro isn't running
# yet — just start `npx expo start` before opening the app on the phone).
install-debug: build-debug
	adb install -r $(APK_DEBUG)
	adb reverse tcp:8081 tcp:8081

install-release: build-release
	adb install -r $(APK_RELEASE)

clean:
	cd $(ANDROID_DIR) && ./gradlew.bat clean
