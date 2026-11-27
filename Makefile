# BirdFinder developer commands. Run `make` or `make help` for the list.
#
# Recipes run in Windows PowerShell, so these work the same whether `make`
# is called from PowerShell, cmd or Git Bash.

SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command

ANDROID_HOME ?= C:/Android/sdk
ADB ?= $(ANDROID_HOME)/platform-tools/adb.exe
PACKAGE := com.cundytech.birdfinder
DEBUG_APK := client/android/app/build/outputs/apk/debug/app-debug.apk
RELEASE_APK := client/android/app/build/outputs/apk/release/app-release.apk

# Pick a phone when more than one is attached: make release DEVICE=<serial from `make devices`>
DEVICE ?=
ADB_TARGET := $(if $(DEVICE),-s $(DEVICE),)
ADB_CMD := & "$(ADB)" $(ADB_TARGET)

.DEFAULT_GOAL := help
.PHONY: help setup prebuild build-debug build-release install-debug install-release \
	debug release start launch uninstall logs devices pair connect clean test \
	test-api api sync-model

help: ## List the available commands
	@Select-String -Path Makefile -Pattern '^([a-z-]+):.*?## (.*)$$' | ForEach-Object { '  make {0,-16} {1}' -f $$_.Matches[0].Groups[1].Value, $$_.Matches[0].Groups[2].Value }
	@''
	@'  Typical flows:'
	@'    make release                standalone build on the phone, no laptop needed afterwards'
	@'    make debug, then make start fast JS reloads from Metro while the phone is attached'
	@'    make connect                reattach the phone over wireless debugging'

# --- Setup ------------------------------------------------------------------

setup: ## Install npm dependencies (also applies client/patches)
	cd client; npm install

prebuild: ## Regenerate android/ from app.json and fix its manifest
	cd client; npx expo prebuild --platform android --no-install
	& client/scripts/dedupe-manifest.ps1

sync-model: ## Copy the trained model and labels from model/h5 into the app
	Copy-Item model/h5/bird_classifier_model.onnx, model/h5/labels.json client/assets/ -Force
	@'Copied. Rebuild the app (make release) to ship the new model.'

# --- Build and install ------------------------------------------------------

build-debug: ## Build the debug APK (loads its JS from Metro)
	cd client/android; ./gradlew.bat assembleDebug

build-release: ## Build the standalone release APK (JS and model bundled in)
	cd client/android; ./gradlew.bat assembleRelease

install-debug: ## Install the last debug build on the phone
	$(ADB_CMD) install -r $(DEBUG_APK)

install-release: ## Install the last release build on the phone
	$(ADB_CMD) install -r $(RELEASE_APK)

debug: build-debug install-debug ## Build, install and open the debug app (then run make start)
	$(ADB_CMD) reverse tcp:8081 tcp:8081
	$(ADB_CMD) shell am start -n $(PACKAGE)/.MainActivity
	@'Debug app installed. Run `make start` so it can load its JavaScript.'

release: build-release install-release launch ## Build, install and open the release app

start: ## Start Metro for the debug app
	$(ADB_CMD) reverse tcp:8081 tcp:8081
	cd client; npx expo start

# --- Phone ------------------------------------------------------------------

launch: ## Open the app on the phone
	$(ADB_CMD) shell am start -n $(PACKAGE)/.MainActivity

uninstall: ## Remove the app from the phone
	$(ADB_CMD) uninstall $(PACKAGE)

logs: ## Stream the app's JS logs and crashes (Ctrl+C to stop)
	$(ADB_CMD) logcat -s ReactNativeJS:V AndroidRuntime:E

devices: ## List attached phones
	& "$(ADB)" devices -l

pair: ## Pair with a phone once: make pair PAIR=<ip:port> CODE=<6 digits>
	@if (-not '$(PAIR)' -or -not '$(CODE)') { 'Usage: make pair PAIR=<ip:port> CODE=<code>, both from the phone''s "Pair device with pairing code" popup'; exit 1 }
	& "$(ADB)" pair $(PAIR) $(CODE)

connect: ## Connect to an already-paired phone over wireless debugging
	& client/scripts/connect-phone.ps1 -Adb "$(ADB)"

# --- Housekeeping -----------------------------------------------------------

clean: ## Delete Android build outputs
	cd client/android; ./gradlew.bat clean

test: ## Run the client's Jest tests
	cd client; npm test

test-api: ## Run the Go API tests
	cd api; go test ./...

api: ## Run the optional API server locally (needs the API_KEY env var)
	cd api; go run .
