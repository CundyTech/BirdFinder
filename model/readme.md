## TensorFlow compatibility

	- Use Python 3.10 or 3.11 x86_64 (most TensorFlow wheels target these versions).
	- Install via `conda`/`conda-forge` which provides more platform builds.
	- See the official TensorFlow install guide for platform-specific instructions: https://www.tensorflow.org/install
**Project**

 - **Description:** UK bird species classifier. Trained with Keras/TensorFlow, deployed for inference via ONNX Runtime (the two run on different machines — see "Deploying a trained model" below). Models and exports are kept under `model/h5`.

**Quick Install (recommended — conda/conda-forge)**

 - **Create env:** Run this to create a reproducible environment from the lockfile:

```powershell
conda env create -f environment.lock.yml
conda activate birdfinder
python -m pip install --upgrade pip
```

 - **Alternative (conda + pip):** If `tensorflow` is not available on conda for your platform, install the other packages with conda and then pip-install numpy/tensorflow as needed:

```powershell
conda create -n birdfinder python=3.11 -c conda-forge -y
conda activate birdfinder
conda install -c conda-forge scipy pillow -y
python -m pip install --no-deps -r model/build/requirements.txt
```

**Files of interest**

 - **Requirements (training):** [model/build/requirements.txt](model/build/requirements.txt) (pinned versions, needs TensorFlow — see ARM64 note below).
 - **Requirements (inference):** [model/build/requirements-inference.txt](model/build/requirements-inference.txt) — what the ARM64 deployment host needs instead (`onnxruntime`, no TensorFlow).
 - **Conda env (solved):** [environment.lock.yml](environment.lock.yml) — exact solved versions for reproducibility.
 - **Image downloader:** [model/build/download_uk_bird_images.py](model/build/download_uk_bird_images.py) — pulls real, licensed UK bird photos from GBIF.
 - **Training script:** [model/build/model_build.py](model/build/model_build.py) — supports `--quick` for a fast, reduced-scope validation run before committing to the full one.
 - **Smoke training:** [model/build/train_smoke.py](model/build/train_smoke.py)
 - **Training-side sanity check:** [model/build/model_predict.py](model/build/model_predict.py) — loads the `.h5` and runs one real TensorFlow prediction against a known test image (`images/test/mallard.jpg`, one of the trained classes). Quick check right after training, before bothering to export/copy anything.
 - **Production inference:** [model/build/predict_cli.py](model/build/predict_cli.py) — what the running app actually calls (invoked by `api/main.go`). Loads `bird_classifier_model.onnx` + `labels.json` and runs real inference via `onnxruntime`. Runs on the ARM64 deployment host, not the training environment — see "Deploying a trained model" below.

**How to train**

 - **1. Download training images** (standard library only, no TensorFlow needed — can run anywhere):

```powershell
python model\build\download_uk_bird_images.py
```

 - **2. Quick smoke run (single batch, confirms the environment works):**

```powershell
conda run -n birdfinder python model\build\train_smoke.py
```

 - **3. Validate the full pipeline on a reduced scope** (few epochs, capped steps — output gets a
   `_quick` suffix so it can't be mistaken for a real model. Native Windows TensorFlow is CPU-only
   as of 2.11+ regardless of GPU hardware present, so this step is worth doing before committing to
   the much longer full run):

```powershell
conda run -n birdfinder python model\build\model_build.py --quick
```

 - **4. Full training** (transfer learning on MobileNetV2; exports `.h5`, `.onnx`, and `labels.json`
   recording the exact class order):

```powershell
conda run -n birdfinder python model\build\model_build.py
```

 - **5. Sanity-check the result** (optional, still on the training machine — loads the real `.h5`
   via TensorFlow and predicts a known test image):

```powershell
conda run -n birdfinder python model\build\model_predict.py
```

**Deploying a trained model**

The app's actual inference (`predict_cli.py`, invoked by `api/main.go`) runs on Windows ARM64, where
TensorFlow has no installable wheel at all — that's the entire reason training exports to ONNX
instead of just shipping the `.h5`. Confirmed working end to end (real inference, ~0.5s per
prediction, well under the Go API's 30s timeout).

 - **1. Copy two files** from the training machine to `model/h5/` on the ARM64 deployment machine:
   - `bird_classifier_model.onnx`
   - `labels.json`

   (the `.h5` isn't needed here — it's only useful for further training)

 - **2. Install the inference-side dependencies**, using the *same* Python interpreter
   `api/main.go` invokes by default. Override it via the `PYTHON_INTERPRETER` env var if
   this deployment machine's interpreter lives somewhere else:

```powershell
C:\Users\<you>\AppData\Local\Programs\Python\Python311-arm64\python.exe -m pip install -r model\build\requirements-inference.txt
```

   (`onnxruntime` ships a native `win_arm64` wheel — confirmed working, no CUDA/TensorFlow needed.)

 - **3. Test it directly**, exactly as the Go API calls it:

```powershell
C:\Users\<you>\AppData\Local\Programs\Python\Python311-arm64\python.exe model\build\predict_cli.py --image path\to\some\photo.jpg
```

   Prints a JSON object with `predicted_class`, `scores`, and `top_predictions`. If
   `bird_classifier_model.onnx` or `labels.json` aren't found in `model/h5/`, it prints a clear JSON
   error explaining what's missing instead of crashing.

   Alternatively, `api/Dockerfile` bundles the interpreter, these dependencies, and the two
   model artifacts above into a single container image — see [api/README.md](../api/README.md)
   for the containerized path, which avoids needing a matching interpreter on the deployment
   host at all.

**Model storage & data layout**

 - **Saved models:** `model/h5/` — `bird_classifier_model.h5`, `bird_classifier_model.onnx`,
   `labels.json` (exact class order — inference must read this, not re-derive it from folder
   listing), and the smoke model `bird_classifier_model_smoke.h5`. `--quick` runs write to
   `bird_classifier_model_quick.{h5,onnx}` instead, so they can never be mistaken for a real model.
 - **Training images:** `model/images/uk_birds/` — one subfolder per species, populated by
   `download_uk_bird_images.py`, plus a `manifest.csv` recording each image's source URL/license/attribution.
 - **Test images:** `model/images/test/` — `mallard.jpg` (Mallard is a trained class, useful for a
   real sanity check) and `american_crow.jpg` (not a trained class — not useful as a sanity check
   against this model, kept for reference).

**Notes & troubleshooting**

 - **TensorFlow compatibility:** TensorFlow wheels may not be available for very new Python versions or certain architectures (e.g., Windows ARM64). If `pip install tensorflow` fails, use the conda-forge pathway or run inside Docker/WSL2, or train on a separate x86_64 machine/VM entirely. See https://www.tensorflow.org/install for platform specifics.
 - **ARM64 deployment target:** the Go API (`api/main.go`) invokes `predict_cli.py` on a Windows ARM64 Python interpreter, where TensorFlow has no installable wheel at all. This is why training exports to ONNX and inference runs on `onnxruntime` instead — see "Deploying a trained model" above.
 - **Preprocessing must match between training and inference:** MobileNetV2 preprocessing scales
   pixels to `[-1, 1]` (`(x / 127.5) - 1.0`), not the more common `[0, 1]`. `model_build.py`,
   `model_predict.py`, and `predict_cli.py` all need to agree on this — if you change one, change
   all three, or predictions will be quietly wrong even with a perfectly good model.
 - **If `conda` is not on PATH:** restart your shell after installing Miniforge/Anaconda, or run the full path to `conda.exe` (e.g. `C:\Users\<you>\miniforge3\Scripts\conda.exe`).
 - **Reproducibility:** Use [environment.lock.yml](environment.lock.yml) to recreate the exact environment used during development.
