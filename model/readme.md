## TensorFlow compatibility

	- Use Python 3.10 or 3.11 x86_64 (most TensorFlow wheels target these versions).
	- Install via `conda`/`conda-forge` which provides more platform builds.
	- See the official TensorFlow install guide for platform-specific instructions: https://www.tensorflow.org/install
**Project**

 - **Description:** Bird species classifier (training + prediction) using Keras/TensorFlow. Models and exports are kept under the `model` folder and trained models are saved to `model/h5`.

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
 - **Requirements (inference):** [model/build/requirements-inference.txt](model/build/requirements-inference.txt) — what the ARM64 deployment host needs instead (onnxruntime, no TensorFlow).
 - **Conda env (solved):** [environment.lock.yml](environment.lock.yml) — exact solved versions for reproducibility.
 - **Image downloader:** [model/build/download_uk_bird_images.py](model/build/download_uk_bird_images.py) — pulls real, licensed UK bird photos from GBIF.
 - **Training script:** [model/build/model_build.py](model/build/model_build.py)
 - **Smoke training:** [model/build/train_smoke.py](model/build/train_smoke.py)
 - **Prediction script:** [model/build/model_predict.py](model/build/model_predict.py) — real TensorFlow inference; not what the running app currently uses (see `predict_cli.py`, which is still mocked pending a trained model).

**How to train**

 - **1. Download training images** (standard library only, no TensorFlow needed — can run anywhere):

```powershell
python model\build\download_uk_bird_images.py
```

 - **2. Quick smoke run (single batch, confirms the environment works):**

```powershell
conda run -n birdfinder python model\build\train_smoke.py
```

 - **3. Full training** (transfer learning on MobileNetV2; exports both `.h5` and `.onnx`):

```powershell
conda run -n birdfinder python model\build\model_build.py
```

**How to predict**

 - **Run the prediction script** (will load `model/h5/bird_classifier_model.h5` or the smoke model if present):

```powershell
conda run -n birdfinder python model\build\model_predict.py
```

**Model storage & data layout**

 - **Saved models:** `model/h5/` — `bird_classifier_model.h5`, `bird_classifier_model.onnx`, `labels.json` (exact class order — inference must read this, not re-derive it from folder listing), and the smoke model `bird_classifier_model_smoke.h5`.
 - **Training images:** `model/images/uk_birds/` — one subfolder per species, populated by `download_uk_bird_images.py`, plus a `manifest.csv` recording each image's source URL/license/attribution.
 - **Test images:** `model/images/test/` (example `american_crow.jpg`).

**Notes & troubleshooting**

 - **TensorFlow compatibility:** TensorFlow wheels may not be available for very new Python versions or certain architectures (e.g., Windows ARM64). If `pip install tensorflow` fails, use the conda-forge pathway or run inside Docker/WSL2, or train on a separate x86_64 machine/VM entirely. See https://www.tensorflow.org/install for platform specifics.
 - **ARM64 deployment target:** the Go API (`api/main.go`) invokes `predict_cli.py` on a Windows ARM64 Python interpreter, where TensorFlow has no installable wheel at all. Train elsewhere, export to ONNX (the training script does this automatically), and run inference on ARM64 with `onnxruntime` (confirmed to have a native `win_arm64` wheel) instead.
 - **If `conda` is not on PATH:** restart your shell after installing Miniforge/Anaconda, or run the full path to `conda.exe` (e.g. `C:\Users\<you>\miniforge3\Scripts\conda.exe`).
 - **Reproducibility:** Use [environment.lock.yml](environment.lock.yml) to recreate the exact environment used during development.

If you want, I can also add a short `Makefile` or PowerShell script to automate create/train/predict steps.
