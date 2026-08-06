"""Training-side sanity check: loads the trained .h5 via TensorFlow and runs
one real prediction against a known test image. For a quick "did training
actually produce something sensible" check right after model_build.py,
before bothering to export/copy anything to the ARM64 deployment host.

For the actual inference path the running app uses, see predict_cli.py
(ONNX Runtime, not TensorFlow — see model/readme.md).
"""
import tensorflow as tf
from PIL import Image
import numpy as np
from os import listdir
from os.path import isdir, join
from pathlib import Path


def preprocess_image(image_path, target_size):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img).astype(np.float32)
    # MobileNetV2's preprocess_input scales pixels to [-1, 1] — must match model_build.py exactly.
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


base_dir = Path(__file__).resolve().parent
# Mallard is one of the trained classes, so this is a real sanity check.
image_path = str((base_dir / '..' / 'images' / 'test' / 'mallard.jpg').resolve())
target_size = (224, 224)
preprocessed_image = preprocess_image(image_path, target_size)

# Load the trained model from repository `h5` folder
model_path = (base_dir / '..' / 'h5' / 'bird_classifier_model.h5').resolve()
if not model_path.exists():
    # fallback to smoke model
    model_path = (base_dir / '..' / 'h5' / 'bird_classifier_model_smoke.h5').resolve()
model = tf.keras.models.load_model(str(model_path))
print(f"Loaded model from {model_path}")

# Make predictions
predictions = model.predict(preprocessed_image)

# Get predicted class labels from the training data directory
labels_dir = str((base_dir / '..' / 'images' / 'uk_birds').resolve())
labels = [d for d in listdir(labels_dir) if isdir(join(labels_dir, d))]

labels.sort()

i = np.argmax(predictions)
predicted_class = labels[i] if 0 <= i < len(labels) else 'unknown'

print(f"Predicted class: {predicted_class}")
