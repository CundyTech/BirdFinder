import tensorflow as tf
from PIL import Image
import numpy as np
from os import listdir
from os.path import isdir, join
from pathlib import Path


def preprocess_image(image_path, target_size):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


# Paths: point to model/h5/test and model/h5/images/segmentations
base_dir = Path(__file__).resolve().parent
# Use test image under `model/images/test` and labels under `model/images/segmentations`
image_path = str((base_dir / '..' / 'images' / 'test' / 'american_crow.jpg').resolve())
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

# Get predicted class labels from segmentation folders
labels_dir = str((base_dir / '..' / 'images' / 'segmentations').resolve())
labels = [d for d in listdir(labels_dir) if isdir(join(labels_dir, d))]

labels.sort()

i = np.argmax(predictions)
predicted_class = labels[i] if 0 <= i < len(labels) else 'unknown'

print(f"Predicted class: {predicted_class}")
