#!/usr/bin/env python
"""CLI wrapper for model prediction.

Usage: python predict_cli.py --image /path/to/image.jpg

This script loads the trained model from ../h5 (falls back to the smoke model),
resizes the provided image to the model input size, runs prediction, and prints a
JSON object with `predicted_class` and `scores`.
"""
import argparse
import json
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isdir, join


def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def load_model(base_dir):
    model_path = (base_dir / '..' / 'h5' / 'bird_classifier_model.h5').resolve()
    if not model_path.exists():
        model_path = (base_dir / '..' / 'h5' / 'bird_classifier_model_smoke.h5').resolve()
    model = tf.keras.models.load_model(str(model_path))
    return model, model_path


def get_labels(base_dir):
    labels_dir = str((base_dir / '..' / 'images' / 'segmentations').resolve())
    labels = [d for d in listdir(labels_dir) if isdir(join(labels_dir, d))]
    labels.sort()
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to image file')
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    img_array = preprocess_image(args.image)

    model, model_path = load_model(base_dir)

    preds = model.predict(img_array)

    labels = get_labels(base_dir)

    scores = preds[0].tolist()
    i = int(np.argmax(preds))
    predicted_class = labels[i] if 0 <= i < len(labels) else 'unknown'

    out = {
        'model_path': str(model_path),
        'predicted_class': predicted_class,
        'scores': scores,
    }

    print(json.dumps(out))


if __name__ == '__main__':
    main()
