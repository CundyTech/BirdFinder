#!/usr/bin/env python
"""CLI wrapper for model prediction.

Usage: python predict_cli.py --image /path/to/image.jpg

Loads the trained model from ../h5/bird_classifier_model.onnx and its label
order from ../h5/labels.json, runs real inference via onnxruntime, and
prints a JSON object with `predicted_class`, `scores`, and `top_predictions`.

Runs on ONNX rather than TensorFlow directly because this script executes on
Windows ARM64 (see api/main.go), where TensorFlow has no installable wheel at
all. The model is trained elsewhere (see model/readme.md) and exported to
ONNX for exactly this reason — onnxruntime does ship a native win_arm64
wheel.
"""
import argparse
import io
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

TOP_N = 5
TARGET_SIZE = (224, 224)


def preprocess_image(image_path, target_size=TARGET_SIZE):
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception:
        # Some uploads arrive without a reliable extension/format hint —
        # retry via BytesIO rather than trusting the file extension.
        with open(image_path, 'rb') as f:
            img_data = f.read()
        img = Image.open(io.BytesIO(img_data)).convert('RGB')

    img = img.resize(target_size)
    img_array = np.array(img).astype(np.float32)
    # MobileNetV2's preprocess_input scales pixels to [-1, 1] — must match model_build.py exactly.
    img_array = (img_array / 127.5) - 1.0
    return np.expand_dims(img_array, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to image file')
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    h5_dir = (base_dir / '..' / 'h5').resolve()
    model_path = h5_dir / 'bird_classifier_model.onnx'
    labels_path = h5_dir / 'labels.json'

    if not model_path.exists() or not labels_path.exists():
        print(json.dumps({
            'error': f'No trained model found (expected {model_path} and {labels_path}). '
                     f'See model/readme.md to train one and copy both files here.',
            'predicted_class': 'model_not_trained',
            'scores': [],
        }))
        return

    try:
        img_array = preprocess_image(args.image)
    except Exception as e:
        print(json.dumps({
            'error': f'Failed to process image: {e}',
            'predicted_class': 'error_processing_image',
            'scores': [],
        }))
        return

    labels = json.loads(labels_path.read_text(encoding='utf-8'))

    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    scores = session.run([output_name], {input_name: img_array})[0][0].tolist()

    i = int(np.argmax(scores))
    predicted_class = labels[i] if 0 <= i < len(labels) else 'unknown'

    ranked = sorted(zip(labels, scores), key=lambda pair: pair[1], reverse=True)
    top_predictions = [{'class': label, 'score': score} for label, score in ranked[:TOP_N]]

    out = {
        'model_path': str(model_path),
        'predicted_class': predicted_class,
        'scores': scores,
        'top_predictions': top_predictions,
    }

    print(json.dumps(out))


if __name__ == '__main__':
    main()
