"""Trains the UK bird classifier.

Run download_uk_bird_images.py first to populate ../images/uk_birds — this
script trains on real photos there, not the CUB-200 segmentation masks the
old version pointed at (those are silhouettes with no color/texture info,
useless for a real classifier).

Uses transfer learning on MobileNetV2 rather than a from-scratch CNN: with
only ~100-200 images per class, a from-scratch network doesn't have enough
data to learn useful visual features on its own. Starting from ImageNet
weights and fine-tuning gets far better results at this data scale.

Exports both a Keras .h5 and an ONNX model. The ONNX export is what the app
actually needs — the deployment target for inference is Windows ARM64, where
TensorFlow has no installable wheel, but onnxruntime does.
"""
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Config ---
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
HEAD_EPOCHS = 20
FINE_TUNE_EPOCHS = 10
VALIDATION_SPLIT = 0.2
FINE_TUNE_UNFREEZE_LAST_N_LAYERS = 30

base_dir = Path(__file__).resolve().parent
data_dir = (base_dir / '..' / 'images' / 'uk_birds').resolve()
out_dir = (base_dir / '..' / 'h5').resolve()
out_dir.mkdir(parents=True, exist_ok=True)

if not data_dir.exists() or not any(data_dir.iterdir()):
    raise SystemExit(
        f"No training data at {data_dir}.\n"
        f"Run download_uk_bird_images.py first to populate it."
    )

num_classes = len([d for d in data_dir.iterdir() if d.is_dir()])
if num_classes < 2:
    raise SystemExit(f"Found {num_classes} class folder(s) under {data_dir} — need at least 2 to train.")

# --- Data ---
# Two separate generators over the same directory/split/seed so training
# gets augmentation and validation doesn't. Keras's validation_split assigns
# files to a subset deterministically from each class's sorted file list, so
# both generators land on the same train/val boundary regardless of `seed`
# (seed only affects shuffle order within a subset, not the split itself).
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=VALIDATION_SPLIT,
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=VALIDATION_SPLIT,
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42,
)

validation_generator = val_datagen.flow_from_directory(
    data_dir,
    target_size=INPUT_SHAPE[:2],
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42,
)

print(f"Found {train_generator.samples} training images, "
      f"{validation_generator.samples} validation images, {num_classes} classes")

# Persist the exact label order the model was trained with. Inference must
# read this file rather than re-deriving order by listing the data
# directory — that only happens to match if the sort order and directory
# contents are identical at train and inference time, which is fragile.
labels_by_index = sorted(train_generator.class_indices, key=train_generator.class_indices.get)
labels_path = out_dir / 'labels.json'
labels_path.write_text(json.dumps(labels_by_index, indent=2), encoding='utf-8')
print(f"Saved {len(labels_by_index)} labels to {labels_path}")

# --- Model: MobileNetV2 base + custom classification head ---
base_model = MobileNetV2(input_shape=INPUT_SHAPE, include_top=False, weights='imagenet')
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

callbacks = [EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True)]

print("\n=== Stage 1: training classification head (MobileNetV2 base frozen) ===")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
    epochs=HEAD_EPOCHS,
    validation_data=validation_generator,
    validation_steps=max(1, validation_generator.samples // BATCH_SIZE),
    callbacks=callbacks,
)

print("\n=== Stage 2: fine-tuning top layers of MobileNetV2 ===")
base_model.trainable = True
for layer in base_model.layers[:-FINE_TUNE_UNFREEZE_LAST_N_LAYERS]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # small LR — don't wreck pretrained weights
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
history = model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator.samples // BATCH_SIZE),
    epochs=FINE_TUNE_EPOCHS,
    validation_data=validation_generator,
    validation_steps=max(1, validation_generator.samples // BATCH_SIZE),
    callbacks=callbacks,
)

final_val_acc = history.history.get('val_accuracy', [None])[-1]
print(f"\nFinal validation accuracy: {final_val_acc}")

# --- Save ---
h5_path = out_dir / 'bird_classifier_model.h5'
model.save(str(h5_path))
print(f"Saved Keras model to {h5_path}")

try:
    import tf2onnx

    onnx_path = out_dir / 'bird_classifier_model.onnx'
    spec = (tf.TensorSpec((None, *INPUT_SHAPE), tf.float32, name='input'),)
    tf2onnx.convert.from_keras(model, input_signature=spec, output_path=str(onnx_path))
    print(f"Saved ONNX model to {onnx_path}")
except ImportError:
    print("\ntf2onnx not installed — skipping ONNX export.")
    print("Install it with: pip install tf2onnx")
    print("Then re-run this script (or add a standalone conversion step) to produce the .onnx file.")
