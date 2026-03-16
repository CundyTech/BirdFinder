from pathlib import Path
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential

base_dir = Path(__file__).resolve().parent
data_dir = (base_dir / '..' / 'images' / 'segmentations').resolve()

if not data_dir.exists():
    print(f"Data directory not found: {data_dir}")
    sys.exit(1)

batch_size = 4
target_size = (224, 224)

train_datagen = ImageDataGenerator(rescale=1.0/255.0)
try:
    train_gen = train_datagen.flow_from_directory(
        str(data_dir),
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
except Exception as e:
    print('Failed to create data generator:', e)
    sys.exit(1)

# Get a single batch
try:
    x_batch, y_batch = next(train_gen)
except Exception as e:
    print('Failed to fetch a batch:', e)
    sys.exit(1)

num_classes = y_batch.shape[1]
input_shape = (224, 224, 3)

# Build a small model matching the main script
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train on the single batch
loss, acc = model.train_on_batch(x_batch, y_batch)
print(f"Smoke train_on_batch result — loss: {loss:.4f}, acc: {acc:.4f}")

# Save smoke model into repository h5 folder
out_dir = (base_dir / '..' / 'h5').resolve()
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / 'bird_classifier_model_smoke.h5'
model.save(str(out_path))
print(f"Saved smoke model to {out_path}")
