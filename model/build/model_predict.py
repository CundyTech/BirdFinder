import tensorflow as tf
from PIL import Image
import numpy as np
from os import listdir
from os.path import isdir, join

def preprocess_image(image_path, target_size):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Preprocess new image
image_path = './model/images/test/american_crow.jpg'
target_size = (224, 224)
preprocessed_image = preprocess_image(image_path, target_size)

# Load the trained model
model = tf.keras.models.load_model('./model/bird_classifier_model.h5')

# Make predictions
predictions = model.predict(preprocessed_image)

# Get predicted class labels
labels = listdir('./model/images/segmentations')
i = np.argmax(predictions)
predicted_class = labels[i]

print(f"Predicted class: {predicted_class}")
