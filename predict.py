import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the saved model once when the application starts
# This ensures it's not reloaded every time a prediction is made.
try:
    model = tf.keras.models.load_model('deepfake_detector.h5')
    print("Deepfake Detector model loaded successfully from predict.py.")
except Exception as e:
    print(f"Error loading model from predict.py: {e}")
    model = None


def predict_image(image_path):
    """
    Loads an image, preprocesses it, and makes a prediction using the
    globally loaded model.

    Args:
        image_path (str): The path to the image file.

    Returns:
        float: The prediction score (0-1), or None if the model isn't loaded.
    """
    if not model:
        return None  # Return None if model is not loaded

    IMAGE_SIZE = (150, 150)

    img = image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    return prediction[0][0]