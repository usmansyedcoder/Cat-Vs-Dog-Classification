import tensorflow as tf
import sys
import os
from utils import load_and_preprocess_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/cat_dog_model.h5")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(img_path):
    img = load_and_preprocess_image(img_path)
    prediction = model.predict(img)[0][0]
    if prediction > 0.5:
        return "Dog 🐶"
    else:
        return "Cat 🐱"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        img_path = sys.argv[1]
        result = predict_image(img_path)
        print(f"Prediction: {result}")
