import os
import cv2
import numpy as np

IMG_SIZE = 150

def load_and_preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # normalize
    return np.expand_dims(img, axis=0)
