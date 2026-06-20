import cv2
import numpy as np
import random


def simulate_airport_conditions11(image_path):
    """
    Simulates real airport live capture conditions.
    """
    img = cv2.imread(image_path)

    if img is None:
        return None

    # resolution
    img = cv2.resize(img, (160, 160))

    # Random brightness variation 
    brightness = random.uniform(0.7, 1.3)
    img = np.clip(img * brightness, 0, 255).astype(np.uint8)

    # Slight blur to simulate camera noise
    if random.random() > 0.5:
        img = cv2.GaussianBlur(img, (5, 5), 0)

    return img
