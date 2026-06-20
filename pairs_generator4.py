import os
import random

def create_verification_pairs(df, num_pairs=1000):
    genuine_pairs = []
    impostor_pairs = []

    
    image_paths = df["image_path"].tolist()

    # Genuine pairs
    for _ in range(num_pairs):
        img = random.choice(image_paths)
        genuine_pairs.append({"img1": img, "img2": img, "true_label": 1})

    # Impostor pairs
    for _ in range(num_pairs):
        img1, img2 = random.sample(image_paths, 2)
        impostor_pairs.append({"img1": img1, "img2": img2, "true_label": 0})

    pairs = genuine_pairs + impostor_pairs
    random.shuffle(pairs)
    return pairs