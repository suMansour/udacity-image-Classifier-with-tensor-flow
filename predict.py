
import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image



image_size = 224

def process_image(img_path: Path) -> np.ndarray:
    """Process an image path into a 4D tensor suitable for making predictions with our model"""
    img = image.load_img(img_path, target_size=(image_size, image_size))
    img = image.img_to_array(img)  # convert the PIL image to a numpy array
    img = tf.convert_to_tensor(img)
    img /= 255
    img = tf.expand_dims(img, 0)

    return img.numpy()

def predict(image_path: Path, model, top_k: int, category_names: Optional[Path]) -> List[Tuple[str, float]]:
    """Use model to predict the top K most likely classes for the image at image_path"""
    processed_image = process_image(image_path)

    predictions = model.predict(processed_image)

    top_k_probs, top_k_classes = tf.math.top_k(predictions, top_k)
    top_k_probs = top_k_probs.numpy().squeeze()
    classes_labels = top_k_classes.numpy().squeeze()

    if category_names is not None:
        with open(category_names, 'r') as f:
            class_names = json.load(f)
        top_k_classes = [class_names[str(i)] for i in classes_labels]
    else:
        top_k_classes = classes_labels

    return list(zip(top_k_classes, top_k_probs))


def main(debug=False):
    if debug:
        image_path     = Path("./test_images/wild_pansy.jpg")
        model_path     = Path("./keras_model.h5")
        top_k          = 5
        category_names = Path("label_map.json")
    else:
        parser = argparse.ArgumentParser(description='Predict flower name from an image')
        parser.add_argument('image_path', type=Path, help='Path to the image')
        parser.add_argument('model_path', type=Path, help='Path to the model')
        parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes')
        parser.add_argument('--category_names', type=Path, help='Path to a JSON file mapping labels to flower names')

        args = parser.parse_args()

        image_path     = args.image_path
        model_path     = args.model_path
        top_k          = args.top_k
        category_names = args.category_names

    model = load_model(model_path, custom_objects={
        'KerasLayer': hub.KerasLayer
        })

    predictions = predict(image_path, model, top_k, category_names)

    print (f"\n*** Top {top_k} classes: ***\n")

    for i, (class_name, prob) in enumerate(predictions):
        print(f"Rank {i+1}:")
        print(f"Class: {class_name}")
        print(f"Probability: {prob:.4f}\n")

if __name__ == "__main__":
    main(debug=True)