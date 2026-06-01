"""
predict.py — Single-image and batch inference for RoadSense AI
Usage:
    python src/predict.py --image path/to/image.jpg
    python src/predict.py --folder path/to/folder/ --output results/batch_output.csv
"""

import os
import sys
import argparse
import pickle
import json
import csv
import numpy as np
from PIL import Image
import tensorflow as tf
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import get_logger, normalize

logger = get_logger("predict", log_dir="logs")

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def load_model_assets():
    model = tf.keras.models.load_model("models/damage_classifier.h5")
    with open("models/class_names.pkl", "rb") as f:
        class_names_map = pickle.load(f)
    if isinstance(class_names_map, dict):
        class_names = [class_names_map[i] for i in sorted(class_names_map.keys())]
    else:
        class_names = class_names_map
    return model, class_names


def preprocess(image_path: str, size: tuple) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_single(model, class_names: list, image_path: str) -> dict:
    img_size = tuple(config["data"]["image_size"])
    arr = preprocess(image_path, img_size)
    probs = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    severity = config["severity"].get(pred_class, "Unknown")
    recommendation = config["recommendations"].get(severity, "Inspect road condition.")
    result = {
        "image":          os.path.basename(image_path),
        "predicted_class": pred_class,
        "confidence":     round(float(probs[pred_idx]) * 100, 2),
        "severity":       severity,
        "recommendation": recommendation,
        "all_probs":      {cls: round(float(p) * 100, 2) for cls, p in zip(class_names, probs)},
    }
    return result


def predict_batch(model, class_names: list, folder: str, output_csv: str):
    valid_exts = {".jpg", ".jpeg", ".png"}
    image_files = [
        os.path.join(folder, f) for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in valid_exts
    ]
    logger.info(f"Found {len(image_files)} images in {folder}")
    results = []
    for path in image_files:
        try:
            r = predict_single(model, class_names, path)
            results.append(r)
            logger.info(f"{r['image']} → {r['predicted_class']} ({r['confidence']}%)")
        except Exception as e:
            logger.warning(f"Failed on {path}: {e}")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "predicted_class", "confidence", "severity", "recommendation"])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in writer.fieldnames})
    logger.info(f"Batch results saved → {output_csv}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RoadSense AI — Inference")
    parser.add_argument("--image",  type=str, help="Path to a single image")
    parser.add_argument("--folder", type=str, help="Path to folder of images")
    parser.add_argument("--output", type=str, default="results/batch_output.csv")
    args = parser.parse_args()

    model, class_names = load_model_assets()

    if args.image:
        result = predict_single(model, class_names, args.image)
        print(json.dumps(result, indent=4))
    elif args.folder:
        predict_batch(model, class_names, args.folder, args.output)
    else:
        parser.print_help()
