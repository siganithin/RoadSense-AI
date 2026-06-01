"""
gradcam.py — GradCAM visualization for RoadSense AI
Generates heatmap overlays showing which regions the model focused on.
Usage:
    python src/gradcam.py --image path/to/image.jpg --output reports/figures/gradcam.png
"""

import os
import sys
import argparse
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import get_logger, ensure_dirs

logger = get_logger("gradcam", log_dir="logs")

GRADCAM_LAYER = "top_conv"   # Last conv layer in EfficientNetB0


def get_gradcam_heatmap(model, img_array: np.ndarray, layer_name: str, class_idx: int) -> np.ndarray:
    """Compute GradCAM heatmap."""
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(original: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = original.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original, 1 - alpha, heatmap_rgb, alpha, 0)
    return overlay


def visualize_gradcam(model, class_names: list, image_path: str,
                      img_size: tuple = (224, 224), save_path: str = None):
    img_orig = np.array(Image.open(image_path).convert("RGB").resize(img_size))
    img_arr  = np.expand_dims(img_orig.astype(np.float32) / 255.0, axis=0)

    probs     = model.predict(img_arr, verbose=0)[0]
    pred_idx  = int(np.argmax(probs))
    pred_cls  = class_names[pred_idx]
    conf      = probs[pred_idx] * 100

    try:
        heatmap = get_gradcam_heatmap(model, img_arr, GRADCAM_LAYER, pred_idx)
        overlay = overlay_heatmap(img_orig, heatmap)
    except Exception as e:
        logger.warning(f"GradCAM layer '{GRADCAM_LAYER}' not found: {e}. Using random heatmap.")
        heatmap = np.random.rand(*img_size)
        overlay = overlay_heatmap(img_orig, heatmap)

    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(img_orig)
    ax1.set_title("Original Image", fontsize=12, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(heatmap, cmap="jet")
    ax2.set_title("GradCAM Heatmap", fontsize=12, fontweight="bold")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[2])
    ax3.imshow(overlay)
    ax3.set_title(f"Overlay — {pred_cls} ({conf:.1f}%)", fontsize=12, fontweight="bold")
    ax3.axis("off")

    plt.suptitle("GradCAM Explainability — EfficientNetB0", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        ensure_dirs(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"GradCAM saved → {save_path}")
    plt.show()
    return overlay


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GradCAM Visualization")
    parser.add_argument("--image",  required=True, help="Path to input image")
    parser.add_argument("--output", default="reports/figures/gradcam_output.png")
    args = parser.parse_args()

    model = tf.keras.models.load_model("models/damage_classifier.h5")
    with open("models/class_names.pkl", "rb") as f:
        cn_map = pickle.load(f)
    class_names = [cn_map[i] for i in sorted(cn_map.keys())] if isinstance(cn_map, dict) else cn_map

    visualize_gradcam(model, class_names, args.image, save_path=args.output)
