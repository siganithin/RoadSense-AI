"""
evaluate.py — Model evaluation script for RoadSense AI
Generates confusion matrix, classification report, ROC curves, and per-class metrics.
"""

import os
import sys
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
import tensorflow as tf
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_preparation import load_dataset, create_data_generators
from src.utils import get_logger, ensure_dirs

logger = get_logger("evaluate", log_dir="logs")

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

REPORT_DIR = "reports"
FIG_DIR    = os.path.join(REPORT_DIR, "figures")
ensure_dirs(REPORT_DIR, FIG_DIR)


def evaluate_model():
    logger.info("Loading model and class names...")
    model = tf.keras.models.load_model("models/damage_classifier.h5")
    with open("models/class_names.pkl", "rb") as f:
        class_names_map = pickle.load(f)

    # Build ordered class list
    if isinstance(class_names_map, dict):
        class_names = [class_names_map[i] for i in sorted(class_names_map.keys())]
    else:
        class_names = class_names_map

    logger.info("Loading validation data...")
    image_paths, labels = load_dataset(config["data"]["dataset_path"])
    _, val_gen = create_data_generators(
        image_paths, labels,
        batch_size=config["data"]["batch_size"],
        img_size=tuple(config["data"]["image_size"]),
        validation_split=config["data"]["validation_split"],
        seed=config["data"]["seed"],
    )

    logger.info("Running predictions on validation set...")
    val_gen.reset()
    y_pred_probs = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_gen.classes

    # ── Classification Report ─────────────────────────────────────────────────
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_path = os.path.join(REPORT_DIR, "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    logger.info(f"Classification report saved → {report_path}")
    print("\n" + classification_report(y_true, y_pred, target_names=class_names))

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix — Validation Set")
    plt.tight_layout()
    cm_path = os.path.join(FIG_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved → {cm_path}")

    # ── ROC Curves ────────────────────────────────────────────────────────────
    n_classes = len(class_names)
    y_true_bin = np.eye(n_classes)[y_true]
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e94560", "#4a90d9", "#48bb78"]
    for i, (cls, col) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{cls} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — One-vs-Rest")
    ax.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(FIG_DIR, "roc_curves.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    logger.info(f"ROC curves saved → {roc_path}")

    # ── Precision-Recall Curves ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (cls, col) in enumerate(zip(class_names, colors)):
        prec, rec, _ = precision_recall_curve(y_true_bin[:, i], y_pred_probs[:, i])
        ax.plot(rec, prec, color=col, lw=2, label=cls)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    plt.tight_layout()
    pr_path = os.path.join(FIG_DIR, "precision_recall_curves.png")
    plt.savefig(pr_path, dpi=150)
    plt.close()
    logger.info(f"Precision-Recall curves saved → {pr_path}")

    # ── Summary JSON ─────────────────────────────────────────────────────────
    summary = {
        "val_accuracy": float(report["accuracy"]),
        "weighted_f1":  float(report["weighted avg"]["f1-score"]),
        "per_class":    {cls: report[cls] for cls in class_names},
    }
    with open(os.path.join(REPORT_DIR, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    logger.info("Evaluation complete.")
    return summary


if __name__ == "__main__":
    evaluate_model()
