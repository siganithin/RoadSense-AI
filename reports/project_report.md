# RoadSense AI — Technical Project Report

**Version:** 2.0  
**Date:** June 2026  
**Domain:** Computer Vision / Smart City Infrastructure

---

## Executive Summary

RoadSense AI is an end-to-end deep learning system for automated road damage detection
and severity classification. Using EfficientNetB0 transfer learning with two-phase
fine-tuning, the system achieves **94.7% validation accuracy** on a dataset of ~5,000
annotated road images across three damage categories: Potholes, Cracks, and Manhole anomalies.

---

## 1. Problem Statement

Road infrastructure degradation is a critical public safety and economic issue.
Manual inspection is:
- **Slow** — city-wide surveys take weeks
- **Expensive** — ₹1.5 lakh crore annual maintenance cost in India
- **Inconsistent** — subjective human assessment varies

**Goal:** Build an AI system that classifies road damage from a single image in < 200ms
with > 90% accuracy, enabling proactive maintenance scheduling.

---

## 2. Dataset

| Property | Value |
|---|---|
| Source | Kaggle — Road Damage Dataset (Lorenzo Arcioni) |
| Total Images | ~5,000 |
| Annotation Format | YOLO (class_id cx cy w h) |
| Classes | Pothole (0), Crack (1), Manhole (2) |
| Native Resolution | 640 × 360 px |
| Train / Val Split | 80% / 20% (stratified) |

### Label Strategy
Multi-label images (multiple damage types) are assigned a single dominant class
based on the most frequent class ID in the annotation file.

---

## 3. Model Architecture

```
Input (224×224×3)
    ↓
EfficientNetB0 (ImageNet pretrained, 5.3M params)
    ↓
GlobalAveragePooling2D
    ↓
Dropout (0.3)
    ↓
Dense (3, softmax)
```

### Training Strategy

**Phase 1 — Feature Extraction (Epochs 1–10)**
- Backbone frozen
- Only classification head trained
- LR = 1e-4

**Phase 2 — Fine-Tuning (Epochs 11–30)**
- Top 20 EfficientNetB0 layers unfrozen
- LR = 1e-5
- Domain-specific feature adaptation

### Callbacks
- `ModelCheckpoint` — saves best val_accuracy model
- `EarlyStopping` — patience=5 on val_loss
- `ReduceLROnPlateau` — factor=0.2, patience=3

---

## 4. Data Augmentation

| Technique | Range |
|---|---|
| Rotation | ±25° |
| Width/Height Shift | ±20% |
| Zoom | ±25% |
| Horizontal Flip | 50% probability |
| Brightness | [0.75, 1.25] |
| Channel Shift | ±20 |
| Shear | ±20° |

---

## 5. Results

### Overall Metrics

| Metric | Value |
|---|---|
| Validation Accuracy | **94.7%** |
| Weighted F1-Score | **0.948** |
| Macro Precision | 0.947 |
| Macro Recall | 0.950 |

### Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Pothole | 0.960 | 0.950 | 0.955 | 412 |
| Crack | 0.930 | 0.940 | 0.935 | 389 |
| Manhole | 0.950 | 0.960 | 0.955 | 201 |

### AUC Scores (One-vs-Rest)

| Class | AUC |
|---|---|
| Pothole | 0.991 |
| Crack | 0.983 |
| Manhole | 0.996 |

---

## 6. Benchmark Comparison

| Model | Val Accuracy | Params (M) | Inference (ms) |
|---|---|---|---|
| MobileNetV2 | 88.2% | 3.4 | 45 |
| ResNet50 | 91.4% | 25.6 | 120 |
| VGG16 | 89.7% | 138.4 | 310 |
| InceptionV3 | 92.1% | 23.9 | 140 |
| **EfficientNetB0 (Ours)** | **94.7%** | **5.3** | **38** |

EfficientNetB0 achieves the best accuracy with the smallest parameter count and
fastest inference — ideal for edge deployment.

---

## 7. Explainability — GradCAM

Gradient-weighted Class Activation Mapping (GradCAM) is used to visualize which
regions of the input image the model focuses on. This provides:
- **Transparency** — stakeholders can verify model decisions
- **Debugging** — identify spurious correlations
- **Trust** — critical for municipal deployment

GradCAM heatmaps are generated from the `top_conv` layer of EfficientNetB0.

---

## 8. Deployment

The system is deployed as a **Streamlit web application** with:
- Live image upload and inference
- Confidence gauge and probability chart
- Severity scoring (High / Medium / Low)
- Actionable maintenance recommendations
- GradCAM explainability overlay
- Model analytics dashboard
- Dataset insights and project documentation

---

## 9. Future Work

1. **Object Detection** — Upgrade to YOLOv8 for bounding-box localization
2. **Video Processing** — Real-time dashcam stream analysis
3. **GIS Integration** — GPS-tagged damage maps for city dashboards
4. **Mobile Deployment** — TFLite export for Android/iOS
5. **Cloud API** — FastAPI + Docker on AWS/GCP
6. **Active Learning** — Human-in-the-loop for low-confidence samples

---

## 10. References

1. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for CNNs. ICML.
2. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. ICCV.
3. Arcioni, L. Road Damage Dataset. Kaggle, 2023.
4. Chollet, F. (2021). Deep Learning with Python. Manning Publications.
5. Howard, A., et al. (2019). Searching for MobileNetV3. ICCV.

---

*RoadSense AI v2.0 — Built for Smart City Infrastructure Intelligence*
