# Model Card — RoadSense AI Damage Classifier

## Model Details

| Property | Value |
|---|---|
| Model Name | damage_classifier |
| Version | 2.0 |
| Architecture | EfficientNetB0 + Custom Head |
| Framework | TensorFlow 2.x / Keras |
| Input | 224 × 224 × 3 (RGB, normalized [0,1]) |
| Output | Softmax probabilities over 3 classes |
| File | `damage_classifier.h5` |
| Class Mapping | `class_names.pkl` |

## Classes

| ID | Name | Severity | Description |
|---|---|---|---|
| 0 | Pothole | High | Bowl-shaped road surface depression |
| 1 | Crack | Medium | Linear or alligator-pattern fractures |
| 2 | Manhole | Low | Manhole cover anomaly or misalignment |

## Performance

| Metric | Value |
|---|---|
| Validation Accuracy | 94.7% |
| Weighted F1 | 0.948 |
| Pothole AUC | 0.991 |
| Crack AUC | 0.983 |
| Manhole AUC | 0.996 |
| Inference Time | < 200ms (CPU) |

## Training Data

- **Dataset:** Road Damage Dataset (Kaggle, Lorenzo Arcioni)
- **Size:** ~5,000 images
- **Split:** 80% train / 20% validation (stratified)
- **Augmentation:** Rotation, flip, zoom, brightness, channel shift

## Intended Use

- Municipal road inspection automation
- Smart city infrastructure monitoring
- Insurance claim assessment
- Autonomous vehicle navigation support

## Limitations

- Trained on a single dataset; may underperform on very different road types
- Single-label classification only (no multi-damage localization)
- Performance may degrade on low-resolution or night-time images
- Not validated for real-time video streams

## Ethical Considerations

- Model decisions should be reviewed by qualified engineers before repair dispatch
- Confidence threshold of 80%+ recommended for automated actions
- Regular retraining recommended as road conditions and camera hardware evolve

## How to Load

```python
import tensorflow as tf
import pickle

model = tf.keras.models.load_model("models/damage_classifier.h5")
with open("models/class_names.pkl", "rb") as f:
    class_names = pickle.load(f)
```
