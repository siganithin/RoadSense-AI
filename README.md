# 🛣️ RoadSense AI — Road Damage Detection System

> **AI-Powered Smart City Infrastructure Monitoring using EfficientNetB0 Transfer Learning**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)](https://streamlit.io)
[![Accuracy](https://img.shields.io/badge/Val%20Accuracy-94.7%25-green)]()

---

## 📌 Overview

**RoadSense AI** is an end-to-end deep learning system that automatically detects and classifies
road surface damage from images. It uses **EfficientNetB0** with transfer learning and two-phase
fine-tuning to classify road images into:

| Class | Severity | Description |
|-------|----------|-------------|
| 🔴 Pothole | High | Bowl-shaped depressions — immediate repair |
| 🟡 Crack | Medium | Surface fractures — schedule repair |
| 🟢 Manhole | Low | Cover anomalies — routine inspection |

---

## �️ Project Structure

```
CNN_road_damage/
├── api/
│   ├── __init__.py
│   └── inference_api.py          ← FastAPI REST endpoints
├── app/
│   └── streamlit_app.py          ← 6-page Streamlit UI
├── logs/                         ← Runtime logs
├── models/
│   └── model_card.md             ← Model documentation
├── notebooks/
│   ├── 01_EDA.ipynb              ← Exploratory Data Analysis
│   ├── 02_model_training.ipynb   ← Training walkthrough
│   └── 03_evaluation_and_gradcam.ipynb
├── reports/
│   ├── project_report.md         ← Full technical report
│   └── figures/                  ← Auto-generated charts
├── results/                      ← Batch inference outputs
├── src/
│   ├── augmentation.py           ← Augmentation pipeline
│   ├── data_preparation.py       ← Dataset loading & generators
│   ├── evaluate.py               ← Evaluation script
│   ├── gradcam.py                ← GradCAM explainability
│   ├── model.py                  ← EfficientNetB0 builder
│   ├── predict.py                ← CLI inference
│   └── utils.py                  ← Shared utilities
├── tests/
│   ├── test_model.py             ← 12 unit tests
│   └── test_data_pipeline.py     ← 7 pipeline tests
├── .gitignore
├── config.yaml                   ← All hyperparameters & paths
├── docker-compose.yml
├── Dockerfile
├── download_data.py
├── requirements.txt
├── setup.py
└── train.py
```

---

## 🚀 Features

- **94.7% Validation Accuracy** using EfficientNetB0 + fine-tuning
- **Live Inference** — upload any road image, get instant results
- **Confidence Gauge** — visual confidence meter per prediction
- **GradCAM Explainability** — heatmap showing model attention regions
- **Severity Scoring** — High / Medium / Low with maintenance recommendations
- **Model Analytics** — training curves, confusion matrix, ROC-AUC, benchmark comparison
- **Dataset Insights** — class distribution, pixel stats, preprocessing pipeline
- **Project Roadmap** — Gantt chart, timeline, future enhancements
- **FastAPI REST API** — production-grade HTTP endpoints
- **Docker ready** — single command deployment

---

## ⚙️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/CNN_road_damage.git
cd CNN_road_damage

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset
python download_data.py
# OR update dataset_path in config.yaml manually

# 5. Train the model
python train.py

# 6. Launch the Streamlit app
streamlit run app/streamlit_app.py
```

---

## 🧠 Model Architecture

```
Input (224×224×3)
    ↓
EfficientNetB0 Backbone (ImageNet weights)
    ↓
GlobalAveragePooling2D
    ↓
Dropout (0.3)
    ↓
Dense (3, softmax)  →  [Pothole, Crack, Manhole]
```

**Training Strategy:**
- **Phase 1** (Epochs 1–10): Backbone frozen, head only trained — fast convergence
- **Phase 2** (Epochs 11–30): Top 20 layers unfrozen, lr=1e-5 — domain adaptation

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | **94.7%** |
| Weighted F1-Score | **0.948** |
| Pothole AUC | 0.991 |
| Crack AUC | 0.983 |
| Manhole AUC | 0.996 |
| Inference Time | < 200ms |

---

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up --build

# App available at http://localhost:8501
```

---

## 🔌 REST API

```bash
# Start API server
uvicorn api.inference_api:app --reload --port 8000

# Predict single image
curl -X POST http://localhost:8000/predict \
     -F "file=@road_image.jpg"

# Swagger UI
http://localhost:8000/docs
```

---

## 🧪 Run Tests

```bash
pytest tests/ -v
```

---

## 🏙️ Industry Applications

- Municipal road inspection automation
- Autonomous vehicle navigation
- Insurance claim assessment
- Post-construction quality inspection
- IoT edge deployment on road cameras
- GIS-integrated city dashboards

---

## 🔮 Future Work

- [ ] YOLOv8 object detection for multi-damage localization
- [ ] Real-time video stream processing
- [ ] GPS tagging + GIS map integration
- [ ] TFLite export for mobile deployment
- [ ] FastAPI + Docker REST API on AWS/GCP
- [ ] Active learning feedback loop

---

## 🛠️ Tech Stack

`TensorFlow` `Keras` `EfficientNetB0` `Streamlit` `FastAPI` `Plotly` `OpenCV` `scikit-learn` `NumPy` `Pandas` `Docker`

---

## 📦 Dataset

[Road Damage Dataset — Potholes, Cracks & Manholes](https://www.kaggle.com/datasets/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes)
by Lorenzo Arcioni on Kaggle.

---

*RoadSense AI v2.0 — Built for Smart City Infrastructure Intelligence*
