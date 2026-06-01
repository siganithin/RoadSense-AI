"""
inference_api.py — FastAPI REST endpoint for RoadSense AI
Provides a production-grade HTTP API for road damage inference.

Endpoints:
    POST /predict        — Single image inference
    POST /predict/batch  — Batch image inference
    GET  /health         — Health check
    GET  /classes        — List supported classes
    GET  /model/info     — Model metadata

Run locally:
    uvicorn api.inference_api:app --reload --port 8000

Swagger UI: http://localhost:8000/docs
"""

import os
import sys
import io
import pickle
import time
import numpy as np
from typing import List
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import tensorflow as tf
    import yaml
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

if not FASTAPI_AVAILABLE:
    raise ImportError("Install fastapi and uvicorn: pip install fastapi uvicorn python-multipart")

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RoadSense AI — Inference API",
    description="REST API for road damage detection using EfficientNetB0",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model ────────────────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

_model = None
_class_names = None

def get_model():
    global _model, _class_names
    if _model is None:
        _model = tf.keras.models.load_model("models/damage_classifier.h5")
        with open("models/class_names.pkl", "rb") as f:
            cn_map = pickle.load(f)
        _class_names = [cn_map[i] for i in sorted(cn_map.keys())] if isinstance(cn_map, dict) else cn_map
    return _model, _class_names


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictionResult(BaseModel):
    predicted_class: str
    confidence: float
    severity: str
    recommendation: str
    all_probabilities: dict
    inference_time_ms: float


class BatchResult(BaseModel):
    filename: str
    predicted_class: str
    confidence: float
    severity: str


# ── Helpers ───────────────────────────────────────────────────────────────────
def preprocess(image_bytes: bytes) -> np.ndarray:
    img_size = tuple(config["data"]["image_size"])
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(img_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    return {"status": "healthy", "model_loaded": _model is not None}


@app.get("/classes", tags=["Model"])
def list_classes():
    return {
        "classes": config["classes"],
        "severity": config["severity"],
    }


@app.get("/model/info", tags=["Model"])
def model_info():
    return {
        "architecture": config["model"]["base_model"],
        "input_size": config["data"]["image_size"],
        "num_classes": len(config["classes"]),
        "version": "2.0.0",
    }


@app.post("/predict", response_model=PredictionResult, tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (jpg/png)")

    model, class_names = get_model()
    image_bytes = await file.read()

    t0 = time.time()
    arr = preprocess(image_bytes)
    probs = model.predict(arr, verbose=0)[0]
    elapsed_ms = (time.time() - t0) * 1000

    pred_idx   = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx]) * 100
    severity   = config["severity"].get(pred_class, "Unknown")
    recommendation = config["recommendations"].get(severity, "Inspect road condition.")

    return PredictionResult(
        predicted_class=pred_class,
        confidence=round(confidence, 2),
        severity=severity,
        recommendation=recommendation,
        all_probabilities={cls: round(float(p) * 100, 2) for cls, p in zip(class_names, probs)},
        inference_time_ms=round(elapsed_ms, 2),
    )


@app.post("/predict/batch", response_model=List[BatchResult], tags=["Inference"])
async def predict_batch(files: List[UploadFile] = File(...)):
    model, class_names = get_model()
    results = []
    for file in files:
        try:
            image_bytes = await file.read()
            arr   = preprocess(image_bytes)
            probs = model.predict(arr, verbose=0)[0]
            pred_idx   = int(np.argmax(probs))
            pred_class = class_names[pred_idx]
            results.append(BatchResult(
                filename=file.filename,
                predicted_class=pred_class,
                confidence=round(float(probs[pred_idx]) * 100, 2),
                severity=config["severity"].get(pred_class, "Unknown"),
            ))
        except Exception as e:
            results.append(BatchResult(
                filename=file.filename,
                predicted_class="Error",
                confidence=0.0,
                severity="Unknown",
            ))
    return results
