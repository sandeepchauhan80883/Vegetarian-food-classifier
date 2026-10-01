"""
backend/app.py
---------------
Minimal REST API around the trained vegetarian-food classifier.

Endpoints:
  GET  /health   -> {"status": "ok", "checkpoint_found": true/false}
  POST /predict  -> multipart file upload -> top-5 predictions (JSON)

Run:
  uvicorn app:app --reload --port 8000
(from inside the backend/ folder, with the trained checkpoint placed at
../artifacts/best_model.pt -- see the top-level README.md)
"""

import io
import os
import sys

# Make the sibling `veg_classifier` package importable when running this
# file directly from inside backend/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from veg_classifier import config, demo as demo_mod

app = FastAPI(title="Vegetarian Food Classifier API")

# Wide-open CORS so the plain HTML frontend (opened as a local file, or
# served from any port) can call this API during development / a project
# demo. Tighten `allow_origins` before deploying this anywhere public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(config.CHECKPOINT_PATH):
            raise FileNotFoundError(config.CHECKPOINT_PATH)
        _model = demo_mod.load_trained_model()
    return _model


@app.get("/health")
def health():
    return {
        "status": "ok",
        "checkpoint_found": os.path.exists(config.CHECKPOINT_PATH),
        "checkpoint_path": config.CHECKPOINT_PATH,
        "classes": config.CLASSES,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        model = get_model()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No trained checkpoint found at '{config.CHECKPOINT_PATH}'. "
                "Train the model (see the notebook) and copy artifacts/best_model.pt "
                "into that path, then restart this server."
            ),
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read that as an image.")

    results = demo_mod.predict_image(model, img, top_k=5)
    return {
        "predictions": [
            {
                "dish": dish.replace("_", " ").title(),
                "meal_category": cat,
                "probability": round(prob, 4),
            }
            for dish, cat, prob in results
        ]
    }
