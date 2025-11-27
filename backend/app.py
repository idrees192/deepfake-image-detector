# backend/app.py
import os
import io
import time
import json
import logging
import hashlib
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

import numpy as np
import cv2
import torch

from utils.gradcam_utils import predict_and_heatmap, load_model as gradcam_load_model

# Configuration (override using env vars if needed)
MODEL_PATH = os.getenv("MODEL_PATH", "model/deepfake_model.pth")
EVIDENCE_DIR = os.getenv("EVIDENCE_DIR", "evidence")
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))

os.makedirs(EVIDENCE_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepfake_api")

app = FastAPI(title="Deepfake Image Verifier")

# Load model on startup
MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@app.on_event("startup")
def on_startup():
    global MODEL
    logger.info("Starting up - loading model on device=%s", DEVICE)
    try:
        MODEL = gradcam_load_model(MODEL_PATH, device=DEVICE)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load model on startup: %s", e)
        # allow server to start; requests will get errors until model is fixed
        MODEL = None

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Accept an uploaded image file, run model inference and Grad-CAM,
    return verdict, confidence, and a heatmap URL.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # read bytes
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    # protect against extremely large uploads
    if len(contents) > 25_000_000:
        raise HTTPException(status_code=400, detail="File too large (max 25MB)")

    # Validate image
    try:
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")

    # compute hash
    file_hash = sha256_bytes(contents)
    original_path = os.path.join(EVIDENCE_DIR, f"{file_hash}.png")
    heatmap_path = os.path.join(EVIDENCE_DIR, f"{file_hash}_heatmap.png")
    meta_path = os.path.join(EVIDENCE_DIR, f"{file_hash}.json")

    # Save original image (safe)
    try:
        pil.save(original_path)
    except Exception:
        # as a fallback, write raw bytes
        with open(original_path, "wb") as f:
            f.write(contents)

    # Run prediction + gradcam safely
    try:
        res = predict_and_heatmap(MODEL, pil, device=DEVICE, img_size=IMG_SIZE)
    except Exception as e:
        logger.exception("Inference failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # res expected: {'verdict': 'FAKE'|'REAL', 'confidence': float, 'heatmap': np.ndarray (H,W,3) uint8}
    verdict = res.get("verdict", "UNKNOWN")
    confidence = float(res.get("confidence", 0.0))
    heatmap = res.get("heatmap", None)

    # Save heatmap if returned
    if isinstance(heatmap, np.ndarray):
        try:
            # ensure uint8 BGR for cv2.imwrite
            if heatmap.dtype != np.uint8:
                heatmap = (np.clip(heatmap, 0, 255)).astype(np.uint8)
            # convert RGB -> BGR for OpenCV writer
            bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
            cv2.imwrite(heatmap_path, bgr)
        except Exception:
            # fallback: save via PIL
            try:
                Image.fromarray(heatmap).save(heatmap_path)
            except Exception as e:
                logger.warning("Failed to save heatmap: %s", e)

    # Save metadata
    meta = {
        "file_hash": file_hash,
        "verdict": verdict,
        "confidence": confidence,
        "timestamp": int(time.time()),
        "model_path": MODEL_PATH
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    # generate response
    resp = {
        "file_hash": file_hash,
        "verdict": verdict,
        "confidence": confidence,
        "heatmap_url": f"/heatmap/{file_hash}"
    }
    return JSONResponse(content=resp)

@app.get("/heatmap/{file_hash}")
def get_heatmap(file_hash: str):
    p = os.path.join(EVIDENCE_DIR, f"{file_hash}_heatmap.png")
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail="Heatmap not found")
    return FileResponse(p, media_type="image/png")
