import io
import torch
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import your model
from model.deepfake_model import load_model

# Security utilities
from utils.hashing import compute_sha256
from utils.validator import validate_image
from utils.rate_limiter import allow_request
from utils.encryption import encrypt_image
from utils.tokens import generate_verification_token
from utils.logger import log_detection
from utils.risk_score import compute_risk

app = FastAPI()

# -----------------------------
# MODEL & PREPROCESSING
# -----------------------------
model = load_model()
model.eval()

IMG_SIZE = 224

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def preprocess(image: Image.Image):
    img = np.array(image)
    img = transform(image=img)["image"]
    return img.unsqueeze(0)  # shape: [1, 3, 224, 224]


@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):

    # Read file bytes
    file_bytes = await file.read()

    # 1. VALIDATION
    valid, msg = validate_image(file_bytes)
    if not valid:
        return {"error": msg}

    # 2. RATE LIMIT
    if not allow_request("user"):
        return {"error": "Rate limit exceeded. Try again later."}

    # 3. HASHING
    image_hash = compute_sha256(file_bytes)

    # 4. ENCRYPT IMAGE
    encrypted_path = encrypt_image(file_bytes, image_hash)

    # 5. LOAD IMAGE
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    # 6. PREPROCESS
    tensor = preprocess(image)

    # 7. MODEL PREDICTION
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).numpy()[0]

    confidence = float(probs[1])  # index 1 = fake
    label = "fake" if confidence > 0.5 else "real"

    # 8. RISK SCORE
    risk_level = compute_risk(confidence)

    # 9. JWT TOKEN
    token = generate_verification_token(
        image_hash=image_hash,
        label=label,
        confidence=confidence
    )

    # 10. LOGGING
    log_detection(
        image_hash=image_hash,
        label=label,
        confidence=confidence
    )

    # 11. RESPONSE
    return {
        "label": label,
        "confidence": confidence,
        "risk": risk_level,
        "token": token,
        "encrypted_path": encrypted_path,
        "hash": image_hash
    }
