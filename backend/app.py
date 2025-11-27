import io
import torch
from fastapi import FastAPI, UploadFile, File
from PIL import Image

# Import your model
from model.deepfake_model import load_model  # adjust if needed

# Import security utilities
from utils.hashing import compute_sha256
from utils.validator import validate_image
from utils.rate_limiter import allow_request
from utils.encryption import encrypt_image
from utils.tokens import generate_verification_token
from utils.logger import log_detection
from utils.risk_score import compute_risk

app = FastAPI()

# Load your model once
model = load_model()

@app.post("/detect")
async def detect_image(file: UploadFile = File(...)):

    # Read file bytes
    file_bytes = await file.read()

    # 1. VALIDATION
    valid, msg = validate_image(file_bytes)
    if not valid:
        return {"error": msg}

    # 2. RATE LIMIT (per user/session)
    if not allow_request("user"):
        return {"error": "Rate limit exceeded. Try again later."}

    # 3. HASHING
    image_hash = compute_sha256(file_bytes)

    # 4. ENCRYPT & STORE IMAGE
    encrypted_path = encrypt_image(file_bytes, image_hash)

    # 5. PREPARE IMAGE FOR MODEL
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image = image.resize((224, 224))
    tensor = torch.tensor(torch.FloatTensor(torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))))
    # (Replace above with your real preprocessing pipeline)

    # 6. MODEL PREDICTION
    with torch.no_grad():
        outputs = model(tensor.unsqueeze(0))
        confidence = float(torch.softmax(outputs, dim=1)[0][1])
        label = "fake" if confidence > 0.5 else "real"

    # 7. RISK SCORE
    risk_level = compute_risk(confidence)

    # 8. JWT TOKEN
    token = generate_verification_token(
        image_hash=image_hash,
        label=label,
        confidence=confidence
    )

    # 9. LOGGING
    log_detection(
        image_hash=image_hash,
        label=label,
        confidence=confidence
    )

    # 10. RETURN RESPONSE
    return {
        "label": label,
        "confidence": confidence,
        "risk": risk_level,
        "token": token,
        "encrypted_path": encrypted_path,
        "hash": image_hash
    }
