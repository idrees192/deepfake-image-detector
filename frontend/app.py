from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from io import BytesIO
from PIL import Image
import timm
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Security utilities
from utils.hashing import compute_sha256
from utils.validator import validate_image
from utils.rate_limiter import allow_request
from utils.encryption import encrypt_image
from utils.tokens import generate_verification_token
from utils.logger import log_detection
from utils.risk_score import compute_risk

# ---------------------------
# CONFIG
# ---------------------------
IMG_SIZE = 224
DEVICE = "cpu"
MODEL_PATH = "model/deepfake_model.pth"

app = FastAPI()

# ---------------------------
# CORS
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# MODEL DEF
# ---------------------------
class DeepfakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=2
        )

    def forward(self, x):
        return self.model(x)


# ---------------------------
# LOAD MODEL
# ---------------------------
def load_model():
    model = DeepfakeModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_model()


# ---------------------------
# PREPROCESS
# ---------------------------
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def preprocess(image: Image.Image):
    img = np.array(image)
    img = transform(image=img)["image"]
    return img.unsqueeze(0)


# ---------------------------
# PREDICT
# ---------------------------
def predict(image: Image.Image):
    tensor = preprocess(image)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).numpy()[0]

    label = "FAKE" if np.argmax(probs) == 1 else "REAL"
    confidence = float(probs.max())

    return label, confidence


# ---------------------------
# MAIN ENDPOINT
# ---------------------------
@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
    # Read bytes
    file_bytes = await file.read()

    # 1. Validate input
    valid, msg = validate_image(file_bytes)
    if not valid:
        return JSONResponse({"error": msg})

    # 2. Rate limit
    if not allow_request("default_user"):
        return JSONResponse({"error": "Too many requests. Wait a minute."})

    # 3. Hash image
    img_hash = compute_sha256(file_bytes)

    # 4. Encrypt image
    encrypt_image(file_bytes, img_hash)

    # 5. Prediction
    image = Image.open(BytesIO(file_bytes)).convert("RGB")
    label, confidence = predict(image)

    # 6. Risk score
    risk = compute_risk(confidence)

    # 7. Generate JWT token
    token = generate_verification_token(
        image_hash=img_hash,
        label=label,
        confidence=confidence
    )

    # 8. Log the detection
    log_detection(img_hash, label, confidence)

    # 9. Return response
    return JSONResponse({
        "label": label,
        "confidence": confidence,
        "risk": risk,
        "hash": img_hash,
        "token": token
    })
