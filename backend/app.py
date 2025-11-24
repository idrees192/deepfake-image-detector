from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import timm
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
import os

# ---------------------------
# CONFIGURATION
# ---------------------------
IMG_SIZE = 224
DEVICE = "cpu"  # Change to 'cuda' if you have a GPU
MODEL_PATH = "model/deepfake_model.pth"  # Path to your trained model file

# ---------------------------
# CORS CONFIGURATION
# ---------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Can be restricted to specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# MODEL DEFINITION
# ---------------------------
class DeepfakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=False,
            num_classes=2  # Two classes: real or fake
        )

    def forward(self, x):
        return self.model(x)

# ---------------------------
# LOAD THE MODEL
# ---------------------------
def load_model():
    model = DeepfakeModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

# ---------------------------
# IMAGE PREPROCESSING
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
    return img.unsqueeze(0)  # Add batch dimension

# ---------------------------
# PREDICTION FUNCTION
# ---------------------------
def predict(image: Image.Image):
    tensor = preprocess(image)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).numpy()[0]  # Apply softmax to get probabilities

    label = "FAKE" if np.argmax(probs) == 1 else "REAL"
    confidence = probs.max()  # Maximum probability
    return label, confidence

# ---------------------------
# API ENDPOINTS
# ---------------------------
@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()  # Read file from request
    image = Image.open(BytesIO(contents))  # Open the image file

    label, confidence = predict(image)  # Get prediction and confidence

    # Prepare response
    response = {
        "label": label,
        "confidence": confidence,
        "message": "Prediction successful!"
    }

    # Optionally: Save the image for logging or debugging purposes
    file_path = os.path.join("evidence", f"{file.filename}")
    os.makedirs("evidence", exist_ok=True)
    image.save(file_path)

    return JSONResponse(content=response)

# ---------------------------
# RUN THE APP
# ---------------------------
# Run the app with:
# uvicorn backend.app:app --reload

