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

# ---------------------------
# CONFIGURATION
# ---------------------------
IMG_SIZE = 224
DEVICE = "cpu"  # Change this to 'cuda' if you are using a GPU
MODEL_PATH = "model/deepfake_model.pth"

# ---------------------------
# CORS CONFIGURATION
# ---------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
# PREPROCESSING FUNCTION
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
# PREDICTION FUNCTION
# ---------------------------
def predict(image: Image.Image):
    tensor = preprocess(image)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).numpy()[0]

    label = "FAKE" if np.argmax(probs) == 1 else "REAL"
    confidence = probs.max()
    return label, confidence


# ---------------------------
# ENDPOINTS
# ---------------------------
@app.post("/analyze/")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents))

    label, confidence = predict(image)

    return JSONResponse(content={
        "label": label,
        "confidence": confidence,
        "message": "Prediction successful!"
    })

# Running the app:
# You can run the app using uvicorn in the terminal:
# uvicorn frontend.app:app --reload
