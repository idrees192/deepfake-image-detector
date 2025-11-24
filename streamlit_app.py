import streamlit as st
import torch
import timm
import torch.nn as nn
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ---------------------------
# CONFIG
# ---------------------------
DEVICE = "cpu"
IMG_SIZE = 224
MODEL_PATH = "model/deepfake_model.pth"

# ---------------------------
# MODEL DEFINITION
# ---------------------------
class DeepfakeModel(nn.Module):
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
@st.cache_resource
def load_model():
    model = DeepfakeModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
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

def preprocess(image):
    img = np.array(image)
    img = transform(image=img)["image"]
    return img.unsqueeze(0)


# ---------------------------
# PREDICT FUNCTION
# ---------------------------
def predict(image):
    tensor = preprocess(image)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).numpy()[0]

    label = "FAKE" if np.argmax(probs) == 1 else "REAL"
    confidence = probs.max()
    return label, confidence


# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("🕵️ Deepfake Image Detector")
st.write("Upload an image to classify whether it is **Real** or **Fake**.")

uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    label, conf = predict(image)

    st.subheader(f"Prediction: **{label}**")
    st.write(f"Confidence: **{conf:.4f}**")

