import streamlit as st
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as T
import timm
import hashlib
import base64
from pymongo import MongoClient
from datetime import datetime
import io

# -------------------------
# MONGODB CONNECTION
# -------------------------
@st.cache_resource
def get_db():
    client = MongoClient(
        "mongodb+srv://khanmidrees693_db_user:mGrml4gCOcBJWNju@cluster0.aav7zwl.mongodb.net/"
    )
    db = client["deepfake_logs"]    # database name
    return db

db = get_db()
logs = db["image_logs"]  # collection name


# -------------------------
# MODEL DEFINITION
# -------------------------
class DeepfakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=2
        )
    def forward(self, x):
        return self.model(x)


# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    model = DeepfakeModel()
    state_dict = torch.load("model/deepfake_model.pth", map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

model = load_model()

# -------------------------
# TRANSFORM
# -------------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

st.title("Deepfake Image Detector — With MongoDB Logging")

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

def compute_hash(image_bytes):
    return hashlib.sha256(image_bytes).hexdigest()

def encode_image_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

if uploaded:

    st.image(uploaded)

    # Read image bytes for hashing
    image_bytes = uploaded.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0]
        real = float(probs[0])
        fake = float(probs[1])

    label = "FAKE" if fake > real else "REAL"

    st.subheader("Prediction")
    st.write(f"**Label:** {label}")
    st.write(f"Fake Probability: {fake:.4f}")
    st.write(f"Real Probability: {real:.4f}")

    # -------------------------
    # STORE LOG INTO MONGODB
    # -------------------------
    try:
        log_data = {
            "image_hash": compute_hash(image_bytes),
            "label": label,
            "fake_prob": fake,
            "real_prob": real,
            "timestamp": datetime.utcnow(),
            "image_base64": encode_image_base64(img)
        }

        logs.insert_one(log_data)

        st.success("Log saved to MongoDB Atlas successfully! 🎉")

    except Exception as e:
        st.error(f"Error saving log to MongoDB: {e}")
