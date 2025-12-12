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

# ================================================================
#  MONGODB CONNECTION  (YOUR DB + COLLECTION)
# ================================================================
@st.cache_resource
def get_db():
    client = MongoClient(
        "mongodb+srv://khanmidrees693_db_user:mGrml4gCOcBJWNju@cluster0.aav7zwl.mongodb.net/"
    )
    db = client["deepfake"]     # your database name
    return db

db = get_db()
logs = db["fake"]  # your collection name


# ================================================================
#  MODEL DEFINITION
# ================================================================
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


# ================================================================
#  LOAD MODEL
# ================================================================
@st.cache_resource
def load_model():
    model = DeepfakeModel()
    state_dict = torch.load("model/deepfake_model.pth", map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

model = load_model()


# ================================================================
#  IMAGE TRANSFORM
# ================================================================
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])


# ================================================================
#  HELPER FUNCTIONS
# ================================================================
def compute_hash(image_bytes):
    """Compute SHA256 hash of uploaded image."""
    return hashlib.sha256(image_bytes).hexdigest()

def encode_image_base64(img):
    """Encode image to base64 for MongoDB storage."""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


# ================================================================
#  STREAMLIT UI
# ================================================================
st.title("Deepfake Image Detector ")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    # Display image
    st.image(uploaded)

    # Read bytes
    image_bytes = uploaded.read()

    # Load image for preprocessing
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess
    x = transform(img).unsqueeze(0)

    # Predict
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0]
        real = float(probs[0])
        fake = float(probs[1])

    label = "FAKE" if fake > real else "REAL"

    # Show prediction
    st.subheader("Prediction Results")
    st.write(f"**Label:** {label}")
    st.write(f"**Fake Probability:** {fake:.4f}")
    st.write(f"**Real Probability:** {real:.4f}")

    # ================================================================
    #  SAVE LOG TO MONGODB
    # ================================================================
    try:
        log_entry = {
            "image_hash": compute_hash(image_bytes),
            "label": label,
            "fake_prob": fake,
            "real_prob": real,
            "timestamp": datetime.utcnow(),
            "image_base64": encode_image_base64(img)
        }

        logs.insert_one(log_entry)

        st.success("✔ Log saved to MongoDB successfully!")

    except Exception as e:
        st.error(f"❌ Error saving to MongoDB: {e}")
