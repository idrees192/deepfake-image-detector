import streamlit as st
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as T
import timm

# -------------------------
# MODEL DEFINITION (MATCH TRAINING EXACTLY)
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
# PREPROCESS
# -------------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

st.title("Deepfake Image Detector — SIMPLE VERSION")

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded:
    st.image(uploaded)

    img = Image.open(uploaded).convert("RGB")
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
