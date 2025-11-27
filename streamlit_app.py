import streamlit as st
import torch
from PIL import Image
import numpy as np
import torchvision.transforms as T
import timm

# -------------- LOAD MODEL --------------
@st.cache_resource
def load_model():
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    model.load_state_dict(torch.load("model/deepfake_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -------------- PREPROCESS --------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

# -------------- UI --------------
st.title("Deepfake Image Detector (Simple Version)")
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    st.image(uploaded, caption="Uploaded Image", use_column_width=True)

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
