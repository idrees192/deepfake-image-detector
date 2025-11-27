import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import io
import hashlib
import os

from utils.gradcam_utils import load_model, predict_and_heatmap

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "model/deepfake_model.pth"
IMG_SIZE = 224
EVIDENCE_DIR = "evidence"
os.makedirs(EVIDENCE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_main_model():
    model = load_model(MODEL_PATH, device=DEVICE)
    return model

model = load_main_model()

# ----------------------------
# HELPERS
# ----------------------------
def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="Deepfake Image Detector", layout="centered")
st.title("Deepfake Image Detector (Single-File Streamlit Version)")
st.write("Upload an image to analyze whether it is REAL or FAKE.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img_bytes = uploaded_file.read()
    file_hash = sha256_bytes(img_bytes)

    st.write("Image Hash:")
    st.code(file_hash)

    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    st.image(pil_image, caption="Uploaded Image", width=350)

    if st.button("Analyze Image"):
        with st.spinner("Running model + Grad-CAM..."):

            # Run prediction + heatmap from your utils
            result = predict_and_heatmap(model, pil_image, device=DEVICE, img_size=IMG_SIZE)

            verdict = result["verdict"]
            confidence = float(result["confidence"])
            heatmap = result["heatmap"]

            # Save evidence (optional)
            orig_path = os.path.join(EVIDENCE_DIR, f"{file_hash}.png")
            heatmap_path = os.path.join(EVIDENCE_DIR, f"{file_hash}_heatmap.png")

            pil_image.save(orig_path)

            if isinstance(heatmap, np.ndarray):
                heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
                cv2.imwrite(heatmap_path, heatmap_bgr)

        # ----------------------------
        # DISPLAY RESULTS
        # ----------------------------
        st.subheader("Result")
        st.write(f"**Verdict:** {verdict}")
        st.write(f"**Confidence:** {confidence:.4f}")

        st.subheader("Grad-CAM Heatmap")
        st.image(heatmap, caption="Heatmap", width=350)

        st.success("Analysis completed successfully!")
