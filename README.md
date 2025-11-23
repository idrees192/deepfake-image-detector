# deepfake-image-detector
A deepfake image detection project using EfficientNet + Grad-CAM + Gradio
# Deepfake Image Detector (Image Only)

This repo contains an Image-based Deepfake Detector built using:

- EfficientNet-B0
- PyTorch
- Grad-CAM explainability
- FastAPI backend
- Gradio frontend (live demo)

---

## 🚀 Features
✔ Detect if image is REAL or FAKE  
✔ Show Grad-CAM heatmap  
✔ Provide Confidence Score  
✔ Gradio Web App  
✔ FastAPI backend for public API  
✔ Export to ONNX (coming soon)

---

## 📂 Folder Structure

backend/ → FastAPI server  
frontend/ → Gradio UI  
utils/ → Grad-CAM + preprocessing code  
model/ → Put your trained model here  
requirements.txt → Dependencies  
README.md → Project docs

---

## 🧰 How To Run Locally

1. Install dependencies:
   pip install -r requirements.txt
2. Run backend:
   uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
3. Run Gradio frontend:

