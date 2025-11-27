import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000/detect" 
# Change to deployed URL if backend is online

st.title("Deepfake Image Detector (Security Enhanced)")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=300)

    if st.button("Analyze"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post(BACKEND_URL, files=files)

        if response.status_code != 200:
            st.error("Server error. Try again.")
        else:
            data = response.json()

            if "error" in data:
                st.error(data["error"])
            else:
                st.success(f"Result: {data['label'].upper()}")
                st.write(f"Confidence: {data['confidence']:.3f}")
                st.write(f"Risk Level: {data['risk']}")
                st.write(f"Image Hash: `{data['hash']}`")

                st.subheader("Verification Token (JWT)")
                st.code(data["token"], language="text")
