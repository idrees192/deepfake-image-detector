"""
Configuration Module
Contains all configuration settings for the deepfake detection system.
"""

import os

# Get the absolute path of the current file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model configuration
MODEL_NAME = "deepfake_universal_patched.keras"
MODEL_PATH = os.path.join(BASE_DIR, "model", MODEL_NAME)

# Image processing configuration
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
IMAGE_SIZE = (224, 224)  # Model input size