"""
Test script for model loading functionality.
This verifies that the model can be loaded correctly.
"""
import os
import sys
import tensorflow as tf

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_PATH
from model.deepfake_model import load_model

def test_model_loading():
    """Test if the model loads successfully."""
    print("=" * 50)
    print("Testing Model Loading")
    print("=" * 50)
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Model Path: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Model file not found at path.")
        return False
    
    try:
        model = load_model()
        if model is not None:
            print("[SUCCESS] Model loaded successfully!")
            print(f"Model Summary:")
            model.summary()
            return True
        else:
            print("[ERROR] Model loading returned None")
            return False
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_loading()
