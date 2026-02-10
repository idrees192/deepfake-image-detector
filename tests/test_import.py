"""
Test script for verifying all imports work correctly.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test if all required modules can be imported."""
    print("=" * 50)
    print("Testing Imports")
    print("=" * 50)
    
    try:
        print("Testing utils.preprocessing...")
        from utils.preprocessing import preprocess_image
        print("[OK] utils.preprocessing imported successfully")
    except Exception as e:
        print(f"[ERROR] Error importing utils.preprocessing: {e}")
        return False
    
    try:
        print("Testing model.deepfake_model...")
        from model.deepfake_model import predict, load_model
        print("[OK] model.deepfake_model imported successfully")
    except Exception as e:
        print(f"[ERROR] Error importing model.deepfake_model: {e}")
        return False
    
    try:
        print("Testing config...")
        from config import MODEL_PATH, ALLOWED_EXTENSIONS
        print("[OK] config imported successfully")
    except Exception as e:
        print(f"[ERROR] Error importing config: {e}")
        return False
    
    print("\n[SUCCESS] All imports successful!")
    return True

if __name__ == "__main__":
    test_imports()
