"""
Image Preprocessing Module
Handles image preprocessing for the deepfake detection model.
"""

from PIL import Image
import numpy as np


def _import_tf():
    try:
        import tensorflow as tf
        return tf
    except Exception as e:
        raise ModuleNotFoundError(
            "TensorFlow is required but not installed. Install it with `pip install tensorflow`."
        )

def preprocess_image(image_file):
    """
    Prepares an uploaded image for the EfficientNet model.
    
    Processing steps:
    1. Opens image and converts to RGB (removes Alpha channel if present)
    2. Resizes to 224x224 pixels (model input size)
    3. Converts to numpy array
    4. Expands dimensions to create batch dimension
    
    Args:
        image_file: File-like object or path to image file
    
    Returns:
        numpy.ndarray: Preprocessed image array with shape (1, 224, 224, 3)
                       Ready for model input
    """
    # 1. Open image and ensure RGB
    image = Image.open(image_file).convert('RGB')

    # 2. Resize to match training size
    image = image.resize((224, 224))

    # 3. Convert to array (lazy import of tensorflow to provide clearer error)
    tf = _import_tf()
    img_array = tf.keras.utils.img_to_array(image)

    # 4. Expand dimensions to fit model input: (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Note: We do NOT divide by 255 here because EfficientNet handles it internally.
    return img_array