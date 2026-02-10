"""
Deepfake Detection Model
This module handles model loading and prediction for deepfake image detection.

The model uses EfficientNet-B0 as a base architecture with custom classification layers.
"""

from config import MODEL_PATH
import os


def _import_tf():
    try:
        import tensorflow as tf
        return tf
    except Exception:
        raise ModuleNotFoundError(
            "TensorFlow is required but not installed. Install it with `pip install tensorflow`."
        )

# Cache the model so we don't reload it for every user request
_model = None

def load_model():
    """
    Constructs the model architecture and loads pre-trained weights.
    
    The model architecture:
    - Base: EfficientNet-B0 (pre-trained on ImageNet, frozen)
    - Global Average Pooling
    - Batch Normalization
    - Dense layer (256 units, ReLU)
    - Dropout (0.5)
    - Output layer (1 unit, Sigmoid)
    
    Returns:
        tf.keras.Model: Loaded and compiled model, or None if loading fails
    """
    global _model
    
    if _model is not None:
        return _model

    print(f"Constructing model architecture...")
    try:
        tf = _import_tf()
        # Reconstruct the exact architecture from the training code
        base_model = tf.keras.applications.EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )
        base_model.trainable = False # It was frozen during training

        _model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])
        
        # Determine weights path (extracted .h5 file)
        weights_path = os.path.join(os.path.dirname(MODEL_PATH), "model.weights.h5")
        
        print(f"Loading weights from: {weights_path}...")
        
        # Build the model with a sample input to initialize variables before loading weights
        _model.build((None, 224, 224, 3))
        
        _model.load_weights(weights_path)
        print("Model loaded successfully (reconstructed)!")
        return _model
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict(image_array):
    """
    Runs prediction on a preprocessed image array.
    
    Args:
        image_array: Preprocessed image array with shape (1, 224, 224, 3)
    
    Returns:
        float: Prediction score between 0.0 (Fake) and 1.0 (Real)
               Returns 0.5 if model loading fails (neutral/uncertain)
    """
    model = load_model()
    if model is None:
        return 0.5  # Return neutral/failure confidence

    # Model prediction returns shape (1, 1), extract the single value
    prediction_score = float(model.predict(image_array, verbose=0)[0][0])
    
    return prediction_score