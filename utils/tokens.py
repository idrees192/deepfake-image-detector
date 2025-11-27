import jwt
import datetime

SECRET = "MY_SECRET_KEY"  # change this

def generate_verification_token(image_hash, label, confidence, model_version="v1"):
    payload = {
        "hash": image_hash,
        "label": label,
        "confidence": confidence,
        "model_version": model_version,
        "timestamp": str(datetime.datetime.utcnow())
    }
    return jwt.encode(payload, SECRET, algorithm="HS256")

