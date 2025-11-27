import json
from datetime import datetime

LOG_PATH = "logs/detections.jsonl"

def log_detection(image_hash, label, confidence):
    entry = {
        "timestamp": str(datetime.utcnow()),
        "image_hash": image_hash,
        "result": label,
        "confidence": confidence
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

