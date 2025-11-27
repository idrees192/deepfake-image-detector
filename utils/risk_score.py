def compute_risk(confidence):
    if confidence > 0.90:
        return "LOW"
    elif confidence > 0.60:
        return "MEDIUM"
    else:
        return "HIGH"

