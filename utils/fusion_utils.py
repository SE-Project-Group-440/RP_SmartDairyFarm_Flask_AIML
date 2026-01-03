import json

with open("Model/cattle_multimodal/thresholds.json") as f:
    THRESHOLDS = json.load(f)

def fuse_results(result):
    # Image has highest priority
    if "image_prediction" in result:
        return result["image_prediction"]

    # Symptom-based decision
    if "symptoms" in result:
        for disease, score in result["symptoms"].items():
            if score >= THRESHOLDS["symptom_thresholds"].get(disease, 99):
                return disease

    # Fallback
    return THRESHOLDS["fallback_prediction"]
