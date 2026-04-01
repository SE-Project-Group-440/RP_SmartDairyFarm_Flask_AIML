def analyze_symptoms(text):
    text = text.lower()
    score = {"FMD": 0, "LSD": 0}
    
    # Define maximum possible scores for confidence calculation
    max_scores = {"FMD": 4, "LSD": 4}  # Max possible: 2+1+1 for each

    # FMD symptoms
    if "salivation" in text or "blisters" in text:
        score["FMD"] += 2

    if "fever" in text or "lameness" in text:
        score["FMD"] += 1

    # LSD symptoms
    if "nodules" in text or "skin lesions" in text:
        score["LSD"] += 2

    if "swollen lymph nodes" in text or "nasal discharge" in text:
        score["LSD"] += 1

    # Reduced appetite
    if "reduced appetite" in text:
        score["FMD"] += 1
        score["LSD"] += 1

    # Calculate confidence based on the highest scoring disease
    max_disease = max(score, key=score.get)
    max_score = score[max_disease]
    confidence = min(max_score / max_scores[max_disease], 1.0) if max_score > 0 else 0.0
    
    return {
        "scores": score,
        "confidence": confidence,
        "predicted_disease": max_disease if max_score > 0 else "Healthy"
    }