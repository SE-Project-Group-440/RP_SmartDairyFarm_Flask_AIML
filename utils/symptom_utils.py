def analyze_symptoms(text):
    text = text.lower()
    score = {"FMD":0, "LSD":0}

    if "salivation" in text or "blisters" in text:
        score["FMD"] += 2

    if "nodules" in text or "skin lesions" in text:
        score["LSD"] += 2

    return score
