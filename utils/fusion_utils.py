import json

with open("Model/cattle_multimodal/thresholds.json") as f:
    THRESHOLDS = json.load(f)

def classify_severity(confidence, disease, symptom_score=0):
    """
    Classify disease severity based on confidence and symptom intensity
    """
    if disease == "Uncertain" or disease == "Healthy":
        return "none"
    
    # Get confidence-based severity
    if confidence >= THRESHOLDS["severity_classification"]["confidence_thresholds"]["severe"]:
        confidence_severity = "severe"
    elif confidence >= THRESHOLDS["severity_classification"]["confidence_thresholds"]["moderate"]:
        confidence_severity = "moderate" 
    elif confidence >= THRESHOLDS["severity_classification"]["confidence_thresholds"]["mild"]:
        confidence_severity = "mild"
    else:
        return "uncertain"
    
    # Factor in symptom intensity if available
    if symptom_score > 0 and disease in THRESHOLDS["severity_classification"]["symptom_severity_multipliers"]:
        multipliers = THRESHOLDS["severity_classification"]["symptom_severity_multipliers"][disease]
        
        if symptom_score >= multipliers["severe"]:
            symptom_severity = "severe"
        elif symptom_score >= multipliers["moderate"]:
            symptom_severity = "moderate"
        else:
            symptom_severity = "mild"
        
        # Take the higher severity between confidence and symptoms
        severities = ["mild", "moderate", "severe"]
        confidence_idx = severities.index(confidence_severity)
        symptom_idx = severities.index(symptom_severity)
        
        final_severity = severities[max(confidence_idx, symptom_idx)]
        
        return {
            "level": final_severity,
            "factors": {
                "confidence_contribution": confidence_severity,
                "symptom_contribution": symptom_severity,
                "confidence_score": confidence,
                "symptom_score": symptom_score
            }
        }
    
    # Confidence-only severity
    return {
        "level": confidence_severity,
        "factors": {
            "confidence_contribution": confidence_severity,
            "confidence_score": confidence
        }
    }

def fuse_results(result):
    """
    Fuse results from multiple modalities and calculate overall confidence
    """
    prediction = THRESHOLDS["fallback_prediction"]
    confidence = 0.0
    confidence_sources = []
    symptom_score = 0
    
    # Image has highest priority
    if "image_prediction" in result:
        prediction = result["image_prediction"]
        if "image_confidence" in result:
            confidence = result["image_confidence"]
            confidence_sources.append(f"image({confidence:.2f})")
        else:
            confidence = 0.85  # Default high confidence for image predictions
            confidence_sources.append("image(0.85)")
        
        # Get symptom score if available for severity calculation
        if "symptoms_analysis" in result and prediction in result["symptoms_analysis"]:
            symptom_score = result["symptoms_analysis"][prediction]
        
        severity = classify_severity(confidence, prediction, symptom_score)
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "severity": severity,
            "confidence_breakdown": {
                "primary_source": "image",
                "sources": confidence_sources,
                "overall": confidence
            }
        }

    # Symptom-based decision
    if "symptoms_analysis" in result:
        symptom_confidence = result.get("symptom_confidence", 0.0)
        
        for disease, score in result["symptoms_analysis"].items():
            if score >= THRESHOLDS["symptom_thresholds"].get(disease, 99):
                prediction = disease
                confidence = symptom_confidence
                symptom_score = score
                confidence_sources.append(f"symptoms({confidence:.2f})")
                
                severity = classify_severity(confidence, prediction, symptom_score)
                
                return {
                    "prediction": prediction,
                    "confidence": confidence,
                    "severity": severity,
                    "confidence_breakdown": {
                        "primary_source": "symptoms",
                        "sources": confidence_sources,
                        "symptom_scores": result["symptoms_analysis"],
                        "overall": confidence
                    }
                }
    
    # Blood report analysis (if implemented in future)
    if "blood_report" in result:
        confidence_sources.append("blood_report(available)")
    
    # Fallback with low confidence
    confidence = 0.1
    confidence_sources.append("fallback(0.1)")
    severity = classify_severity(confidence, prediction)
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "severity": severity,
        "confidence_breakdown": {
            "primary_source": "fallback",
            "sources": confidence_sources,
            "overall": confidence
        }
    }
