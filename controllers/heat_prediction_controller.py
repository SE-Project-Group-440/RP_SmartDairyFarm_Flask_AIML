from services.heat_prediction_service import predict_thi_1hr

def get_heat_prediction(cattle_id: str):
    
    cattle_id = cattle_id.capitalize()

    thi, latest = predict_thi_1hr(cattle_id)

    if thi is None:
        return {
            "cattle_id": cattle_id,
            "error": "Not enough data (need 6 readings, 20-min interval)"
        }

    if thi >= 88:
        level = "Severe"
    elif thi >= 79:
        level = "High"
    elif thi >= 72:
        level = "Mild"
    else:
        level = "Normal"

    return {
        "cattle_id": cattle_id,

        "body_temp": latest.get("bodyTemp"),
        "env_temp": latest.get("envTemp"),
        "humidity": latest.get("humidity"),
        "thi_index_1hr_ahead": thi,
        "stress_level": level
    }
