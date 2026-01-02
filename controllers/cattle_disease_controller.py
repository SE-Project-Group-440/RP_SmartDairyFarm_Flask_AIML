from flask import request, jsonify
from services.cattle_disease_service import run_prediction

def predict_cattle_disease():
    image = request.files.get("image")
    report = request.files.get("report")
    symptoms = request.form.get("symptoms")

    result = run_prediction(image, report, symptoms)
    return jsonify(result)
