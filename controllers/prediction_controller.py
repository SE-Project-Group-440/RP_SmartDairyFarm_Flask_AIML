from services.Lactation_Service import ModelService
from services.auth_service import AuthService
from flask import jsonify, request

model_service = ModelService()

class PredictionController:

    @staticmethod
    def predict():
        auth = request.headers.get("Authorization", "")
        token = auth.replace("Bearer ", "")
        
        if not AuthService.validate_token(token):
            return jsonify({"error": "Unauthorized"}), 401
        
        data = request.get_json()
        features = data.get("features")

        prediction = model_service.predict(features)
        return jsonify({"prediction": prediction})
