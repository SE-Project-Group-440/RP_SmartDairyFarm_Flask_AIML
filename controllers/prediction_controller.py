from fastapi import HTTPException, Header
from pydantic import BaseModel
from services.Lactation_Service import ModelService
from services.auth_service import AuthService

model_service = ModelService()

class PredictRequest(BaseModel):
    features: list

class PredictionController:

    @staticmethod
    def predict(data: PredictRequest, authorization: str | None):
        # Extract token
        token = (authorization or "").replace("Bearer ", "")

        # Validate token
        if not AuthService.validate_token(token):
            raise HTTPException(status_code=401, detail="Unauthorized")

        # Prediction
        prediction = model_service.predict(data.features)
        return {"prediction": prediction}
