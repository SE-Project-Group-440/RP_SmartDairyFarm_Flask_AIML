from fastapi import HTTPException, Header
from pydantic import BaseModel
from services.ai_model_service import predict_pregnancy
from services.ai_model_service import (
    predict_pregnancy,
    recommend_next_ai
)

class PredictRequest(BaseModel):
    row: dict  # Accept a single cow's row as dict

class PredictionController:

    @staticmethod
    def recommend(data: PredictRequest):
        return recommend_next_ai(data.row)

    @staticmethod
    def pregnancy(data: PredictRequest):
        return predict_pregnancy(data.row)