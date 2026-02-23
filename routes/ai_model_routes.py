from fastapi import APIRouter, Header
from controllers.ai_model_controller import PredictionController, PredictRequest

router = APIRouter()

@router.post("/ai/recommend")
def recommend(data: PredictRequest):
    return PredictionController.recommend(data)

@router.post("/ai/pregnancy")
def pregnancy(data: PredictRequest):
    return PredictionController.pregnancy(data)