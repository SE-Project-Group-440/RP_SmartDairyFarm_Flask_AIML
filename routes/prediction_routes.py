from fastapi import APIRouter, Header
from controllers.prediction_controller import PredictionController, PredictRequest

router = APIRouter()

@router.post("/predict")
def predict(
    data: PredictRequest,
    authorization: str | None = Header(default=None)
):
    return PredictionController.predict(data, authorization)
