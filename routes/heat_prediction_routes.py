from fastapi import APIRouter
from controllers.heat_prediction_controller import get_heat_prediction

router = APIRouter()

@router.get("/predict/{cattle_id}")
def predict_heat(cattle_id: str):
    return get_heat_prediction(cattle_id)
