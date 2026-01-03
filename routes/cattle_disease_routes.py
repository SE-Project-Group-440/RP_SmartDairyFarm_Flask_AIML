from fastapi import APIRouter, Header
from controllers.cattle_disease_controller import predict_cattle_disease, CattlePredictRequest

router = APIRouter(prefix="/api/cattle", tags=["Cattle Disease"])

@router.post("/predict")
def predict(
    data: CattlePredictRequest,
    authorization: str | None = Header(default=None)
):
    return predict_cattle_disease(data, authorization)
