from fastapi import APIRouter
from controllers.cattle_disease_controller import predict_cattle_disease

router = APIRouter(prefix="/api/cattle", tags=["Cattle Disease"])

@router.post("/predict")
def predict(
    image=File(...),
    report=File(None),
    symptoms=Form(None),
    authorization=Header(None)
):
    return predict_cattle_disease(
        image=image,
        report=report,
        symptoms=symptoms,
        authorization=authorization
    )
