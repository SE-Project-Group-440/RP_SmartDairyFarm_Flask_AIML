# routes/cattle_disease_routes.py
from fastapi import APIRouter, File, Form, Header, UploadFile
from controllers.cattle_disease_controller import predict_cattle_disease

router = APIRouter(prefix="/api/cattle", tags=["Cattle Disease"])

@router.post("/predict")
async def predict(
    image: UploadFile | None = File(None),
    report: UploadFile | None = File(None),
    symptoms: str | None = Form(None),
    authorization: str | None = Header(None)
):
    return await predict_cattle_disease(
        image=image,
        report=report,
        symptoms=symptoms,
        authorization=authorization
    )
