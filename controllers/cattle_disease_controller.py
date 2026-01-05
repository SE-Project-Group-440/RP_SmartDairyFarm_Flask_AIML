# controllers/cattle_disease_controller.py
from fastapi import UploadFile, HTTPException
from services.auth_service import AuthService
from services.cattle_disease_service import run_prediction

async def predict_cattle_disease(
    image: UploadFile | None = None,
    report: UploadFile | None = None,
    symptoms: str | None = None,
    authorization: str | None = None
):
    token = (authorization or "").replace("Bearer ", "")

    # Validate token
    if not AuthService.validate_token(token):
        raise HTTPException(status_code=401, detail="Unauthorized")

    # At least one input must be provided
    if not any([image, report, symptoms]):
        raise HTTPException(
            status_code=400,
            detail="At least one of image, report, or symptoms must be provided"
        )

    result = await run_prediction(image, report, symptoms)
    return result
