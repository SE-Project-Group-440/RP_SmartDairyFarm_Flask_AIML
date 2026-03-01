from fastapi import APIRouter, UploadFile, File
from controllers.stt_controller import stt_controller

router = APIRouter(prefix="/stt", tags=["STT"])

@router.post("/")
async def stt_endpoint(audio: UploadFile = File(...)):
    return await stt_controller(audio)