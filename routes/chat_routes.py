from fastapi import APIRouter, Request
from controllers.chat_controller import chat_controller

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("/")
async def chat_endpoint(req: Request):
    base_url = str(req.base_url).rstrip("/")
    return chat_controller(await req.json(), base_url)