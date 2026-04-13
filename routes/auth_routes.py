from fastapi import APIRouter
from controllers.auth_controller import AuthController, LoginRequest

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/login")
def login(data: LoginRequest):
    return AuthController.login(data)
