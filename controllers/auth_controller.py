from fastapi import HTTPException
from pydantic import BaseModel, Field
from services.auth_service import AuthService


class LoginRequest(BaseModel):
    username: str | None = None
    email: str | None = None
    password: str = Field(min_length=1)


class AuthController:
    @staticmethod
    def login(data: LoginRequest):
        identifier = (data.username or data.email or "").strip()
        if not identifier:
            raise HTTPException(status_code=422, detail="username or email is required")

        if not AuthService.authenticate(identifier, data.password):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        token = AuthService.issue_token(identifier)
        return {
            "message": "Login successful",
            "access_token": token,
            "token": token,
            "token_type": "bearer"
        }
