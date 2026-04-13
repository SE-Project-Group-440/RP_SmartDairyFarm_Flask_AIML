import os
from utils.jwt_utils import create_token, decode_token

class AuthService:
    @staticmethod
    def validate_token(token):
        payload = decode_token(token)
        return payload is not None

    @staticmethod
    def authenticate(identifier: str, password: str):
        if not identifier or not password:
            return False

        configured_user = os.getenv("API_LOGIN_USERNAME")
        configured_password = os.getenv("API_LOGIN_PASSWORD")

        # If explicit credentials are configured, enforce them.
        if configured_user and configured_password:
            return identifier == configured_user and password == configured_password

        # Development fallback: accept non-empty credentials.
        return True

    @staticmethod
    def issue_token(subject: str):
        return create_token(subject)
