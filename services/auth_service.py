from utils.jwt_utils import decode_token

class AuthService:
    @staticmethod
    def validate_token(token):
        payload = decode_token(token)
        return payload is not None
