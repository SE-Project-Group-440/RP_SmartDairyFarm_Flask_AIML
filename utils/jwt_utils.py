import jwt
from datetime import datetime, timedelta, timezone
from config import SECRET_KEY


def create_token(subject: str, expires_minutes: int = 60 * 24):
    expiry = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    payload = {"sub": subject, "exp": expiry}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def decode_token(token):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None
