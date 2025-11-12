from jose import jwt, JWTError, ExpiredSignatureError
from app.core.config import settings
from fastapi import HTTPException, Depends, Request
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.user import User

ALGO = "HS256"

def _now_ts() -> int:
    return int(datetime.now(timezone.utc).timestamp())

def create_access_token(sub: str, minutes: int = 60) -> str:
    now = _now_ts()
    payload = {
        "sub": str(sub),
        "iat": now,
        "exp": now + int(minutes * 60),   # LƯU exp dưới dạng int (epoch seconds)
    }
    # ✅ encode cần algorithm là STRING, không phải list
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=ALGO)

def decode_token(token: str):
    # ✅ decode cho phép truyền danh sách algorithms
    return jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGO])

def verify_access_token_from_cookie(request: Request) -> str:
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Missing access token")
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGO])
        uid = payload.get("sub")
        if not uid:
            raise HTTPException(status_code=401, detail="Invalid token")
        return str(uid)
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    uid = verify_access_token_from_cookie(request)
    user = db.query(User).filter(User.id == uid).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user
