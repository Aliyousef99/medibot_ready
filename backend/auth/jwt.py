# backend/auth/jwt.py
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

from jose import jwt, JWTError  # <-- python-jose
from passlib.hash import pbkdf2_sha256, bcrypt

# Config
SECRET_KEY = os.getenv("JWT_SECRET", "dev-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# ---- Password hashing ----
def hash_password(password: str) -> str:
    if not password:
        raise ValueError("Password cannot be empty")

    # Avoid double hashing if the caller passes a pre-hashed value.
    if password.startswith("$pbkdf2-sha256$") or password.startswith("$2"):
        return password

    return pbkdf2_sha256.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    if not password_hash:
        return False

    # Primary: pbkdf2 (current scheme)
    try:
        if password_hash.startswith("$pbkdf2-sha256$") and pbkdf2_sha256.verify(password, password_hash):
            return True
    except Exception:
        pass

    # Fallback: legacy bcrypt hashes
    try:
        if password_hash.startswith("$2") and bcrypt.verify(password, password_hash):
            return True
    except Exception:
        pass

    # Final fallback: plain-text match for legacy dev databases
    return password == password_hash

# ---- JWT helpers ----
def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_refresh_token(token: str) -> Optional[Dict[str, Any]]:
    payload = decode_token(token)
    if not payload:
        return None
    if payload.get("type") != "refresh":
        return None
    return payload

def decode_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

def get_current_user_from_token(token: str) -> Optional[Dict[str, Any]]:
    payload = decode_token(token)
    if not payload:
        return None
    # Expecting a "sub" claim to carry the username or user id
    sub = payload.get("sub")
    if not sub:
        return None
    return payload
