# backend/routes/auth_routes.py
import base64
import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, HTTPException, Body, status, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, constr
from jose import jwt, JWTError
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from backend.db.session import get_db
from backend.models.user import User
from backend.auth.jwt import verify_password, create_access_token, create_refresh_token, verify_refresh_token, hash_password

router = APIRouter(prefix="/api/auth", tags=["auth"])

GOOGLE_OAUTH_STATE_SECRET = os.getenv("GOOGLE_OAUTH_STATE_SECRET", os.getenv("JWT_SECRET", "dev-secret-change-me"))
GOOGLE_OAUTH_STATE_TTL_SECONDS = int(os.getenv("GOOGLE_OAUTH_STATE_TTL_SECONDS", "600"))
GOOGLE_OAUTH_ALG = "HS256"
ALLOWED_EMAIL_DOMAINS = [
    d.strip().lower()
    for d in (os.getenv("ALLOWED_EMAIL_DOMAINS") or "gmail.com,yahoo.com,outlook.com,hotmail.com,live.com,icloud.com,aol.com,protonmail.com").split(",")
    if d.strip()
]

def _email_domain_allowed(email: str) -> bool:
    try:
        domain = email.split("@", 1)[1].lower()
    except Exception:
        return False
    return domain in ALLOWED_EMAIL_DOMAINS


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

def _make_google_state(payload: dict) -> str:
    exp = datetime.now(timezone.utc) + timedelta(seconds=GOOGLE_OAUTH_STATE_TTL_SECONDS)
    body = payload.copy()
    body.update({"exp": exp, "typ": "google_oauth_state"})
    return jwt.encode(body, GOOGLE_OAUTH_STATE_SECRET, algorithm=GOOGLE_OAUTH_ALG)

def _decode_google_state(state: str) -> dict | None:
    try:
        data = jwt.decode(state, GOOGLE_OAUTH_STATE_SECRET, algorithms=[GOOGLE_OAUTH_ALG])
    except JWTError:
        return None
    if data.get("typ") != "google_oauth_state":
        return None
    return data

def _require_google_env() -> tuple[str, str]:
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise HTTPException(status_code=500, detail="Google OAuth is not configured")
    return client_id, client_secret

def _finish_google_oauth(code: str, state: str, request: Request, db: Session) -> dict:
    client_id, client_secret = _require_google_env()
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing OAuth parameters")

    state_data = _decode_google_state(state)
    if not state_data:
        raise HTTPException(status_code=400, detail="Invalid OAuth state")

    redirect_uri = state_data.get("redirect_uri") or os.getenv("GOOGLE_REDIRECT_URI") or str(request.url_for("google_oauth_callback"))
    token_payload = {
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
        "code_verifier": state_data.get("code_verifier"),
    }
    try:
        token_resp = httpx.post("https://oauth2.googleapis.com/token", data=token_payload, timeout=15)
        token_data = token_resp.json()
    except Exception:
        raise HTTPException(status_code=502, detail="Google token exchange failed")

    if token_resp.status_code != 200 or "id_token" not in token_data:
        raise HTTPException(status_code=401, detail="Google token exchange failed")

    try:
        idinfo = id_token.verify_oauth2_token(
            token_data["id_token"],
            google_requests.Request(),
            client_id,
        )
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Google ID token")

    if idinfo.get("nonce") != state_data.get("nonce"):
        raise HTTPException(status_code=401, detail="Invalid OAuth nonce")

    if idinfo.get("email_verified") is False:
        raise HTTPException(status_code=401, detail="Google account email is not verified")

    email = idinfo.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Google account has no email")
    if not _email_domain_allowed(str(email)):
        raise HTTPException(status_code=400, detail="Email domain is not allowed")

    user = db.query(User).filter(User.email == str(email)).first()
    if not user:
        user = User(
            email=str(email),
            name=idinfo.get("name"),
            hashed_password=hash_password(secrets.token_urlsafe(32)),
        )
        user.email_verified = True
        user.email_verification_token_hash = None
        user.email_verification_expires_at = None
        user.email_verification_sent_at = None
        db.add(user)
        db.commit()
        db.refresh(user)
    elif not user.name and idinfo.get("name"):
        user.name = idinfo.get("name")
        db.commit()
    if not user.email_verified:
        user.email_verified = True
        user.email_verification_token_hash = None
        user.email_verification_expires_at = None
        user.email_verification_sent_at = None
        db.commit()

    access = create_access_token({"sub": str(user.id), "email": user.email})
    refresh = create_refresh_token({"sub": str(user.id), "email": user.email})
    response = {
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "bearer",
        "email": user.email,
    }
    redirect = state_data.get("redirect")
    if redirect:
        response["redirect"] = redirect
    return response

class LoginAny(BaseModel):
    email: EmailStr | None = None
    username: EmailStr | None = None
    password: constr(min_length=6, max_length=24)

@router.post("/login")
def login(payload: LoginAny = Body(...), db: Session = Depends(get_db)):
    email = payload.email or payload.username
    if not email:
        raise HTTPException(status_code=422, detail="Provide 'email' or 'username'")
    user = db.query(User).filter(User.email == str(email)).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"sub": str(user.id), "email": user.email})
    refresh = create_refresh_token({"sub": str(user.id), "email": user.email})
    return {"access_token": token, "refresh_token": refresh, "token_type": "bearer"}


class RegisterIn(BaseModel):
    email: EmailStr
    password: constr(min_length=6, max_length=24)


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(payload: RegisterIn = Body(...), db: Session = Depends(get_db)):
    """Create a new user account.
    Frontend calls this then immediately logs in, so we just return 201 on success.
    """
    if not _email_domain_allowed(str(payload.email)):
        raise HTTPException(status_code=400, detail="Email domain is not allowed")

    existing = db.query(User).filter(User.email == str(payload.email)).first()
    if existing:
        raise HTTPException(status_code=409, detail="User with this email already exists")

    user = User(
        email=str(payload.email),
        hashed_password=hash_password(payload.password),
        name=None,
    )
    user.email_verified = True
    db.add(user)
    db.commit()
    user.email_verification_token_hash = None
    user.email_verification_sent_at = None
    user.email_verification_expires_at = None
    db.commit()
    return {"status": "created"}


# ---- Email verification scaffold (placeholder) ----
class VerifyRequest(BaseModel):
    email: EmailStr


@router.post("/request_verification", status_code=status.HTTP_202_ACCEPTED)
def request_verification(payload: VerifyRequest):
    """Verification disabled; keep endpoint for compatibility."""
    return {"status": "disabled", "message": "Email verification is disabled.", "email": str(payload.email)}


@router.get("/verify_email")
def verify_email(token: str):
    """Verification disabled; keep endpoint for compatibility."""
    return {"status": "disabled", "message": "Email verification is disabled.", "token": token}


class RefreshIn(BaseModel):
    refresh_token: str


@router.post("/refresh")
def refresh(payload: RefreshIn, db: Session = Depends(get_db)):
    """Exchange a refresh token for a new access token."""
    data = verify_refresh_token(payload.refresh_token)
    if not data:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    sub = data.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    user = db.query(User).filter(User.id == str(sub)).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    access = create_access_token({"sub": str(user.id), "email": user.email})
    return {"access_token": access, "token_type": "bearer"}


# ---- Google OAuth stubs ----
@router.get("/google/start")
def google_oauth_start(request: Request, redirect: str | None = None):
    """Return a Google OAuth authorization URL."""
    client_id, _client_secret = _require_google_env()
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI") or str(request.url_for("google_oauth_callback"))
    nonce = secrets.token_urlsafe(24)
    code_verifier = secrets.token_urlsafe(64)
    code_challenge = _b64url(hashlib.sha256(code_verifier.encode("ascii")).digest())

    state = _make_google_state({
        "nonce": nonce,
        "code_verifier": code_verifier,
        "redirect_uri": redirect_uri,
        "redirect": redirect or "",
    })

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "nonce": nonce,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "access_type": "offline",
        "prompt": "consent",
    }
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
    format_hint = (request.query_params.get("format") or "").lower()
    accept = (request.headers.get("accept") or "").lower()
    if format_hint == "json" or "application/json" in accept:
        return {"auth_url": auth_url}
    return RedirectResponse(url=auth_url, status_code=302)


@router.get("/google/callback", name="google_oauth_callback")
def google_oauth_callback(request: Request, code: str = "", state: str = "", db: Session = Depends(get_db)):
    """Callback handler for Google OAuth."""
    return _finish_google_oauth(code=code, state=state, request=request, db=db)


class GoogleCompleteIn(BaseModel):
    code: str
    state: str


@router.post("/google/complete")
def google_oauth_complete(payload: GoogleCompleteIn, request: Request, db: Session = Depends(get_db)):
    """Complete Google OAuth from a server-side callback proxy."""
    return _finish_google_oauth(code=payload.code, state=payload.state, request=request, db=db)
