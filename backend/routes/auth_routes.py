# backend/routes/auth_routes.py
from fastapi import APIRouter, Depends, HTTPException, Body, status, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, constr
from backend.db.session import get_db
from backend.models.user import User
from backend.auth.jwt import verify_password, create_access_token, create_refresh_token, verify_refresh_token, hash_password

router = APIRouter(prefix="/api/auth", tags=["auth"])

class LoginAny(BaseModel):
    email: EmailStr | None = None
    username: EmailStr | None = None
    password: str

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
    password: constr(min_length=6, max_length=128)


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(payload: RegisterIn = Body(...), db: Session = Depends(get_db)):
    """Create a new user account.
    Frontend calls this then immediately logs in, so we just return 201 on success.
    """
    existing = db.query(User).filter(User.email == str(payload.email)).first()
    if existing:
        raise HTTPException(status_code=409, detail="User with this email already exists")

    user = User(
        email=str(payload.email),
        hashed_password=hash_password(payload.password),
        name=None,
    )
    db.add(user)
    db.commit()
    refresh = create_refresh_token({"sub": str(user.id), "email": user.email})
    access = create_access_token({"sub": str(user.id), "email": user.email})
    return {"status": "created", "access_token": access, "refresh_token": refresh}


# ---- Email verification scaffold (placeholder) ----
class VerifyRequest(BaseModel):
    email: EmailStr


@router.post("/request_verification", status_code=status.HTTP_202_ACCEPTED)
def request_verification(payload: VerifyRequest):
    """Stub endpoint for requesting a verification email."""
    # In a production setup, enqueue email with a signed token here.
    return {
        "status": "queued",
        "message": "Verification email flow not yet implemented.",
        "email": str(payload.email),
    }


@router.get("/verify_email")
def verify_email(token: str):
    """Stub endpoint for verifying an email token."""
    # In a production setup, decode token and mark user as verified.
    return {"status": "pending", "message": "Email verification is not yet implemented.", "token": token}


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
def google_oauth_start(request: Request):
    """Return a placeholder Google OAuth authorization URL."""
    redirect_uri = str(request.url_for("google_oauth_callback"))
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?redirect_uri={redirect_uri}&client_id=your-google-client-id&response_type=code&scope=openid%20email%20profile"
    return {"auth_url": auth_url}


@router.get("/google/callback", name="google_oauth_callback")
def google_oauth_callback(code: str = "", state: str = ""):
    """Stub callback handler for Google OAuth."""
    # Exchange 'code' for tokens and create/login the user in a real implementation.
    raise HTTPException(status_code=501, detail="Google OAuth not yet implemented")
