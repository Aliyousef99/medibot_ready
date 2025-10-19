# backend/routes/auth_routes.py
from fastapi import APIRouter, Depends, HTTPException, Body, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, constr
from backend.db.session import get_db
from backend.models.user import User
from backend.auth.jwt import verify_password, create_access_token, hash_password

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
    return {"access_token": token, "token_type": "bearer"}


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
    # No token returned here; the UI performs a separate login call.
    return {"status": "created"}
