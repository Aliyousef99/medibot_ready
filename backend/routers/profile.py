# backend/routes/profile.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.db.session import get_db
from backend.auth.deps import get_current_user
from backend.models.user import User, UserProfile
from backend.schemas.profile import UserProfileIn, UserProfileOut

router = APIRouter(prefix="/api/profile", tags=["profile"])

@router.get("/", response_model=UserProfileOut)
def get_profile(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    prof = db.query(UserProfile).filter(UserProfile.user_id == user.id).first()
    if not prof:
        # create an empty profile on first read (optional, but handy)
        prof = UserProfile(user_id=user.id)
        db.add(prof)
        db.commit()
        db.refresh(prof)
    return UserProfileOut.model_validate(prof, from_attributes=True)

@router.put("/", response_model=UserProfileOut)
def upsert_profile(
    payload: UserProfileIn,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    prof = db.query(UserProfile).filter(UserProfile.user_id == user.id).first()
    if not prof:
        prof = UserProfile(user_id=user.id)

    prof.age = payload.age
    prof.sex = payload.sex
    # store lists as lists (your model uses JSON columns for these)
    prof.conditions = payload.conditions or []
    prof.medications = payload.medications or []
    prof.notes = payload.notes

    db.add(prof)
    db.commit()
    db.refresh(prof)
    return UserProfileOut.model_validate(prof, from_attributes=True)
