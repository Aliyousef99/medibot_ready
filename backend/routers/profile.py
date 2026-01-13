# backend/routes/profile.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import cast, Text as SAText
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
    try:
        db.rollback()
    except Exception:
        pass
    prof = db.query(UserProfile).filter(cast(UserProfile.user_id, SAText) == str(user.id)).first()
    if not prof:
        # create an empty profile on first read (optional, but handy)
        prof = UserProfile(user_id=user.id)
        db.add(prof)
        db.commit()
        db.refresh(prof)
    data = UserProfileOut.model_validate(prof, from_attributes=True).model_dump()
    data["name"] = user.name
    return data

@router.put("/", response_model=UserProfileOut)
def upsert_profile(
    payload: UserProfileIn,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    try:
        db.rollback()
    except Exception:
        pass
    prof = db.query(UserProfile).filter(cast(UserProfile.user_id, SAText) == str(user.id)).first()
    if not prof:
        prof = UserProfile(user_id=user.id)

    prof.age = payload.age
    prof.sex = payload.sex
    # store lists as lists (your model uses JSON columns for these)
    prof.conditions = payload.conditions or []
    prof.medications = payload.medications or []
    prof.notes = payload.notes
    if payload.name is not None:
        next_name = payload.name.strip()
        user.name = next_name or None
    if payload.consent_given is not None:
        prof.consent_given = bool(payload.consent_given)
        if prof.consent_given:
            from datetime import datetime
            prof.consent_at = datetime.utcnow()
        else:
            prof.consent_at = None

    db.add(prof)
    db.commit()
    db.refresh(prof)
    data = UserProfileOut.model_validate(prof, from_attributes=True).model_dump()
    data["name"] = user.name
    return data
