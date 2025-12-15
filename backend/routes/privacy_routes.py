from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.db.session import get_db
from backend.auth.deps import get_current_user
from backend.models.user import User, UserProfile
from backend.models.lab_report import LabReport
from backend.models.recommendation import RecommendationSet
from backend.models.symptom_event import SymptomEvent
from backend.models.conversation import Conversation
from backend.models.message import Message

router = APIRouter(prefix="/api/privacy", tags=["privacy"])


@router.get("/consent")
def get_consent(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    prof = db.query(UserProfile).filter(UserProfile.user_id == str(user.id)).first()
    if not prof:
        prof = UserProfile(user_id=str(user.id), consent_given=False)
        db.add(prof)
        db.commit()
        db.refresh(prof)
    return {"consent_given": bool(prof.consent_given), "consent_at": prof.consent_at}


@router.post("/consent", status_code=status.HTTP_200_OK)
def set_consent(payload: dict, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    consent = payload.get("consent_given")
    if consent is None:
        raise HTTPException(status_code=422, detail="consent_given is required")
    prof = db.query(UserProfile).filter(UserProfile.user_id == str(user.id)).first()
    if not prof:
        prof = UserProfile(user_id=str(user.id))
    prof.consent_given = bool(consent)
    prof.consent_at = datetime.utcnow() if prof.consent_given else None
    db.add(prof)
    db.commit()
    db.refresh(prof)
    return {"consent_given": prof.consent_given, "consent_at": prof.consent_at}


@router.delete("/delete_data", status_code=status.HTTP_204_NO_CONTENT)
def delete_all_user_data(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    uid = str(user.id)
    # Delete dependent records (manual because some FKs are not cascaded)
    db.query(Message).filter(Message.user_id == uid).delete(synchronize_session=False)
    db.query(Conversation).filter(Conversation.user_id == uid).delete(synchronize_session=False)
    db.query(LabReport).filter(LabReport.user_id == uid).delete(synchronize_session=False)
    db.query(RecommendationSet).filter(RecommendationSet.user_id == uid).delete(synchronize_session=False)
    db.query(SymptomEvent).filter(SymptomEvent.user_id == uid).delete(synchronize_session=False)
    db.query(UserProfile).filter(UserProfile.user_id == uid).delete(synchronize_session=False)
    db.commit()
    return None
