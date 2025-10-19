# backend/routes/symptoms_routes.py
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.db.session import get_db
from backend.models.user import User, UserProfile
from backend.models.symptom_event import SymptomEvent
from backend.auth.deps import get_current_user
from backend.schemas.symptoms import SymptomAnalysisRequest, SymptomAnalysisResult
from backend.schemas.profile import UserProfileOut
from backend.services import symptoms as symptoms_service

router = APIRouter(prefix="/api/symptoms", tags=["symptoms"])
logger = logging.getLogger("medibot")


@router.post("/analyze", response_model=SymptomAnalysisResult, status_code=status.HTTP_200_OK)
def analyze_symptoms(
    payload: SymptomAnalysisRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Analyzes user-provided text for symptoms, classifies urgency, and persists the event.
    """
    profile = db.query(UserProfile).filter(UserProfile.user_id == str(current_user.id)).first()
    profile_data = UserProfileOut.model_validate(profile, from_attributes=True) if profile else None

    analysis_result = symptoms_service.analyze(payload.text, profile_data)

    # Persist the event
    symptom_event = SymptomEvent(
        user_id=str(current_user.id),
        raw_text=payload.text,
        result_json=analysis_result,
    )
    db.add(symptom_event)
    db.commit()

    return SymptomAnalysisResult(**analysis_result)