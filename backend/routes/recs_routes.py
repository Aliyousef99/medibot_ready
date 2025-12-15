from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

from ..auth.deps import get_current_user
from ..models.user import User
from ..models.lab_report import LabReport
from ..models.symptom_event import SymptomEvent
from ..models.recommendation import RecommendationSet
from ..db.session import get_db
from ..services import recs

router = APIRouter()

class RecsRequest(BaseModel):
    lab_id: Optional[str] = None
    labs: Optional[List[Dict[str, Any]]] = None
    symptoms: Optional[List[Dict[str, Any]]] = None
    patient_info: Optional[Dict[str, Any]] = None

@router.post("/generate", status_code=201, response_model_exclude_none=True)
def generate_recommendations(
    request: RecsRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Generate clinical recommendations based on lab results and symptoms.
    """
    labs = request.labs
    symptoms = request.symptoms
    patient_info = request.patient_info

    if request.lab_id:
        lab_report = db.query(LabReport).filter(LabReport.id == request.lab_id, LabReport.user_id == current_user.id).first()
        if not lab_report:
            raise HTTPException(status_code=404, detail="Lab report not found")
        # Use structured JSON payload from stored lab report; prefer explicit tests array when available
        structured = lab_report.structured_json or {}
        if isinstance(structured, dict) and "tests" in structured:
            labs = structured.get("tests") or []
        else:
            labs = structured if structured is not None else []

        from datetime import datetime, timedelta
        recent_symptoms = db.query(SymptomEvent).filter(
            SymptomEvent.user_id == current_user.id,
            SymptomEvent.created_at >= datetime.utcnow() - timedelta(days=7)
        ).all()
        symptoms = [{"description": getattr(s, "symptom_description", None) or getattr(s, "raw_text", "")} for s in recent_symptoms]

    if not labs and not symptoms:
        raise HTTPException(status_code=400, detail="Either labs or symptoms must be provided.")

    risk_tier = recs.score_risk(labs=labs, symptoms=symptoms, patient_info=patient_info)
    recommendations = recs.get_recommendations(risk_tier, labs=labs, symptoms=symptoms)
    patient_copy, llm_used = recs.render_patient_copy(recommendations["actions"])

    new_rec_set = RecommendationSet(
        user_id=current_user.id,
        risk_tier=risk_tier,
        actions=recommendations["actions"],
        text=patient_copy,
        llm_used=llm_used,
        rules_version="1.0"
    )
    db.add(new_rec_set)
    db.commit()
    db.refresh(new_rec_set)

    return new_rec_set
