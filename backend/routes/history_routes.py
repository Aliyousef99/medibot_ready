# backend/routes/history_routes.py
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from sqlalchemy.orm import Session

from backend.db.session import get_db
from backend.models.user import User
from backend.models.lab_report import LabReport
from backend.schemas.profile import LabReportCreate, LabReportOut
from backend.services.report_pipeline import process_text
from backend.auth.deps import get_current_user

router = APIRouter(prefix="/api/history", tags=["history"])

@router.get("/labs", response_model=List[LabReportOut])
def list_lab_reports(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    items = (
        db.query(LabReport)
        .filter(LabReport.user_id == str(current_user.id))
        .order_by(LabReport.created_at.desc())
        .all()
    )
    return items

@router.post("/labs", response_model=LabReportOut, status_code=status.HTTP_201_CREATED)
def create_lab_report(
    payload: LabReportCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    structured = payload.structured_json
    summary = payload.summary
    if not structured or not summary:
        try:
            analysis = process_text(payload.raw_text)
        except Exception:
            analysis = None
        if analysis:
            if not structured:
                structured = dict(analysis.get("structured") or {})
                structured["entities"] = analysis.get("entities", [])
                structured["language"] = analysis.get("language")
                structured["ner_meta"] = analysis.get("ner_meta")
            if not summary:
                summary = analysis.get("explanation")
    item = LabReport(
        user_id=str(current_user.id),
        title=payload.title,
        raw_text=payload.raw_text,
        structured_json=structured,
        summary=summary,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item

@router.get("/labs/{lab_id}", response_model=LabReportOut)
def get_lab_report(
    lab_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    item = (
        db.query(LabReport)
        .filter(LabReport.id == lab_id, LabReport.user_id == str(current_user.id))
        .first()
    )
    if not item:
        raise HTTPException(status_code=404, detail="Lab report not found")
    return item

@router.delete("/labs/{lab_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_lab_report(
    lab_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    item = (
        db.query(LabReport)
        .filter(LabReport.id == lab_id, LabReport.user_id == str(current_user.id))
        .first()
    )
    if not item:
        raise HTTPException(status_code=404, detail="Lab report not found")
    db.delete(item)
    db.commit()
    return None
