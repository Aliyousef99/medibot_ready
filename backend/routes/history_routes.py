# backend/routes/history_routes.py
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from sqlalchemy.orm import Session

from backend.db.session import get_db
from backend.models.user import User
from backend.models.lab_report import LabReport
from backend.schemas.profile import LabReportCreate, LabReportOut
from backend.services.report_pipeline import process_text, process_upload
from backend.auth.deps import get_current_user
from fastapi import File, UploadFile

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

@router.post("/labs/upload", response_model=LabReportOut, status_code=status.HTTP_201_CREATED)
async def upload_lab_report(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    meta = {
        "filename": file.filename or "upload",
        "content_type": (file.content_type or "").lower(),
        "size_bytes": len(data),
    }
    try:
        result = process_upload(data, file.filename or "upload", file.content_type or "")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Upload failed: {exc}")

    structured = result.get("structured") or {}
    structured["file_meta"] = {
        **meta,
        "stored_path": result.get("stored_path"),
        "stored_name": result.get("stored_name"),
        "source": result.get("source"),
    }
    summary = result.get("explanation") or "Lab summary unavailable."
    raw_text = result.get("raw_text") or result.get("text") or ""

    item = LabReport(
        user_id=str(current_user.id),
        title=file.filename or "Lab Report",
        raw_text=raw_text,
        structured_json=structured,
        summary=summary,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item
