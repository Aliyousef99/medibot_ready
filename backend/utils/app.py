from typing import Any, Optional, Tuple
import os
import json
from sqlalchemy.orm import Session
from sqlalchemy import cast, Text as SAText
from backend.models.lab_report import LabReport
from backend.models.user import User, UserProfile
from backend.models.symptom_event import SymptomEvent


# ---------------- Chat Context Helpers ----------------
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (ValueError, TypeError):
        return default


def estimate_tokens(text: str) -> int:
    # Simple heuristic: ~4 chars per token for English
    if not text:
        return 0
    return max(1, len(text) // 4)


def _profile_lines(profile: Optional[UserProfile]) -> str:
    lines = [
        f"Age: {getattr(profile, 'age', None) if profile else 'N/A'}",
        f"Sex: {getattr(profile, 'sex', None) if profile else 'N/A'}",
        f"Pre-existing Conditions: {', '.join(profile.conditions) if (profile and getattr(profile, 'conditions', None)) else 'None listed'}",
        f"Medications: {', '.join(profile.medications) if (profile and getattr(profile, 'medications', None)) else 'None listed'}",
    ]
    return "\n".join(lines)

def build_chat_context(db: Session, user: User, latest_message: str, structured_override: Optional[dict] = None) -> Tuple[str, Optional[str]]:
    """Build a safe context string for LLM calls with truncation.

    Returns (context_text, notice_if_truncated)
    """
    # Ensure clean transaction before queries
    try:
        db.rollback()
    except Exception:
        pass
    profile = db.query(UserProfile).filter(cast(UserProfile.user_id, SAText) == str(user.id)).first()
    profile_str = _profile_lines(profile) 
    # Latest lab report (structured) — allow override from request context
    if structured_override is not None:
        lab_struct = structured_override
    else:
        lab = (
            db.query(LabReport)
            .filter(cast(LabReport.user_id, SAText) == str(user.id))
            .order_by(LabReport.created_at.desc())
            .first() 
        )
        lab_struct = lab.structured_json if (lab and lab.structured_json) else None
    lab_str = json.dumps(lab_struct, ensure_ascii=False, indent=2) if lab_struct else "None provided"

    # Latest symptom event (optional)
    sym = (
        db.query(SymptomEvent)
        .filter(cast(SymptomEvent.user_id, SAText) == str(user.id))
        .order_by(SymptomEvent.created_at.desc())
        .first()
    )
    symptom_str = "None"
    if sym and sym.result_json:
        try:
            # brief symptom overview
            syms = sym.result_json.get("symptoms", [])
            names = ", ".join(s.get("text") for s in syms[:5] if s.get("text"))
            symptom_str = f"{names or 'Detected 0 symptoms'} (urgency: {sym.result_json.get('urgency', 'unknown')})"
        except Exception:
            symptom_str = "(unavailable)"

    context = (
        "--- User Profile ---\n"
        f"{profile_str}\n\n"
        "--- Latest Lab Report (structured JSON) ---\n"
        f"{lab_str}\n\n"
        "--- Latest Symptom Summary ---\n"
        f"{symptom_str}\n\n"
        "--- User's Question ---\n"
        f"{latest_message}\n"
    )

    return context, None
