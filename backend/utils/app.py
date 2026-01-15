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
    # Latest lab report (structured) - allow override from request context
    if structured_override is not None:
        lab_struct = structured_override
        lab_source = "structured_override"
    else:
        lab = (
            db.query(LabReport)
            .filter(cast(LabReport.user_id, SAText) == str(user.id))
            .order_by(LabReport.created_at.desc())
            .first() 
        )
        lab_struct = lab.structured_json if (lab and lab.structured_json) else None
        lab_source = "structured" if lab_struct else "none"
    def _fmt_val(v: Any) -> str:
        if v is None:
            return "?"
        try:
            num = float(str(v).replace(",", ""))
        except Exception:
            return str(v)
        if abs(num - round(num)) < 1e-6:
            return str(int(round(num)))
        return f"{num:.2f}".rstrip("0").rstrip(".")

    key_tests = []
    if lab_struct:
        try:
            tests = lab_struct.get("tests") if isinstance(lab_struct, dict) else None
            if isinstance(tests, list):
                for t in tests[:12]:
                    if not isinstance(t, dict):
                        continue
                    name = t.get("name") or "Test"
                    val = _fmt_val(t.get("value"))
                    unit = t.get("unit") or ""
                    ref_min = t.get("ref_min")
                    ref_max = t.get("ref_max")
                    parts = [f"{name}: {val}{(' ' + unit) if unit else ''}"]
                    if ref_min is not None or ref_max is not None:
                        if ref_min is not None and ref_max is not None:
                            parts.append(f"ref {ref_min}-{ref_max}")
                        elif ref_min is not None:
                            parts.append(f"ref >= {ref_min}")
                        else:
                            parts.append(f"ref <= {ref_max}")
                    if t.get("reference_source") == "universal" or t.get("reference_note"):
                        parts.append("ref source: general (not lab)")
                    key_tests.append("; ".join(str(p) for p in parts if p))
        except Exception:
            key_tests = []
        lab_str = json.dumps(lab_struct, ensure_ascii=False, indent=2)
    else:
        lab_str = "None provided"
        if structured_override is None:
            if lab and getattr(lab, "raw_text", None):
                lab_str = f"Raw lab text:\n{str(lab.raw_text)}"
                lab_source = "raw_text"
            elif lab and getattr(lab, "summary", None):
                lab_str = f"Lab summary text:\n{str(lab.summary)}"
                lab_source = "summary"

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

    lab_available = "yes" if lab_str != "None provided" else "no"
    key_tests_block = "\n".join(f"- {line}" for line in key_tests) if key_tests else "None"
    context = (
        "--- User Profile ---\n"
        f"{profile_str}\n\n"
        f"--- Lab Report Availability ---\n"
        f"Available: {lab_available}\n"
        f"Source: {lab_source}\n\n"
        "--- Key Tests (use these for follow-up answers) ---\n"
        f"{key_tests_block}\n\n"
        "--- Latest Lab Report (structured JSON) ---\n"
        f"{lab_str}\n\n"
        "--- Latest Symptom Summary ---\n"
        f"{symptom_str}\n\n"
        "--- User's Question ---\n"
        f"{latest_message}\n"
    )

    return context, None
