# --- imports (top of backend/app.py) ---
import os, io, json, base64, re
import pytesseract, httpx
import sys

from typing import Any, Dict, List, Optional, Set, Tuple
from sqlalchemy.orm import Session
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, constr
from dotenv import load_dotenv
from pathlib import Path

# Resolve paths early so env vars are available before importing the app modules
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
ENV_PATH = BASE_DIR / ".env"

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

load_dotenv(ENV_PATH, override=True)

from pypdf import PdfReader
from PIL import Image
from backend.models import init_db, symptom_event # noqa - needed to register models
from backend.db.session import SessionLocal, get_db
from backend.auth.deps import hash_password, get_current_user
from backend.models.user import User, UserProfile
from backend.models.lab_report import LabReport
from backend.models.symptom_event import SymptomEvent
from backend.routes import auth_routes, history_routes, symptoms_routes, recs_routes
from backend.services.symptom_analysis import analyze_text as analyze_symptom_text
from backend.services.symptom_events import save_symptom_event
from backend.services.recommendations import recommend, red_flag_triage, lab_triage
from backend.routers.profile import router as profile_router

# --- app & router setup ---
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY", "") or "").strip()
# Allow overriding the Gemini model via env; default to 2.5 flash to match README/services
GEMINI_MODEL = (os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()
HUGGINGFACE_TOKEN = (os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_API_KEY") or "").strip()
BIOBERT_MODEL_NAME = os.getenv("BIOBERT_MODEL_NAME", "d4data/biobert_ner").strip() or "d4data/biobert_ner"
AI_EXPLANATION_ENABLED = (os.getenv("AI_EXPLANATION_ENABLED", "true") or "true").strip().lower() not in {"0","false","off","no"}

app = FastAPI(title="MediBot Backend", version="0.1.0")
router = APIRouter(prefix="/api")

# --- logging setup ---
import logging
from datetime import datetime

from backend.middleware.tracing import TracingMiddleware

class JsonFormatter(logging.Formatter):
    def format(self, record):  # type: ignore[override]
        payload = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "level": record.levelname,
            "function": record.funcName,
            "message": record.getMessage(),
        }
        return json.dumps(payload, ensure_ascii=False)

def configure_logging() -> logging.Logger:
    logger = logging.getLogger("medibot")
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    return logger

logger = configure_logging()

# ---- Rate limiting (slowapi) ----
try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware

    limiter = Limiter(key_func=get_remote_address, default_limits=[])
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)

    # Provide a reset hook for tests
    def _reset_limiter():
        try:
            limiter.reset()
        except Exception:
            try:
                limiter.storage.clear()
            except Exception:
                pass
    setattr(app.state.limiter, "reset", _reset_limiter)
except Exception:
    class _NoopLimiter:
        def limit(self, *_args, **_kwargs):
            def deco(fn):
                return fn
            return deco
        def reset(self):
            return None
    limiter = _NoopLimiter()
    app.state.limiter = limiter
    # Provide fallback get_remote_address to keep code paths working without slowapi
    def get_remote_address(request):  # type: ignore[no-redef]
        try:
            c = getattr(request, 'client', None)
            return getattr(c, 'host', 'unknown') or 'unknown'
        except Exception:
            return 'unknown'

# ---- Rate limit helpers & handler ----
import time
from fastapi import Request as _FastAPIRequest

def user_rate_key(request: _FastAPIRequest) -> str:
    """Return a per-user key when available; otherwise fall back to IP.

    Assumes a dependency sets request.state.user_id for authenticated routes.
    """
    try:
        uid = getattr(request.state, "user_id", None)
        if uid:
            return str(uid)
    except Exception:
        pass
    return get_remote_address(request)

if 'RateLimitExceeded' in globals():
    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: _FastAPIRequest, exc: RateLimitExceeded):
        try:
            retry_after = max(1, int(getattr(exc, "reset_time", time.time()) - time.time()))
        except Exception:
            retry_after = 60
        try:
            logger.info({
                "function": "rate_limit",
                "path": str(request.url.path),
                "client": get_remote_address(request),
            })
        except Exception:
            pass
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(retry_after)},
            content={
                "detail": "Too many requests. Please wait a bit and try again.",
            },
        )


def _maybe_seed_demo_user() -> None:
    email = (os.getenv("DEMO_USER_EMAIL", "demo@example.com") or "").strip()
    password = os.getenv("DEMO_USER_PASSWORD", "demo123")
    name = (os.getenv("DEMO_USER_NAME", "Demo User") or "").strip()

    if not email or not password:
        return

    try:
        with SessionLocal() as db:
            exists = db.query(User).filter(User.email == email).first()
            if exists:
                return

            user = User(
                email=email,
                hashed_password=hash_password(password),
                name=name,
            )
            db.add(user)
            db.commit()
    except Exception:
        # Seeding is best-effort; never block startup
        logger.warning("Demo user seeding failed", exc_info=True)


@app.on_event("startup")
def _init_db():
    init_db()
    _maybe_seed_demo_user()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include existing routers
app.include_router(auth_routes.router)
app.include_router(history_routes.router)
app.include_router(profile_router)   # profile PUT/GET live here
app.include_router(symptoms_routes.router)
app.include_router(recs_routes.router, prefix="/api/recommendations")

# --- schemas ---
class ParseRequest(BaseModel):
    text: constr(max_length=5000)

class ChatIn(BaseModel):
    message: str

class ChatOut(BaseModel):
    request_id: str
    summary: Optional[str] = ""
    symptom_analysis: Dict[str, Any]
    local_recommendations: Dict[str, Any]
    ai_explanation: str
    ai_explanation_source: Optional[str] = None  # "model" | "fallback" | "skipped"
    timed_out: Optional[bool] = None
    disclaimer: str
    pipeline: Optional[Dict[str, Any]] = None
    missing_fields: Optional[List[str]] = None
    triage: Optional[Dict[str, Any]] = None

class ExplainIn(BaseModel):
    structured: Dict[str, Any]

class ExplainOut(BaseModel):
    explanation: str

# -------------------- Utils --------------------

# Robust patterns with named groups; unit is always optional but CAPTURED
TEST_PATTERNS = [
    # Hemoglobin
    ("Hemoglobin", r"(?:hemoglobin|hb)\\s*[:\\-]?\\s*(?P<value>\d+(?:\\.\\d+)?)\\s*(?P<unit>g/?d[il]|g\\s?per\\s?d[il])?"),
    # WBC
    ("WBC", r"(?:wbc|white\\s+blood\\s+cells?)\\s*[:\\-]?\\s*(?P<value>\\d{1,3}(?:,\\d{3})+|\\d+(?:\\.\\d+)?)\\s*(?P<unit>(?:/[\\u00B5\\u03BC]l|/ul|/uL|/µL|/μL|x?10\^3/?[µμ]l|x?10\^9/?l)?)"),
    # Platelets
    ("Platelets", r"(?:platelets?\b|plt)\\s*[:\\-]?\\s*(?P<value>\\d{1,3}(?:,\\d{3})+|\\d+(?:\\.\\d+)?)\\s*(?P<unit>(?:/[\\u00B5\\u03BC]l|/ul|/uL|/µL|/μL|x?10\^3/?[µμ]l|x?10\^9/?l)?)"),
    # Creatinine
    ("Creatinine", r"(?:creatinine|creat)\\s*[:\\-]?\\s*(?P<value>\\d+(?:\\.\\d+)?)\\s*(?P<unit>mg/?d[il]|µ?mol/?l|umol/?l|mg/?dL)"),
    # Glucose
    ("Glucose", r"(?:glucose|fasting\\s+glucose|\bfbs)\\s*[:\\-]?\\s*(?P<value>\\d+(?:\\.\\d+)?)\\s*(?P<unit>mg/?d[il]|mmol/?l|mg/?dL)"),
    # ALT
    ("ALT", r"(?:alt|\bsgpt)\\s*[:\\-]?\\s*(?P<value>\\d+(?:\\.\\d+)?)\\s*(?P<unit>u/?l|U/?L)"),
    # AST
    ("AST", r"(?:ast|\bsgot)\\s*[:\\-]?\\s*(?P<value>\\d+(?:\\.\\d+)?)\\s*(?P<unit>u/?l|U/?L)"),
    # HDL
    ("HDL", r"(?:hdl)\\s*[:\\-]?\\s*(?P<value>\\d+(?:\\.\\d+)?)\\s*(?P<unit>mg/?d[il]|mg/?dL)"),
    # LDL
    ("LDL", r"(?:ldl)\\s*[:\\-]?\\s*(?P<value>\\d+(?:\\.\\d+)?)\\s*(?P<unit>mg/?d[il]|mg/?dL)"),
]

STATUS_WORDS = {
    "high": [" high ", " elevated ", " above ", "↑", " high)", "(high", "high,"],
    "low": [" low ", " decreased ", " below ", "↓", " low)", "(low", "low,"],
    "normal": [" normal ", " within range ", " in range "],
}

CONDITION_WORDS = [
    "anemia", "infection", "inflammation", "diabetes", "kidney", "liver",
    "thyroid", "cholesterol", "triglycerides", "cardiac", "covid", "vitamin d",
    "iron", "b12", "urinary", "pregnancy", "hepatitis"
]

def normalize_unit_text(u: str) -> str:
    if not u:
        return ""
    # Normalize common variants/mis-encodings to a canonical form
    u = u.replace("Âµ", "µ").replace("μ", "µ")  # fix micro variants
    u = u.replace(" ", "")                      # remove stray spaces
    # common equivalents
    u = u.replace("/uL", "/µL").replace("/ul", "/µL").replace("/ULL", "/µL")
    u = u.replace("mg/dl", "mg/dL").replace("g/dl", "g/dL")
    u = u.replace("U/l", "U/L").replace("u/l", "U/L")
    # allow "per" style
    u = u.replace("gperdL", "g/dL").replace("gperdl", "g/dL")
    return u


def normalize_text(s: str) -> str:
    # normalize micro symbol variations and spacing
    s = s.replace("μ", "µ")  # greek mu -> micro sign
    return re.sub(r"\\s+", " ", s.strip().lower())

def detect_status(window: str) -> Optional[str]:
    w = " " + normalize_text(window) + " "  # pad to catch word boundaries
    for k, words in STATUS_WORDS.items():
        if any(word in w for word in words):
            return k
    return None

def detect_conditions(text: str) -> List[str]:
    low = normalize_text(text)
    found = []
    for c in CONDITION_WORDS:
        if c in low:
            found.append(c)
    return sorted(list(set(found)))

def parse_lab_text(text: str) -> Dict[str, Any]:
    # Pre-normalize noisy OCR/HTML text before any extraction
    def _normalize_ocr_text(s: str) -> str:
        try:
            # HTML entities and angle variants
            s = s.replace("&lt;=", "<=").replace("&gt;=", ">=")
            s = s.replace("&lt;", "<").replace("&gt;", ">")
            s = s.replace("&amp;", "&")
            # Fix tokens like 1270H.1 -> 1270.1 H (preserve H/L semantics explicitly)
            s = re.sub(r"(?i)(\b\d+)([HL])\.(\d+\b)", r"\1.\3 \2", s)
            # Collapse stray trailing periods on numeric tokens: 4.7. -> 4.7, 181. -> 181
            s = re.sub(r"(\b\d+(?:\.\d+)?)(\.)\b", r"\1", s)
            # Normalize whitespace
            s = re.sub(r"[\t\r]+", " ", s)
            s = re.sub(r"\u00A0", " ", s)  # non-breaking space
            # Normalize line endings and trim excessive internal spaces
            lines = []
            for ln in s.splitlines():
                lines.append(re.sub(r"\s+", " ", ln).strip())
            return "\n".join(ln for ln in lines if ln)
        except Exception:
            return s

    original_text = text
    text = _normalize_ocr_text(text)
    pipeline, model_name, init_warning = get_ner_pipeline()
    entities_raw: List[Dict[str, Any]] = []
    entities_formatted: List[Dict[str, Any]] = []
    ner_error: Optional[str] = None

    if pipeline is not None:
        try:
            entities_raw = pipeline(text)  # type: ignore[operator]
            entities_formatted = format_entities(entities_raw)
        except Exception as err:  # pragma: no cover - runtime inference issues
            ner_error = f"NER inference failed: {err}"
            entities_raw = []
    else:
        ner_error = init_warning or "NER pipeline unavailable"

    tests = extract_tests_from_entities(text, entities_raw)

    heuristics_data: Optional[Dict[str, Any]] = None

    if not tests:
        heuristics_data = _parse_lab_text_heuristic(text)
        meta = heuristics_data["meta"].copy()
        if model_name:
            meta["ner_model"] = model_name
        if init_warning:
            meta["ner_warning"] = init_warning
        if ner_error and ner_error != init_warning:
            meta["ner_error"] = ner_error
        if not meta.get("engine"):
            meta["engine"] = "heuristic-fallback"

        # Compute a quick overall confidence for labs parsed via heuristics
        try:
            confs = [t.get("confidence", 0.5) for t in heuristics_data.get("tests", []) if t.get("unit") or t.get("ref_min") or t.get("ref_max")]
            overall_conf = round(sum(confs)/max(1, len(confs)), 2)
        except Exception:
            overall_conf = 0.5

        out = {
            "tests": heuristics_data["tests"],
            "conditions": heuristics_data["conditions"],
            "entities": entities_formatted,
            "meta": {**meta, "overall_lab_confidence": overall_conf, "cleaned_text": text},
        }
        try:
            abnormal_count = sum(1 for t in heuristics_data.get("tests", []) if t.get("abnormal") in ("high", "low"))
            logger.info({
                "function": "lab_parse_summary",
                "parsed_count": len(heuristics_data.get("tests", [])),
                "abnormal_count": abnormal_count,
                "overall_lab_confidence": overall_conf,
            })
        except Exception:
            pass
        return out

    conditions = detect_conditions(text)
    engine = "huggingface-biobert"
    note = "Values derived by aligning BioBERT entities with table parser."
    if model_name and model_name != BIOBERT_MODEL_NAME:
        engine = "huggingface-ner-fallback"
        note = "BioBERT unavailable; using fallback HuggingFace NER model."

    meta: Dict[str, Any] = {
        "engine": engine,
        "ner_model": model_name,
        "note": note,
    }
    if init_warning and model_name == NER_FALLBACK_MODEL_NAME:
        meta["ner_warning"] = init_warning
    if ner_error and ner_error != init_warning:
        meta["ner_warning"] = ner_error

    heuristics_data = heuristics_data or _parse_lab_text_heuristic(text)
    seen_keys: Set[Tuple[str, str, str]] = {
        (t["name"].lower(), t["value"], t.get("unit", "")) for t in tests
    }
    supplemented = False
    for candidate in heuristics_data["tests"]:
        key = (candidate["name"].lower(), candidate["value"], candidate.get("unit", ""))
        if key in seen_keys:
            continue
        tests.append(candidate)
        seen_keys.add(key)
        supplemented = True

    if supplemented:
        meta["supplemented_with"] = "heuristic-fallback"
        # Combine any condition hints discovered during heuristic pass
        heuristic_conditions = heuristics_data["conditions"]
        combined = sorted(set(conditions) | set(heuristic_conditions))
        conditions = combined

    # Compute a quick overall confidence for labs parsed via NER+heuristics tests
    try:
        confs = [t.get("confidence", 0.5) for t in tests if t.get("unit") or t.get("ref_min") or t.get("ref_max")]
        overall_conf = round(sum(confs)/max(1, len(confs)), 2)
    except Exception:
        overall_conf = 0.5

    out = {
        "tests": tests,
        "conditions": conditions,
        "entities": entities_formatted,
        "meta": {**meta, "overall_lab_confidence": overall_conf, "cleaned_text": text},
    }
    try:
        abnormal_count = sum(1 for t in tests if t.get("abnormal") in ("high", "low"))
        logger.info({
            "function": "lab_parse_summary",
            "parsed_count": len(tests),
            "abnormal_count": abnormal_count,
            "overall_lab_confidence": overall_conf,
        })
    except Exception:
        pass
    return out
# -------------------- Routes --------------------

def get_ner_pipeline():
    # This function can be expanded to select models based on availability or other criteria
    # For now, it's a placeholder for where such logic would go.
    # In a real app, you might check for GPU availability, model versions, etc.
    return None, "placeholder_model", "NER pipeline not available"

def format_entities(entities):
    # Placeholder for formatting raw NER output
    return entities

def _line_spans(text: str) -> List[Tuple[str, int, int]]:
    spans: List[Tuple[str, int, int]] = []
    offset = 0
    for chunk in text.splitlines(True):
        line = chunk.rstrip("\r\n")
        start = offset
        offset += len(chunk)
        end = offset
        spans.append((line, start, end))
    if not spans and text:
        spans.append((text, 0, len(text)))
    return spans

def parse_line_into_test(line: str, preferred_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    raw = line.strip()
    if not raw:
        return None
    m = ROW_PATTERN.match(raw)
    if not m:
        return None

    proc_raw = m.group("proc").strip()
    valtok = m.group("valtok").strip()
    unit_tok = (m.group("unit") or "").strip()
    rest = m.group("rest") or ""

    name = proc_raw
    name_unit = None
    mnu = NAME_UNIT_IN_PARENS.match(proc_raw)
    if mnu:
        name = mnu.group("name").strip()
        name_unit = normalize_unit_text(mnu.group("unit").strip())

    # Inline abnormal flags (H/L) may be glued to values; detect before cleaning
    inline_flag = None
    if re.search(r"(?i)\d+[H](?:\.|\b)", valtok):
        inline_flag = "high"
    elif re.search(r"(?i)\d+[L](?:\.|\b)", valtok):
        inline_flag = "low"

    value_str = clean_value_token(valtok).replace(",", "")
    if not value_str:
        return None
    try:
        value_num = float(value_str)
    except ValueError:
        return None

    unit = normalize_unit_text(unit_tok) or name_unit or ""
    status = inline_flag or detect_flag(valtok) or detect_status(rest) or "unspecified"

    ref = parse_bracket_range(raw)
    if ref:
        if ref.get("unit"):
            unit = normalize_unit_text(ref["unit"]) or unit
        inferred = compare_to_range(value_num, ref)
        if inferred:
            status = inferred

    canon = preferred_name.strip() if preferred_name else name
    canon_low = canon.lower().replace(" ", "")
    if "cholesteroltotal" in canon_low:
        canon = "Cholesterol Total"
    elif canon_low.startswith("triglyceride"):
        canon = "Triglyceride"

    # Confidence heuristic: base on row match (0.7), +unit (0.15), +ref (0.15)
    conf = 0.7 + (0.15 if unit else 0.0) + (0.15 if ref else 0.0)
    conf = max(0.0, min(1.0, conf))

    # Normalize reference dict to ref_min/ref_max
    ref_min: Optional[float] = None
    ref_max: Optional[float] = None
    if ref:
        try:
            kind = ref.get("kind")
            if kind == "between":
                ref_min = float(ref.get("lo")) if ref.get("lo") is not None else None
                ref_max = float(ref.get("hi")) if ref.get("hi") is not None else None
            elif kind in ("lt", "lte"):
                ref_max = float(ref.get("v")) if ref.get("v") is not None else None
            elif kind in ("gt", "gte"):
                ref_min = float(ref.get("v")) if ref.get("v") is not None else None
        except Exception:
            ref_min = ref_min or None
            ref_max = ref_max or None

    abnormal = status if status in ("high", "low", "normal") else "unknown"

    result = {
        "name": canon,
        "value": value_str,
        "unit": unit,
        "status": status,
        "reference": ref or None,
        "ref_min": ref_min,
        "ref_max": ref_max,
        "abnormal": abnormal,
        "evidence": raw,
        "confidence": round(conf, 2),
    }
    return result

def _keyword_pattern_fallback(text: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    lowered = text
    lowered_n = normalize_text(text)
    for name, pattern in TEST_PATTERNS:
        regex = re.compile(pattern, re.IGNORECASE)
        for match in regex.finditer(lowered):
            gd = match.groupdict()
            value = (gd.get("value") or "").strip().replace(",", "")
            unit = normalize_unit_text((gd.get("unit") or "").strip())
            if not value:
                continue
            start, end = match.span()
            window = lowered_n[max(0, start - 30): min(len(lowered_n), end + 30)]
            status = detect_status(window) or "unspecified"
            # Confidence lower for keyword fallback
            conf = 0.5 + (0.2 if unit else 0.0)
            results.append({
                "name": name,
                "value": value,
                "unit": unit,
                "status": status,
                "reference": None,
                "ref_min": None,
                "ref_max": None,
                "abnormal": status if status in ("high","low","normal") else "unknown",
                "evidence": text[start:end],
                "confidence": round(min(conf, 1.0), 2),
            })
    return results

def _parse_lab_text_heuristic(text: str) -> Dict[str, Any]:
    tests: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str]] = set()
    for raw_line in text.splitlines():
        candidate = parse_line_into_test(raw_line)
        if not candidate:
            continue
        key = (candidate["name"].lower(), candidate["value"], candidate["unit"])
        if key in seen:
            continue
        seen.add(key)
        tests.append(candidate)

    for candidate in _keyword_pattern_fallback(text):
        key = (candidate["name"].lower(), candidate["value"], candidate["unit"])
        if key in seen:
            continue
        seen.add(key)
        tests.append(candidate)

    conditions = detect_conditions(text)

    # Post-process for unit/value mapping for common analytes
    def add_display_values(t: Dict[str, Any]) -> Dict[str, Any]:
        analyte = (t.get("name") or "").lower()
        unit = t.get("unit") or ""
        try:
            val = float(str(t.get("value") or "").replace(",",""))
        except Exception:
            return t
        display = {}
        # mg/dL <-> mmol/L mapping for common analytes
        CHOL_FACT = 38.67
        TRIG_FACT = 88.57
        GLUC_FACT = 18.0
        def add_pair(mgdl: float, mmol: float):
            display["mg/dL"] = round(mgdl, 2)
            display["mmol/L"] = round(mmol, 2)
        if "cholesterol" in analyte or analyte in ("ldl","hdl"):
            if unit.lower() == "mg/dl":
                add_pair(val, val/CHOL_FACT)
            elif unit.lower() in ("mmol/l","mmol/L"):
                add_pair(val*CHOL_FACT, val)
        elif "triglyceride" in analyte:
            if unit.lower() == "mg/dl":
                add_pair(val, val/TRIG_FACT)
            elif unit.lower() in ("mmol/l","mmol/L"):
                add_pair(val*TRIG_FACT, val)
        elif "glucose" in analyte:
            if unit.lower() == "mg/dl":
                add_pair(val, val/GLUC_FACT)
            elif unit.lower() in ("mmol/l","mmol/L"):
                add_pair(val*GLUC_FACT, val)
        if display:
            t["display_values"] = display
        # Ensure numeric value
        t["value"] = val
        # Ensure abnormal from ref if unknown
        if t.get("abnormal") in (None, "unknown"):
            try:
                if t.get("ref_min") is not None and val < float(t["ref_min"]):
                    t["abnormal"] = "low"
                elif t.get("ref_max") is not None and val > float(t["ref_max"]):
                    t["abnormal"] = "high"
                else:
                    if t.get("ref_min") is not None or t.get("ref_max") is not None:
                        t["abnormal"] = "normal"
            except Exception:
                pass
        return t

    tests = [add_display_values(t) for t in tests]

    # Confidence aggregate
    try:
        confs = [t.get("confidence", 0.5) for t in tests if (t.get("unit") or t.get("ref_min") is not None or t.get("ref_max") is not None)]
        overall = round(sum(confs)/max(1, len(confs)), 2)
    except Exception:
        overall = 0.5

    return {
        "tests": tests,
        "conditions": conditions,
        "meta": {
            "engine": "heuristic-fallback",
            "note": "BioBERT parsing unavailable; using keyword/row heuristics.",
            "overall_lab_confidence": overall,
        },
    }

def extract_tests_from_entities(text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # This is a placeholder for a more sophisticated entity extraction logic.
    # For now, we'll just return an empty list as the NER service is being removed.
    return []

NER_LAB_LABELS = set() # Placeholder
VALUE_UNIT_INLINE = re.compile(r'.*') # Placeholder

def normalize_entity_label(label: str) -> str:
    # Placeholder
    return label

def _select_value_match(segment: str, from_start: bool = True) -> Optional[re.Match]:
    # Placeholder
    return None

def _naive_test_from_line(line: str, name: str, entity_offset: int) -> Optional[Dict[str, Any]]:
    # Placeholder
    return None

NER_FALLBACK_MODEL_NAME = "placeholder_fallback_model"


@app.post("/api/parse_lab")
def parse_lab(req: ParseRequest):
    try:
        text = req.text if isinstance(req.text, str) else (req.text or "")
        text = str(text)
        if not text.strip():
            raise ValueError("Empty text")

        data = parse_lab_text(text)
        return data

    except Exception as e:
        # Never crash the server; return a helpful message
        # and include a short error string so we know what happened.
        logger.exception("parse_lab failed")
        return {
            "tests": [],
            "conditions": [],
            "meta": {
                "engine": "heuristic-ner-placeholder",
                "error": f"parse_lab failed: {e.__class__.__name__}: {e}",
            },
        }


@app.post("/api/analyze_symptoms")
@limiter.limit("20/minute")
async def analyze_symptoms_ep(
    request: Request,
    payload: ParseRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Extract symptom entities and suggest likely tests.

    Accepts: {"text": "I feel dizzy and weak"}
    Returns: {"symptoms": ["dizziness", "weakness"], "possible_tests": ["blood sugar (glucose)", "hemoglobin (CBC)"], "confidence": 0.xx}
    """
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    try:
        result = analyze_symptom_text(text)
        # required logging format
        try:
            logger.info({
                "function": "analyze_text",
                "symptoms": result.get("symptoms", []),
                "tests": result.get("possible_tests", []),
                "confidence": result.get("confidence", 0.0),
            })
        except Exception:
            pass

        event = SymptomEvent(
            user_id=str(user.id),
            raw_text=text,
            result_json=result,
        )
        db.add(event)
        db.commit()
        return result
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.exception("/api/analyze_symptoms failed")
        raise HTTPException(status_code=500, detail=f"analysis failed: {e}")

@app.get("/api/list_models")
@limiter.limit("5/minute")
async def list_models(request: Request):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=400, detail="No GEMINI_API_KEY set")
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error("list_models failed")
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text) from e

@app.post("/api/chat", response_model=ChatOut)
@limiter.limit("30/minute", key_func=user_rate_key)
@limiter.limit("100/minute", key_func=get_remote_address)
async def chat(
    request: Request,
    payload: ChatIn,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ChatOut:
    # Record user id on request state for per-user rate limiting
    try:
        request.state.user_id = str(user.id)
    except Exception:
        pass
    from backend.utils.app import build_chat_context
    from backend.services import symptoms as symp

    # Generate a request_id for this chat call
    import uuid as _uuid
    request_id = str(_uuid.uuid4())
    try:
        logger.info({"function": "start_chat", "request_id": request_id})
    except Exception:
        pass

    # Run deterministic symptom parsing first
    try:
        pipeline_json = symp.summarize_to_json(payload.message).model_dump()
    except Exception as _e:
        pipeline_json = {"symptoms": [], "overall_confidence": 0.0}

    # Run symptom analysis ONCE; do not persist here
    try:
        symptom_analysis = analyze_symptom_text(payload.message)
        try:
            logger.info({
                "function": "analyze_text",
                "request_id": request_id,
                "symptom_count": len(symptom_analysis.get("symptoms", [])),
                "confidence": symptom_analysis.get("confidence", 0.0),
            })
        except Exception:
            pass
    except Exception:
        symptom_analysis = {"symptoms": [], "possible_tests": [], "confidence": 0.0}

    # Persist (idempotent within 10s) and attach event_id; do not duplicate across retries
    ev_id: Optional[str] = None
    try:
        ev = save_symptom_event(db, str(user.id), payload.message, symptom_analysis)
        ev_id = str(ev.id)
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        try:
            logger.exception("save_symptom_event failed in /api/chat", extra={"request_id": request_id})
        except Exception:
            pass
    if ev_id:
        try:
            symptom_analysis["event_id"] = ev_id
        except Exception:
            pass
        try:
            logger.info({
                "function": "chat_symptom_analysis_saved",
                "request_id": request_id,
                "event_id": ev_id,
                "symptoms": symptom_analysis.get("symptoms", []),
                "tests": symptom_analysis.get("possible_tests", []),
                "confidence": symptom_analysis.get("confidence", 0.0),
            })
        except Exception:
            pass

    # Build local recommendations block using profile + latest lab + reported symptoms
    local_recs: Dict[str, Any] = {}
    latest_lab_struct = None
    profile_dict = None
    try:
        prof = db.query(UserProfile).filter(UserProfile.user_id == str(user.id)).first()
        if prof is not None:
            try:
                profile_dict = {
                    "age": getattr(prof, "age", None),
                    "sex": getattr(prof, "sex", None),
                    "conditions": list(getattr(prof, "conditions", []) or []),
                    "medications": list(getattr(prof, "medications", []) or []),
                }
            except Exception:
                profile_dict = None

        lab = (
            db.query(LabReport)
            .filter(LabReport.user_id == str(user.id))
            .order_by(LabReport.created_at.desc())
            .first()
        )
        latest_lab_struct = lab.structured_json if (lab and lab.structured_json) else None

        # Compute triage from red flags and labs
        rf = red_flag_triage(symptom_analysis.get("symptoms", []), payload.message)
        lt = lab_triage(latest_lab_struct)
        # Normalize triage to low/moderate/high scale
        # If red flags urgent -> treat as 'high'
        level_map = {"ok": "low", "watch": "moderate", "urgent": "high"}
        rf_level = level_map.get(rf.get("level"), "low") if isinstance(rf, dict) else "low"
        # Choose the higher of lab and red-flag levels
        order = ["low", "moderate", "high"]
        def max_level(a: str, b: str) -> str:
            return a if order.index(a) >= order.index(b) else b
        triage_level = max_level(lt.get("level", "low"), rf_level)
        triage_reasons = []
        if isinstance(lt, dict):
            triage_reasons += lt.get("reasons", []) or []
        if isinstance(rf, dict) and rf.get("reasons"):
            triage_reasons += [f"red flag: {r}" for r in rf.get("reasons")]
        triage_window = lt.get("suggested_window") if triage_level != "low" else "routine follow-up"
        triage = {"level": triage_level, "reasons": triage_reasons, "suggested_window": triage_window}

        # Build recommendations and then sync priority with triage
        local_recs = recommend(profile_dict, latest_lab_struct, symptom_analysis.get("symptoms", []), raw_text=payload.message)
        try:
            # Map triage level to priority
            tri_to_pri = {"high": "high", "moderate": "moderate", "low": "low"}
            mapped = tri_to_pri.get(triage_level, local_recs.get("priority", "low"))
            local_recs["priority"] = mapped
            if triage_level == "high":
                # Ensure urgent prompts present
                urgent_action = "Seek medical attention promptly"
                if urgent_action not in (local_recs.get("actions") or []):
                    local_recs["actions"] = [urgent_action] + (local_recs.get("actions") or [])
                # Add explicit clinician follow-up
                follow = local_recs.get("follow_up") or ""
                if "urgent care" not in follow.lower() and "emergency" not in follow.lower():
                    local_recs["follow_up"] = "Seek urgent care or emergency services now."
                # Ensure explicit clinician follow-up action appears
                add_action = "Arrange clinician follow-up to review abnormal labs"
                if add_action not in (local_recs.get("actions") or []):
                    local_recs["actions"].append(add_action)
        except Exception:
            pass
        try:
            logger.info({
                "function": "recommend",
                "request_id": request_id,
                "priority": local_recs.get("priority"),
                "actions_count": len(local_recs.get("actions", []) or []),
                "triage_level": triage.get("level") if isinstance(triage, dict) else None,
            })
        except Exception:
            pass
        # compact triage log
        try:
            logger.info({
                "function": "triage",
                "request_id": request_id,
                "level": triage.get("level"),
                "reason_count": len(triage.get("reasons", [])),
            })
        except Exception:
            pass
        try:
            triage_log = {
                "function": "triage",
                "request_id": request_id,
                "level": triage.get("level") if isinstance(triage, dict) else None,
                "reasons": triage.get("reasons") if isinstance(triage, dict) else None,
            }
            if isinstance(triage, dict) and triage.get("level") == "urgent":
                triage_log.update({
                    "priority": local_recs.get("priority"),
                    "first_action": (local_recs.get("actions") or [None])[0],
                })
            logger.info(triage_log)
        except Exception:
            pass
    except Exception:
        local_recs = {}
        triage = {"level": "ok", "reasons": []}

    # Build AI prompt regardless; we may choose to skip the call
    context, notice = build_chat_context(db, user, payload.message)
    prompt = (
        "You are a helpful medical assistant. Use the provided context to answer the user's question. "
        "The context includes the user's profile, their latest lab report, and their latest symptom summary. "
        "If context is missing, inform the user. Do not provide medical advice. "
        "Keep responses concise and easy to understand.\n\n"
        f"TRIAGE: {(triage or {}).get('level', 'low')} | Reasons: {', '.join((triage or {}).get('reasons', []) or [])}\n"
        f"{context}\n\n--- Symptom Parser JSON (low confidence may be noisy) ---\n"
        f"{json.dumps(pipeline_json, ensure_ascii=False)}\n\n"
        "--- Extracted Symptom Entities ---\n"
        f"{', '.join(symptom_analysis.get('symptoms') or []) or 'none'}\n"
        "--- Suggested Tests From Symptoms ---\n"
        f"{', '.join(symptom_analysis.get('possible_tests') or []) or 'none'}\n"
    )

    # Always attempt AI explanation unless explicitly disabled by config or missing key
    ai_explanation = ""
    ai_explanation_source = "skipped"
    timed_out = False

    if not AI_EXPLANATION_ENABLED:
        ai_explanation = "AI explanation disabled by config."
        ai_explanation_source = "skipped"
    elif not GEMINI_API_KEY:
        ai_explanation = "AI explanation skipped: model not configured."
        ai_explanation_source = "skipped"
    else:
        # Proceed to call the model with strict timeout
        try:
            logger.info({"function": "ai_explanation", "request_id": request_id, "stage": "ai_start"})
        except Exception:
            pass
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
                r = await client.post(
                    url,
                    params={"key": GEMINI_API_KEY},
                    headers={"Content-Type": "application/json"},
                    json={"contents": [{"role": "user", "parts": [{"text": prompt.strip()}]}]},
                )
                r.raise_for_status()
                data = r.json()
                ai_explanation = (
                    data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                    or ""
                ).strip()
                ai_explanation_source = "model"
                timed_out = False
        except httpx.TimeoutException:
            timed_out = True
            ai_explanation_source = "fallback"
            ai_explanation = "AI explanation unavailable; showing structured recommendations."
        except Exception:
            timed_out = False
            ai_explanation_source = "fallback"
            ai_explanation = "AI explanation unavailable; showing structured recommendations."
        finally:
            try:
                logger.info({
                    "function": "ai_explanation",
                    "request_id": request_id,
                    "stage": "ai_done",
                    "ai_explanation_source": ai_explanation_source,
                    "timed_out": timed_out,
                    "chars": len(ai_explanation or ""),
                })
            except Exception:
                pass

    # Compute missing context fields for guided completion banner
    missing_fields: List[str] = []
    try:
        if not profile_dict or profile_dict.get("age") in (None, ""):
            missing_fields.append("age")
        if not profile_dict or not (profile_dict.get("sex") or "").strip():
            missing_fields.append("sex")
        if not profile_dict or not (profile_dict.get("conditions") or []):
            missing_fields.append("conditions")
        if not profile_dict or not (profile_dict.get("medications") or []):
            missing_fields.append("medications")
        if latest_lab_struct is None:
            missing_fields.append("latest_lab")
    except Exception:
        # Non-fatal
        pass

    # Compose summary and disclaimer
    try:
        n_sym = len(symptom_analysis.get("symptoms", []) or [])
        prio = (local_recs or {}).get("priority", "low") if isinstance(local_recs, dict) else "low"
        summary = f"Detected {n_sym} symptom(s). Priority: {prio}."
    except Exception:
        summary = ""
    disclaimer = "For educational purposes only. Consult a medical professional for medical advice."

    # Ensure event_id key exists even when not persisted
    symptom_analysis.setdefault("event_id", None)

    # Final integrity log before return: keys present in payload
    try:
        top_keys = [
            "request_id",
            "summary",
            "symptom_analysis",
            "local_recommendations",
            "ai_explanation",
            "ai_explanation_source",
            "timed_out",
            "disclaimer",
            "pipeline",
            "triage",
        ]
        logger.info({"function": "end_chat", "request_id": request_id, "keys": top_keys, "missing_fields": missing_fields, "triage": triage})
    except Exception:
        pass

    return ChatOut(
        request_id=request_id,
        summary=summary,
        symptom_analysis=symptom_analysis,
        local_recommendations=local_recs or {"priority": "low", "actions": ["Monitor symptoms and rest"], "follow_up": "If symptoms persist >48h, worsen, or include red flags (fainting, chest pain), seek medical care.", "rationale": "Initial self-care suggestions based on reported symptoms."},
        ai_explanation=ai_explanation,
        ai_explanation_source=ai_explanation_source,
        timed_out=timed_out,
        disclaimer=disclaimer,
        pipeline={"symptom_parse": pipeline_json},
        missing_fields=missing_fields,
        triage=triage,
    )



@router.post("/explain", response_model=ExplainOut)
@limiter.limit("2/minute")
async def explain(request: Request, payload: ExplainIn) -> ExplainOut:
    structured = payload.structured

    # fallback if no key
    if not GEMINI_API_KEY:
        lines = []
        for t in (structured.get("tests") or [])[:10]:
            name = t.get("name", "Test")
            val  = t.get("value", "")
            unit = t.get("unit", "")
            status = t.get("status", "unspecified")
            lines.append(f"- {name}: {val} {unit} — {status}")
        txt = ("Here's a brief summary of your labs.\n\n" +
               "\n".join(lines) +
               "\n\nThis is educational only; please consult a clinician.")
        return ExplainOut(explanation=txt)

    # call Gemini
    prompt = (
        "You are a clinician assistant. Given structured lab data, produce a concise, "
        "patient-friendly explanation (<=200 words). Not medical advice.\n\n" +
        f"{json.dumps(structured, ensure_ascii=False)}"
    )
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
            )
            r = await client.post(
                url,
                params={"key": GEMINI_API_KEY},
                headers={"Content-Type": "application/json"},
                json={"contents": [{"role": "user", "parts": [{"text": prompt.strip()}]}]},
            )
            r.raise_for_status()
            data = r.json()
            text = (
                data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
            ) or "I couldn’t generate a response."
            return ExplainOut(explanation=text.strip())
    except Exception as e:
        logger.exception("explain failed")
        return ExplainOut(explanation=f"⚠️ Gemini error: {e}")

# Register the local /api/explain route after it has been attached to the router
app.include_router(router)

def _sanitize_filename(name: str) -> str:
    name = os.path.basename(name or "file")
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return name or "file"


MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "10"))


@app.post("/api/extract_text")
@limiter.limit("10/minute", key_func=user_rate_key)
async def extract_text(request: Request, file: UploadFile = File(...), user: User = Depends(get_current_user)):
    # Record user id on request state for per-user rate limiting
    try:
        request.state.user_id = str(user.id)
    except Exception:
        pass
    # Read into memory; never write to disk
    data = await file.read()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds the {MAX_FILE_MB}MB limit",
        )

    mt = (file.content_type or "").lower()
    name = _sanitize_filename(file.filename or "file")

    # Enforce allowed MIME types strictly
    if not (mt == "application/pdf" or mt.startswith("image/")):
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {mt or 'unknown'}")

    # PDF path
    if mt == "application/pdf" or name.lower().endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(data))
            text = [page.extract_text() or "" for page in reader.pages]
            joined = "\n".join(text).strip()
            if not joined:
                raise ValueError("No text found in PDF")
            return {"text": joined}
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PDF extraction failed: {e}")

    # Image OCR path
    try:
        img = Image.open(io.BytesIO(data))
        txt = pytesseract.image_to_string(img, lang="eng")
        if not txt.strip():
            raise ValueError("OCR found no text")
        return {"text": txt}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OCR failed: {e}")

# -------------------- Table/line parser helpers (new) --------------------

RANGE_BRACKET = re.compile(r"[\[](?P<body>[^\\\]]+)[\]]")  # [<=5.2], [10-120], [200-239mg/dL]
NUM = r"\\d+(?:\\.\\d+)?"

def clean_value_token(tok: str) -> str:
    """Strip trailing punctuation and glued flags like '1270H.1' -> '1270' and return numeric string."""
    t = tok.strip()
    # split glued H/L flags next to number (e.g., 1270H.1 or 1270H)
    t = re.sub(r"(?i)\b(" + NUM + r")[HhLl](?:\\."+")?", r"\1", t)  # tolerate an extra dot after H/L
    # remove trailing dots and commas
    t = re.sub(r"[.,]+", "", t)
    # keep only leading numeric
    m = re.match(r"^\\s*(" + NUM + ")", t)
    return m.group(1) if m else ""

def detect_flag(token_or_line: str) -> Optional[str]:
    s = token_or_line.lower()
    if re.search(r"\bhigh\b|\(high\)|\bH\b", s): return "high"
    if re.search(r"\blow\b|\(low\)|\bL\b", s): return "low"
    return None


def parse_bracket_range(line: str) -> Optional[Dict[str, Any]]:
    """
    Canonicalize bracket ranges to one of:
      {"kind":"lte","v":N,"unit":"..."}
      {"kind":"lt","v":N,"unit":"..."}
      {"kind":"gte","v":N,"unit":"..."}
      {"kind":"gt","v":N,"unit":"..."}
      {"kind":"between","lo":A,"hi":B,"unit":"..."}
    """
    m = RANGE_BRACKET.search(line)
    if not m:
        return None
    body = m.group("body").replace(" ", "")
    # <=, <, >=, >
    m2 = re.match(r"^(<=|<|>=|>)(" + NUM + r")([a-zA-Zµμ%/\\^-]+)?$", body)
    if m2:
        op, val, unit_tail = m2.groups()
        unit = normalize_unit_text(unit_tail or "")
        v = float(val)
        op_map = {"<=": "lte", "<": "lt", ">=": "gte", ">": "gt"}
        return {"kind": op_map[op], "v": v, "unit": unit}
    # a-b (optional unit)
    m3 = re.match(r"^(" + NUM + r")-(" + NUM + r")([a-zA-Zµμ%/\\^-]+)?$", body)
    if m3:
        lo, hi, unit_tail = m3.groups()
        unit = normalize_unit_text(unit_tail or "")
        return {"kind": "between", "lo": float(lo), "hi": float(hi), "unit": unit}
    return None

def compare_to_range(value: float, rng: Dict[str, Any]) -> Optional[str]:
    """Return 'high'/'low'/'normal' from value vs canonical range."""
    if not rng:
        return None
    k = rng.get("kind")
    if k == "lte":
        return "high" if value > rng["v"] else "normal"
    if k == "lt":
        return "high" if value >= rng["v"] else "normal"
    if k == "gte":
        return "low"  if value < rng["v"] else "normal"
    if k == "gt":
        return "low"  if value <= rng["v"] else "normal"
    if k == "between":
        lo, hi = rng["lo"], rng["hi"]
        if value < lo: return "low"
        if value > hi: return "high"
        return "normal"
    return None

NAME_UNIT_IN_PARENS = re.compile(r"^(?P<name>.+?)\((?P<unit>[^)]+)\)\s*$", re.IGNORECASE)

ROW_PATTERN = re.compile(
    r"""^
    (?P<proc>[A-Za-z][A-Za-z0-9\s/_\-]+\((?:[^)]+\))?)"""   # Procedure name (might include (unit))
    """\s+"""
    """(?P<valtok>[-+]?""" + NUM + r"(?:[HhLl](?:\.)?)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:[HhLl](?:\.)?)?)"""  # value token with optional H/L
    """\s*\.?"""                                               # optional trailing dot after value
    """\s+"""
    """(?P<unit>[^\[\]\s]+(?:/[^\[\]\s]+)?)?"""                # unit token (e.g., mg/dL)
    """(?P<rest>.*)$"""
    , re.VERBOSE
)
