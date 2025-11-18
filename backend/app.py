# --- imports (top of backend/app.py) ---
import os, io, json, base64, re
import pytesseract, httpx
import sys

from typing import Any, Dict, List, Optional, Set, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, cast, Text as SAText
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, APIRouter, Request, Response
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
from backend.routes import chat_routes
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
app.add_middleware(TracingMiddleware)
try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware

    limiter = Limiter(key_func=get_remote_address, default_limits=[])
    app.state.limiter = limiter

    app.add_middleware(SlowAPIMiddleware)
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

# ---- Simple latest-lab cache (TTL 15 min) ----
if not hasattr(app.state, "latest_lab_cache"):
    app.state.latest_lab_cache = {}  # type: ignore[attr-defined]

LAB_CACHE_TTL_SECONDS = 15 * 60

def set_latest_lab_cache(user_id: str, structured: Dict[str, Any]) -> None:
    try:
        app.state.latest_lab_cache[user_id] = {  # type: ignore[attr-defined]
            "ts": time.time(),
            "data": structured,
        }
    except Exception:
        pass

def get_latest_lab_cache(user_id: str) -> Optional[Dict[str, Any]]:
    try:
        entry = app.state.latest_lab_cache.get(user_id)  # type: ignore[attr-defined]
        if not entry:
            return None
        if time.time() - float(entry.get("ts", 0)) > LAB_CACHE_TTL_SECONDS:
            # expired
            try:
                del app.state.latest_lab_cache[user_id]  # type: ignore[attr-defined]
            except Exception:
                pass
            return None
        data = entry.get("data")
        return data if isinstance(data, dict) else None
    except Exception:
        return None

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
app.include_router(chat_routes.router)
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
    user_view: Optional[Dict[str, Any]] = None

class ExplainIn(BaseModel):
    structured: Dict[str, Any]

class ExplainOut(BaseModel):
    explanation: str
    lab_report_id: Optional[str] = None

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

    # Drop legend/threshold headings like 'Desirable', 'Normal', 'Borderline High', 'High:>' etc.
    try:
        low_name = (canon or "").strip().lower()
        if re.match(r"^(desirable|normal|borderline\s*high?|very\s*high|high:?|low:?)(\b|:|\s)", low_name):
            return None
        slug = re.sub(r"[^a-z0-9]+", "", low_name)
        if slug in {"desirable", "normal", "borderline", "borderlinehigh", "veryhigh", "high", "low", "reference", "ref"}:
            return None
    except Exception:
        pass

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
def parse_lab(req: ParseRequest, request: Request, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    try:
        text = req.text if isinstance(req.text, str) else (req.text or "")
        text = str(text)
        if not text.strip():
            raise ValueError("Empty text")

        data = parse_lab_text(text)

        # Persist structured lab for current user (UPSERT-lite) and cache
        try:
            tests = data.get("tests") or []
            analyte_names = [t.get("name") for t in tests if t.get("name")][:3]
            abnormal_count = sum(1 for t in tests if (t.get("abnormal") or t.get("status")) in ("high", "low"))
            title_suffix = ", ".join([n for n in analyte_names if n]) or "Lab Report"
            ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
            title = f"{title_suffix} — {ts}"
            summary = f"Parsed {len(tests)} analytes; {abnormal_count} abnormal."
            # simple dedupe: if last lab for user has identical structured_json, reuse it
            lr = (
                db.query(LabReport)
                .filter(LabReport.user_id == str(user.id))
                .order_by(LabReport.created_at.desc())
                .first()
            )
            if not lr or (lr.structured_json or {}) != data:
                lr = LabReport(
                    user_id=str(user.id),
                    title=title,
                    raw_text=text,
                    structured_json=data,
                    summary=summary,
                )
                db.add(lr)
                db.commit()
            # cache the structured lab for quick reuse in /api/chat
            set_latest_lab_cache(str(user.id), data)
            try:
                logger.info({
                    "function": "lab_persist",
                    "user_id": str(user.id),
                    "lab_report_id": str(lr.id),
                    "analyte_count": len(tests),
                    "abnormal_count": abnormal_count,
                })
            except Exception:
                pass
            data["lab_id"] = str(lr.id)  # backward-compatible key
            data["lab_report_id"] = str(lr.id)
        except Exception:
            try:
                db.rollback()
            except Exception:
                pass
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

# ---- Simple per-user chat history (in-memory) ----
if not hasattr(app.state, "chat_history_store"):
    app.state.chat_history_store = {}  # type: ignore[attr-defined]

MAX_CHAT_HISTORY = 50  # keep last 50 turns (user/assistant messages)

def append_chat_history(user_id: str, role: str, content: str) -> None:
    try:
        store = app.state.chat_history_store  # type: ignore[attr-defined]
        bucket = store.setdefault(user_id, [])
        bucket.append({"role": role, "content": content})
        # trim oldest beyond limit
        if len(bucket) > MAX_CHAT_HISTORY:
            del bucket[: len(bucket) - MAX_CHAT_HISTORY]
    except Exception:
        pass

def get_chat_history(user_id: str):
    try:
        store = app.state.chat_history_store  # type: ignore[attr-defined]
        hist = store.get(user_id) or []
        return hist if isinstance(hist, list) else []
    except Exception:
        return []


@app.post("/api/chat", response_model=ChatOut)
# Relax rate limits to support continuous chat sessions
@limiter.limit("120/minute", key_func=user_rate_key)
@limiter.limit("300/minute", key_func=get_remote_address)
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

    # Append the incoming user message to per-user chat history
    try:
        append_chat_history(str(user.id), "user", payload.message or "")
    except Exception:
        pass

    # Inline extractor: run early for pasted lab-like messages; do NOT write to DB
    user_view: Optional[Dict[str, Any]] = None
    try:
        raw_msg = (payload.message or "")
        # Minimal pre-clean per spec
        cleaned = (
            raw_msg
            .replace("&lt;=", "<=")
            .replace("&gt;=", ">=")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
        )
        # 1270H.1 -> 1270.1 H (record flag separately)
        cleaned = re.sub(r"(?i)(\b\d+)([HL])\.(\d+\b)", r"\1.\3 \2", cleaned)
        # Strip trailing dots after numbers: 4.7. -> 4.7
        cleaned = re.sub(r"(\b\d+(?:\.\d+)?)(\.)\b", r"\1", cleaned)
        # Normalize analyte names
        cleaned = re.sub(r"\bCholesterolTotal\b", "Total Cholesterol", cleaned)
        cleaned = re.sub(r"\bTriglyceride\b", "Triglycerides", cleaned)

        # Trigger condition: long text AND contains unit token OR bracketed range
        long_enough = len(cleaned) >= 200
        has_unit = bool(re.search(r"(?i)\b(ng/mL|mmol/L|mg/dL)\b", cleaned))
        has_brackets = ("[" in cleaned and "]" in cleaned)
        if long_enough and (has_unit or has_brackets):
            # Prefer dedicated inline parser if available
            inline_result: Optional[Dict[str, Any]] = None
            try:
                if '_inline_parse' in globals() and callable(globals().get('_inline_parse')):
                    inline_fn = globals().get('_inline_parse')  # type: ignore
                    from datetime import datetime as _dt
                    now_iso = _dt.utcnow().isoformat() + "Z"
                    inline_result = inline_fn(cleaned, received_at=now_iso)  # type: ignore[misc]
                else:
                    inline_result = None
            except Exception:
                inline_result = None

            presentation: Dict[str, Any] = {}
            structured_json: Dict[str, Any] = {}
            if isinstance(inline_result, dict) and inline_result.get("presentation"):
                presentation = inline_result.get("presentation") or {}
                structured_json = inline_result.get("structured_json") or {}
            else:
                # Fallback: use built-in heuristic parser
                maybe_lab = parse_lab_text(cleaned)  # returns {tests:[...]}
                tests_now = (maybe_lab or {}).get("tests") if isinstance(maybe_lab, dict) else []
                abnormal_items = []
                normal_items = []
                for t in (tests_now or []):
                    try:
                        flag = (t.get("abnormal") or t.get("status") or "").lower()
                        item = {
                            "name": t.get("name"),
                            "value": t.get("value"),
                            "unit": t.get("unit"),
                            "status": flag or "unspecified",
                        }
                        if flag in ("high", "low"):
                            abnormal_items.append(item)
                        else:
                            normal_items.append(item)
                    except Exception:
                        continue
                # simple confidence heuristic
                parsed = len(tests_now or [])
                try:
                    line_count = max(1, len((cleaned or '').splitlines()))
                except Exception:
                    line_count = 1
                confidence = round(min(0.95, max(0.2, (parsed / line_count) + 0.2)), 2) if tests_now else 0.4
                presentation = {"abnormal": abnormal_items, "normal": normal_items, "confidence": confidence}
                structured_json = {"analytes": tests_now or []}

            parsed_n = len((structured_json.get("analytes") or [])) if isinstance(structured_json, dict) else 0
            abnormal_n = len((presentation.get("abnormal") or [])) if isinstance(presentation, dict) else 0
            conf = float(presentation.get("confidence") or 0.0) if isinstance(presentation, dict) else 0.0

            # Build user_view for UI consumption when we have parsed lines
            if parsed_n >= 1:
                summary_txt = f"Parsed {parsed_n} line(s); {abnormal_n} abnormal."
                # Append received date if present in the raw text as "ReceivedDate/Time dd/mm/yyyy"
                try:
                    m = re.search(r"(?i)received\s*date\s*/\s*time[^\n\r]*?(\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b)", cleaned)
                    if m:
                        summary_txt += f" ({m.group(1)})"
                except Exception:
                    pass
                # Build a concise, non-vague explanation with clear next steps
                try:
                    causes_map = {
                        'ferritin': 'iron overload, significant inflammation, or liver disease',
                        'ldl': 'increased cardiovascular risk',
                        'hdl': 'reduced protective cholesterol',
                        'triglycerides': 'metabolic factors such as insulin resistance or diet',
                        'glucose': 'elevated blood sugar or diabetes risk',
                        'hemoglobin': 'anemia (if low) or dehydration (if high)',
                        'tsh': 'thyroid dysfunction',
                        'crp': 'inflammation',
                    }
                    def _name_key(n: str) -> str:
                        return (n or '').strip().lower().replace(' ', '')
                    ab_names = []
                    critical = False
                    critical_names = []
                    for it in (presentation.get('abnormal') or []):
                        nm = str(it.get('name') or '').strip()
                        st = (it.get('status') or '').lower()
                        if not nm:
                            continue
                        # Detect critical by value vs ref_max when available
                        try:
                            v = float(str(it.get('value') or '').replace(',', ''))
                        except Exception:
                            v = None
                        ref_max = it.get('ref_max')
                        ref_min = it.get('ref_min')
                        if ref_max is None and isinstance(it.get('reference'), dict) and it['reference'].get('kind') in ('lt','lte'):
                            ref_max = it['reference'].get('v')
                        try:
                            if v is not None:
                                if st == 'high' and ref_max is not None and float(ref_max) > 0:
                                    ratio = v / float(ref_max)
                                    if ratio >= 2.0:
                                        critical = True
                                        critical_names.append(nm)
                                if st == 'low' and ref_min is not None and float(ref_min) > 0:
                                    ratio_low = v / float(ref_min)
                                    if ratio_low <= 0.5:
                                        critical = True
                                        critical_names.append(nm)
                        except Exception:
                            pass
                        # Collect name + status label
                        ab_names.append(f"{nm} ({'elevated' if st=='high' else 'low' if st=='low' else st or 'abnormal'})")
                    ab_part = ", ".join(ab_names[:3]) if ab_names else ''
                    # Choose analyte-specific cause if available
                    cause_hint = None
                    try:
                        for it in (presentation.get('abnormal') or []):
                            nm = str(it.get('name') or '')
                            k = _name_key(nm)
                            # normalize keys for common analytes
                            if 'ferritin' in k:
                                cause_hint = causes_map['ferritin']; break
                            if k in ('ldl','hdl') or 'cholesterol' in k:
                                cause_hint = causes_map['ldl']; break
                            if 'triglycer' in k:
                                cause_hint = causes_map['triglycerides']; break
                            if 'glucose' in k:
                                cause_hint = causes_map['glucose']; break
                            if 'hemoglobin' in k or k=='hb':
                                cause_hint = causes_map['hemoglobin']; break
                            if 'tsh' in k:
                                cause_hint = causes_map['tsh']; break
                            if 'crp' in k:
                                cause_hint = causes_map['crp']; break
                    except Exception:
                        pass
                    if abnormal_n == 0:
                        explanation_txt = 'Your results are within normal limits based on the ranges shown.'
                    else:
                        base = f"The following result(s) are abnormal: {ab_part}. " if ab_part else "Some results are abnormal. "
                        if critical:
                            more = "This is significantly outside the reference range"
                            if cause_hint:
                                more += f" and may indicate {cause_hint}"
                            explanation_txt = base + more + ". Based on your available information, please seek immediate medical attention."
                        else:
                            more = "This may indicate an underlying issue"
                            if cause_hint:
                                more += f" such as {cause_hint}"
                            explanation_txt = base + more + ". Please discuss these results with your clinician."
                except Exception:
                    explanation_txt = ''
                if abnormal_n > 0:
                    recommendation_txt = "Some results are outside reference. Please consider discussing with your clinician, especially if you have symptoms."
                else:
                    recommendation_txt = "Most results appear within range. Maintain healthy habits and follow your clinician's guidance."
                user_view = {
                    "summary": summary_txt,
                    "abnormal": presentation.get("abnormal") or [],
                    "normal": presentation.get("normal") or [],
                    "recommendation": recommendation_txt,
                    "confidence": round(conf, 2),
                    "explanation": explanation_txt,
                }

            # Attach to request state for downstream context but DO NOT persist
            try:
                if structured_json:
                    request.state.structured_lab = structured_json
            except Exception:
                pass

            # Single info log as specified
            try:
                logger.info({
                    "function": "inline_lab_extract",
                    "parsed": parsed_n,
                    "abnormal": abnormal_n,
                    "confidence": round(conf, 2),
                })
            except Exception:
                pass
    except Exception:
        # Inline extraction is best-effort; never block chat
        user_view = None

    # Resolve latest structured lab: current -> cache -> db -> none
    # Helpers: safe JSON load and validation for structured lab objects
    def _deserialize_structured(obj: Any) -> Optional[Dict[str, Any]]:
        try:
            if obj is None:
                return None
            # Skip empty string or empty object strings
            if isinstance(obj, str):
                s = obj.strip()
                if not s or s in ("{}", "null", "NULL", "None"):
                    return None
                try:
                    loaded = json.loads(s)
                except json.JSONDecodeError:
                    return None
                return loaded if isinstance(loaded, dict) and loaded else None
            if isinstance(obj, dict):
                return obj if obj else None
            return None
        except Exception:
            return None

    def _extract_analytes_and_abnormal(structured: Optional[Dict[str, Any]]) -> Tuple[List[str], int]:
        names: List[str] = []
        abnormal = 0
        if not isinstance(structured, dict):
            return names, abnormal
        # Primary expected shape: { tests: [{name, abnormal/status, ...}, ...] }
        tests = structured.get("tests")
        if isinstance(tests, list):
            for t in tests:
                if not isinstance(t, dict):
                    continue
                nm = t.get("name")
                if isinstance(nm, str) and nm and nm not in names:
                    names.append(nm)
                flag = (t.get("abnormal") or t.get("status") or "")
                if isinstance(flag, str) and flag.lower() in ("high", "low"):
                    abnormal += 1
        # Alternate shape: { labs: { analytes: [{name, ...}, ...] } }
        labs = structured.get("labs") if isinstance(structured.get("labs"), dict) else None
        analytes = labs.get("analytes") if isinstance(labs, dict) else None
        if isinstance(analytes, list):
            for a in analytes:
                if not isinstance(a, dict):
                    continue
                nm = a.get("name") or a.get("analyte")
                if isinstance(nm, str) and nm and nm not in names:
                    names.append(nm)
                flag = (a.get("abnormal") or a.get("status") or a.get("flag") or "")
                if isinstance(flag, str) and flag.lower() in ("high", "low"):
                    abnormal += 1
        # Fallback: top-level keys that look like analytes (e.g., Ferritin)
        # Only consider a small whitelist to avoid noise
        KNOWN_KEYS = {
            "Ferritin", "LDL", "HDL", "Glucose", "Hemoglobin", "WBC", "Platelets", "Creatinine",
            "AST", "ALT", "TSH", "Triglycerides", "CRP",
        }
        for k, v in structured.items():
            if k in KNOWN_KEYS and k not in names:
                names.append(k)
        return names, abnormal

    def _is_valid_lab_obj(structured: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(structured, dict) or not structured:
            return False
        # Valid if tests list has one or more items
        tests = structured.get("tests")
        if isinstance(tests, list) and len(tests) > 0:
            return True
        # Or if labs.analytes exists and non-empty
        labs = structured.get("labs") if isinstance(structured.get("labs"), dict) else None
        analytes = labs.get("analytes") if isinstance(labs, dict) else None
        if isinstance(analytes, list) and len(analytes) > 0:
            return True
        # Or if any known analyte keys (like Ferritin) are present
        names, _ = _extract_analytes_and_abnormal(structured)
        return len(names) > 0

    latest_lab_struct_early: Optional[Dict[str, Any]] = None
    lab_source = "none"
    _lab_context_logged = False
    try:
        # a) current (middleware may attach to request.state)
        current = getattr(request.state, "structured_lab", None)
        cur_obj = _deserialize_structured(current) if not isinstance(current, dict) else current
        if _is_valid_lab_obj(cur_obj):
            latest_lab_struct_early = cur_obj
            lab_source = "current"
        else:
            # b) cache (per-user)
            cached = get_latest_lab_cache(str(user.id))
            if _is_valid_lab_obj(cached):
                latest_lab_struct_early = cached
                lab_source = "cache"
            else:
                # c) DB (most recent rows; do not assume non-null
                # Fetch newest 10 with non-empty structured_json (raw DB filter)
                trimmed = func.trim(cast(LabReport.structured_json, SAText))
                candidates = (
                    db.query(LabReport)
                    .filter(
                        LabReport.user_id == str(user.id),
                        LabReport.structured_json.isnot(None),
                        func.length(trimmed) > 2,
                        trimmed.notin_(['{}', 'null', ''])
                    )
                    .order_by(LabReport.created_at.desc())
                    .limit(10)
                    .all()
                )
                # summary before filtering
                try:
                    logger.info({
                        "function": "lab_context_scan",
                        "request_id": request_id,
                        "fetched": len(candidates),
                    })
                except Exception:
                    pass
                chosen: Optional[Dict[str, Any]] = None
                for lr in candidates:
                    sj_raw = getattr(lr, "structured_json", None)
                    # classify raw empties first
                    if sj_raw is None:
                        try:
                            logger.info({
                                "function": "lab_context_skip",
                                "request_id": request_id,
                                "reason": "empty_or_null",
                                "lab_report_id": str(lr.id),
                            })
                        except Exception:
                            pass
                        continue
                    sj: Optional[Dict[str, Any]] = None
                    if isinstance(sj_raw, str):
                        s = sj_raw.strip()
                        if not s or s in ("{}", "null", "NULL", "None"):
                            try:
                                logger.info({
                                    "function": "lab_context_skip",
                                    "request_id": request_id,
                                    "reason": "empty_or_null",
                                    "lab_report_id": str(lr.id),
                                })
                            except Exception:
                                pass
                            continue
                        try:
                            loaded = json.loads(s)
                            sj = loaded if isinstance(loaded, dict) and loaded else None
                        except json.JSONDecodeError:
                            try:
                                logger.info({
                                    "function": "lab_context_skip",
                                    "request_id": request_id,
                                    "reason": "json_decode_error",
                                    "lab_report_id": str(lr.id),
                                })
                            except Exception:
                                pass
                            continue
                        if not sj:
                            try:
                                logger.info({
                                    "function": "lab_context_skip",
                                    "request_id": request_id,
                                    "reason": "empty_or_null",
                                    "lab_report_id": str(lr.id),
                                })
                            except Exception:
                                pass
                            continue
                    elif isinstance(sj_raw, dict):
                        if not sj_raw:
                            try:
                                logger.info({
                                    "function": "lab_context_skip",
                                    "request_id": request_id,
                                    "reason": "empty_or_null",
                                    "lab_report_id": str(lr.id),
                                })
                            except Exception:
                                pass
                            continue
                        sj = sj_raw
                    else:
                        try:
                            logger.info({
                                "function": "lab_context_skip",
                                "request_id": request_id,
                                "reason": "empty_or_null",
                                "lab_report_id": str(lr.id),
                            })
                        except Exception:
                            pass
                        continue

                    # validate analytes presence
                    tests = sj.get("tests") if isinstance(sj, dict) else None
                    labs_obj = sj.get("labs") if isinstance(sj, dict) and isinstance(sj.get("labs"), dict) else None
                    analytes_list = labs_obj.get("analytes") if isinstance(labs_obj, dict) else None

                    if isinstance(tests, list):
                        if len(tests) == 0:
                            try:
                                logger.info({
                                    "function": "lab_context_skip",
                                    "request_id": request_id,
                                    "reason": "analytes_empty",
                                    "lab_report_id": str(lr.id),
                                })
                            except Exception:
                                pass
                            continue
                        else:
                            chosen = sj
                    elif isinstance(analytes_list, list):
                        if len(analytes_list) == 0:
                            try:
                                logger.info({
                                    "function": "lab_context_skip",
                                    "request_id": request_id,
                                    "reason": "analytes_empty",
                                    "lab_report_id": str(lr.id),
                                })
                            except Exception:
                                pass
                            continue
                        else:
                            chosen = sj
                    else:
                        names_tmp, _ = _extract_analytes_and_abnormal(sj)
                        if len(names_tmp) == 0:
                            try:
                                logger.info({
                                    "function": "lab_context_skip",
                                    "request_id": request_id,
                                    "reason": "no_analytes",
                                    "lab_report_id": str(lr.id),
                                    "keys": list(sj.keys()) if isinstance(sj, dict) else [],
                                })
                            except Exception:
                                pass
                            continue
                        chosen = sj

                    if chosen is not None:
                        latest_lab_struct_early = chosen
                        lab_source = "db"
                        try:
                            names_ok, abn_ok = _extract_analytes_and_abnormal(chosen)
                            logger.info({
                                "function": "lab_context",
                                "request_id": request_id,
                                "source": lab_source,
                                "has_structured_lab": True,
                                "analyte_names": names_ok,
                                "abnormal_count": abn_ok,
                            })
                            _lab_context_logged = True
                        except Exception:
                            pass
                        break
                latest_lab_struct_early = chosen
                lab_source = "db" if chosen is not None else "none"
        # store on request for downstream consumers
        try:
            request.state.resolved_lab = latest_lab_struct_early
        except Exception:
            pass
    except Exception:
        latest_lab_struct_early = None
        lab_source = "none"
    # Single lab_context success log per request when found via current/cache (DB branch logs on success)
    try:
        if not _lab_context_logged and isinstance(latest_lab_struct_early, dict) and lab_source in ("current", "cache"):
            analyte_names, abnormal_count = _extract_analytes_and_abnormal(latest_lab_struct_early)
            logger.info({
                "function": "lab_context",
                "request_id": request_id,
                "source": lab_source,
                "has_structured_lab": True,
                "analyte_names": analyte_names,
                "abnormal_count": abnormal_count,
            })
            _lab_context_logged = True
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
    # Reuse the early-resolved latest lab, if any
    latest_lab_struct = latest_lab_struct_early
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

        # Compute triage from red flags and labs
        rf = red_flag_triage(symptom_analysis.get("symptoms", []), payload.message)
        if latest_lab_struct is None:
            lt = {"level": "low", "reasons": [], "suggested_window": "routine follow-up"}
        else:
            lt = lab_triage(latest_lab_struct)
        # Normalize red-flag triage to low/moderate/high
        level_map = {"ok": "low", "watch": "moderate", "urgent": "high"}
        rf_level = level_map.get((rf or {}).get("level"), "low")
        lab_level = (lt or {}).get("level", "low")
        # Choose the higher of lab and red-flag levels
        order = ["low", "moderate", "high"]
        def max_level(a: str, b: str) -> str:
            return a if order.index(a) >= order.index(b) else b
        triage_level = max_level(lab_level, rf_level)
        # Merge reasons and de-duplicate while preserving order
        triage_reasons_raw = []
        triage_reasons_raw += (lt.get("reasons", []) if isinstance(lt, dict) else [])
        triage_reasons_raw += ([f"red flag: {r}" for r in (rf.get("reasons") or [])] if isinstance(rf, dict) else [])
        seen_r: set = set()
        triage_reasons: list = []
        for r in triage_reasons_raw:
            if not r:
                continue
            if r not in seen_r:
                seen_r.add(r)
                triage_reasons.append(r)
        # Suggested window by final level (moderate: routine follow-up; high: as soon as practical)
        triage_window = "as soon as practical" if triage_level == "high" else "routine follow-up"
        triage = {"level": triage_level, "reasons": triage_reasons, "suggested_window": triage_window}

        # Build recommendations and then sync priority with triage
        local_recs = recommend(profile_dict, latest_lab_struct, symptom_analysis.get("symptoms", []), raw_text=payload.message)
        try:
            # Map triage level to at least the same priority; do not downgrade if symptom rules already raised it
            order = {"low": 0, "moderate": 1, "high": 2}
            current = (local_recs.get("priority") or "low").lower()
            if order.get(triage_level, 0) > order.get(current, 0):
                local_recs["priority"] = triage_level
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
        # Single triage log per request with both sources and final
        try:
            logger.info({
                "function": "triage",
                "request_id": request_id,
                "symptom_level": rf_level,
                "lab_level": lab_level,
                "final_level": triage_level,
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
    # Build conversation history text (last ~10 messages)
    try:
        hist = get_chat_history(str(user.id))
        recent = hist[-10:]
        hist_lines = []
        for item in recent:
            role = (item or {}).get("role") or "user"
            content = (item or {}).get("content") or ""
            if not isinstance(content, str):
                continue
            lbl = "User" if role == "user" else "Assistant"
            # keep each line reasonably short in the system prompt
            content_trim = content.strip()
            hist_lines.append(f"{lbl}: {content_trim}")
        history_block = "\n".join(hist_lines)
    except Exception:
        history_block = ""

    context, notice = build_chat_context(db, user, payload.message, structured_override=latest_lab_struct_early)
    prompt = (
        "You are a helpful medical assistant. Use the provided context to answer the user's question. "
        "The context includes the user's profile, their latest lab report, and their latest symptom summary. "
        "Provide general, educational information and you may answer follow-up questions about likely causes of abnormal tests (e.g., high ferritin) and typical next steps. "
        "Avoid diagnosis or prescriptions; include caveats and when to seek care. "
        "Keep responses concise and easy to understand.\n\n"
        f"TRIAGE: {(triage or {}).get('level', 'low')} | Reasons: {', '.join((triage or {}).get('reasons', []) or [])}\n"
        "\n--- Conversation History (most recent first) ---\n"
        f"{history_block}\n\n"
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
            async with httpx.AsyncClient(timeout=30) as client:
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

    # Append assistant reply to history (best-effort)
    try:
        append_chat_history(str(user.id), "assistant", ai_explanation or "")
    except Exception:
        pass

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
        user_view=user_view,
    )



@router.post("/explain", response_model=ExplainOut)
@limiter.limit("2/minute")
async def explain(request: Request, payload: ExplainIn, db: Session = Depends(get_db), user: User = Depends(get_current_user), response: Response = None) -> ExplainOut:
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
        # Persist a LabReport with structured and summary; update cache
        try:
            tests = structured.get("tests") or []
            abnormal_count = sum(1 for t in tests if (t.get("abnormal") or t.get("status")) in ("high","low"))
            names = [t.get("name") for t in tests if t.get("name")][:3]
            ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
            title = f"{', '.join([n for n in names if n]) or 'Lab Report'} — {ts}"
            lr = LabReport(
                user_id=str(user.id),
                title=title,
                raw_text=json.dumps(structured, ensure_ascii=False),
                structured_json=structured,
                summary=txt,
            )
            db.add(lr)
            db.commit()
            set_latest_lab_cache(str(user.id), structured)
            if response is not None:
                try:
                    response.headers["X-Lab-Id"] = str(lr.id)
                except Exception:
                    pass
            try:
                logger.info({
                    "function": "lab_persist",
                    "user_id": str(user.id),
                    "lab_report_id": str(lr.id),
                    "analyte_count": len(tests),
                    "abnormal_count": abnormal_count,
                })
            except Exception:
                pass
        except Exception:
            try:
                db.rollback()
            except Exception:
                pass
        return ExplainOut(explanation=txt, lab_report_id=str(lr.id))

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
            try:
                final = text.strip()
                # Persist LabReport for structured; update cache
                tests = structured.get("tests") or []
                abnormal_count = sum(1 for t in tests if (t.get("abnormal") or t.get("status")) in ("high","low"))
                names = [t.get("name") for t in tests if t.get("name")][:3]
                ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M")
                title = f"{', '.join([n for n in names if n]) or 'Lab Report'} — {ts}"
                lr = LabReport(
                    user_id=str(user.id),
                    title=title,
                    raw_text=json.dumps(structured, ensure_ascii=False),
                    structured_json=structured,
                    summary=final,
                )
                db.add(lr)
                db.commit()
                set_latest_lab_cache(str(user.id), structured)
                if response is not None:
                    try:
                        response.headers["X-Lab-Id"] = str(lr.id)
                    except Exception:
                        pass
                try:
                    logger.info({
                        "function": "lab_persist",
                        "user_id": str(user.id),
                        "lab_report_id": str(lr.id),
                        "analyte_count": len(tests),
                        "abnormal_count": abnormal_count,
                    })
                except Exception:
                    pass
            except Exception:
                try:
                    db.rollback()
                except Exception:
                    pass
            return ExplainOut(explanation=final, lab_report_id=str(lr.id))
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

# -------------------- Lightweight Inline Extractor (no DB, no ML) --------------------
try:
    from lab_parser import parse_lab_text as _inline_parse
except Exception:
    try:
        from backend.services.lab_parser import parse_lab_text as _inline_parse  # type: ignore
    except Exception:
        _inline_parse = None  # type: ignore

class InlineParseIn(BaseModel):
    text: constr(max_length=10000)

class InlineParseOut(BaseModel):
    presentation: Dict[str, Any]
    structured_json: Dict[str, Any]

@app.post("/api/labs/inline_extract", response_model=InlineParseOut)
@limiter.limit("60/minute", key_func=user_rate_key)
async def inline_extract(request: Request, payload: InlineParseIn, user: User = Depends(get_current_user)):
    # Record user id on request state for per-user rate limiting
    try:
        request.state.user_id = str(user.id)
    except Exception:
        pass
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if _inline_parse is None:
        # Try dynamic import fallback if initial import failed
        try:
            from lab_parser import parse_lab_text as _dyn_inline
            globals()["_inline_parse"] = _dyn_inline  # cache for subsequent calls
        except Exception:
            try:
                from backend.services.lab_parser import parse_lab_text as _dyn_inline  # type: ignore
                globals()["_inline_parse"] = _dyn_inline
            except Exception:
                raise HTTPException(status_code=500, detail="inline extractor unavailable")
    from datetime import datetime as _dt
    now_iso = _dt.utcnow().isoformat() + "Z"
    try:
        # _inline_parse may have been initialized at import time or via dynamic fallback above
        parser_fn = globals().get("_inline_parse")
        if not callable(parser_fn):
            raise RuntimeError("inline extractor unavailable")
        result = parser_fn(text, received_at=now_iso)  # type: ignore[misc]
    except Exception as e:
        logger.exception("inline_extract failed")
        raise HTTPException(status_code=500, detail=f"parse failed: {e}")
    pres = result.get("presentation") or {"abnormal": [], "normal": [], "confidence": 0.2}
    try:
        logger.info({
            "function": "lab_parse_summary",
            "parsed_count": len((result.get("structured_json") or {}).get("analytes", [])),
            "abnormal_count": len(pres.get("abnormal", [])),
            "overall_lab_confidence": pres.get("confidence", 0.2),
        })
    except Exception:
        pass
    return InlineParseOut(presentation=pres, structured_json=result.get("structured_json") or {"analytes": [], "received_at": now_iso})

# -------------------- Table/line parser helpers (new) --------------------

RANGE_BRACKET = re.compile(r"[\[](?P<body>[^\\\]]+)[\]]")  # [<=5.2], [10-120], [200-239mg/dL]
NUM = r"\\d+(?:\\.\\d+)?"

def clean_value_token(tok: str) -> str:
    """Strip trailing punctuation and glued flags like '1270H.1' -> '1270' and return numeric string."""
    t = tok.strip()
    # split glued H/L flags next to number (e.g., 1270H.1 or 1270H)
    t = re.sub(r"(?i)\b(" + NUM + r")[HhLl](?:\.)?", r"\1", t)
    # remove thousands separators, but keep decimal points
    t = t.replace(",", "")
    # strip only trailing periods after a numeric token (avoid removing decimal points)
    t = re.sub(r"(?<=\d)\.(?=\s|$)", "", t)
    # keep only leading numeric (with optional decimal)
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
    # normalize unicode dashes inside the range/threshold body
    body = body.replace("\u2013", "-").replace("\u2014", "-")
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
