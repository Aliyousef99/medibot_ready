# --- imports (top of backend/app.py) ---
import os, io, json, base64, re
import uuid as _uuid
import pytesseract, httpx
import sys
import shutil

from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple
from redis import Redis
from sqlalchemy.orm import Session
from sqlalchemy import func, cast, Text as SAText, text as SAtext
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, APIRouter, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, constr
from dotenv import load_dotenv
from pathlib import Path

# Resolve paths early so env vars are available before importing the app modules
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
ENV_PATH = BASE_DIR / ".env"
ROOT_ENV_PATH = ROOT_DIR / ".env"

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

if ROOT_ENV_PATH.exists():
    load_dotenv(ROOT_ENV_PATH, override=True)
load_dotenv(ENV_PATH, override=True)

from pypdf import PdfReader
from PIL import Image
from backend.models import init_db, symptom_event # noqa - needed to register models
from backend.db.session import SessionLocal, get_db
from backend.auth.deps import hash_password, get_current_user
from backend.models.user import User, UserProfile
from backend.models.lab_report import LabReport
from backend.models.symptom_event import SymptomEvent
from backend.models.conversation import Conversation
from backend.models.conversation_context import ConversationContext
from backend.models.message import Message, MessageRole
from backend.routes import auth_routes, history_routes, symptoms_routes, recs_routes, chat_routes, privacy_routes
from backend.services.symptom_analysis import analyze_text as analyze_symptom_text
from backend.services.symptom_events import save_symptom_event
from backend.services.recommendations import recommend, red_flag_triage, lab_triage
from backend.services.reference_ranges import apply_universal_reference_ranges
from backend.services.ocr import extract_text_from_bytes
from backend.routers.profile import router as profile_router
from backend.utils.app import build_chat_context

# --- app & router setup ---
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY", "") or "").strip()
# Allow overriding the Gemini model via env; default to 2.5 flash to match README/services
GEMINI_MODEL = (os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()
HUGGINGFACE_TOKEN = (os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_API_KEY") or "").strip()
BIOBERT_MODEL_NAME = os.getenv("BIOBERT_MODEL_NAME", "d4data/biobert_ner").strip() or "d4data/biobert_ner"
AI_EXPLANATION_ENABLED = (os.getenv("AI_EXPLANATION_ENABLED", "true") or "true").strip().lower() not in {"0","false","off","no"}
REDIS_URL = (os.getenv("REDIS_URL") or "redis://localhost:6379/0").strip()
REDIS_PREFIX = (os.getenv("REDIS_PREFIX") or "medibot").strip()
NER_MODEL_NAME = (os.getenv("NER_MODEL_NAME") or BIOBERT_MODEL_NAME).strip() or BIOBERT_MODEL_NAME
NER_DEVICE = (os.getenv("NER_DEVICE") or "auto").strip().lower()  # "auto" | "cpu" | "cuda" | gpu index
NER_MAX_CHARS = int(os.getenv("NER_MAX_CHARS") or 4000)
NER_TIMEOUT_SECONDS = float(os.getenv("NER_TIMEOUT_SECONDS") or 8.0)
CORS_ORIGINS_RAW = os.getenv("CORS_ORIGINS")
CORS_ORIGIN_REGEX = (os.getenv("CORS_ORIGIN_REGEX") or "https://.*\\.vercel\\.app").strip() or None
if CORS_ORIGINS_RAW:
    CORS_ALLOW_ORIGINS = [o.strip() for o in CORS_ORIGINS_RAW.split(",") if o.strip()]
else:
    CORS_ALLOW_ORIGINS = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

OFFLINE_SUMMARY_TEMPLATE = """\
**Offline Mode:**
We are unable to reach the AI service. Here is your summary based on the extracted data:
- **Test:** {test_name}
- **Result:** {value} {unit}
- **Status:** {status} (Reference Range: {range})
*Please consult a healthcare professional for verification.*
"""

app = FastAPI(title="MediBot Backend", version="0.1.0")
router = APIRouter(prefix="/api")

# --- logging setup ---
import logging
from datetime import datetime

from backend.middleware.tracing import TracingMiddleware
from backend.utils.exceptions import handle_http_exception, handle_unhandled_exception

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS if "*" not in CORS_ALLOW_ORIGINS else ["*"],
    allow_origin_regex=CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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

# Standardized error envelopes with trace_id
from fastapi import Request as _Request, HTTPException as _HTTPException
app.add_exception_handler(_HTTPException, handle_http_exception)
app.add_exception_handler(Exception, handle_unhandled_exception)

# ---- Rate limit helpers & handler ----
import time
from fastapi import Request as _FastAPIRequest

# ---- Redis-backed cache (TTL 15 min) ----
LAB_CACHE_TTL_SECONDS = 15 * 60

def set_latest_lab_cache(user_id: str, structured: Dict[str, Any]) -> None:
    client = get_redis_client()
    if not client:
        return
    try:
        client.setex(f"{REDIS_PREFIX}:latest_lab:{user_id}", LAB_CACHE_TTL_SECONDS, json.dumps(structured))
    except Exception as exc:
        logger.debug("set_latest_lab_cache failed", extra={"error": str(exc)})

def get_latest_lab_cache(user_id: str) -> Optional[Dict[str, Any]]:
    client = get_redis_client()
    if not client:
        return None
    try:
        raw = client.get(f"{REDIS_PREFIX}:latest_lab:{user_id}")
        if not raw:
            return None
        return json.loads(raw)
    except Exception as exc:
        logger.debug("get_latest_lab_cache failed", extra={"error": str(exc)})
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

def _init_redis():
    try:
        app.state.redis_client = Redis.from_url(REDIS_URL, decode_responses=True)  # type: ignore[attr-defined]
        app.state.redis_client.ping()  # type: ignore[attr-defined]
    except Exception as exc:
        app.state.redis_client = None  # type: ignore[attr-defined]
        logger.warning("Redis unavailable; falling back to no-cache mode", extra={"error": str(exc)})

def get_redis_client() -> Optional[Redis]:
    try:
        return getattr(app.state, "redis_client", None)
    except Exception:
        return None


def _verify_tesseract_available() -> None:
    """Fail fast if Tesseract is missing or unusable."""
    binary = shutil.which("tesseract")
    if not binary:
        raise RuntimeError(
            "Tesseract OCR binary not found. Install it (e.g., apt-get install tesseract-ocr) "
            "or run via the provided Dockerfile where it is preinstalled."
        )
    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        raise RuntimeError(f"Tesseract found at {binary} but unusable: {exc}")


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
            user.email_verified = True
            db.add(user)
            db.commit()
    except Exception:
        # Seeding is best-effort; never block startup
        logger.warning("Demo user seeding failed", exc_info=True)


@app.on_event("startup")
def _init_db():
    _verify_tesseract_available()
    try:
        init_db()
    except Exception as exc:  # pragma: no cover - environment specific
        logger.error("Database initialization failed. Check DATABASE_URL (current: %s)", os.getenv("DATABASE_URL", "unset"))
        raise
    _init_redis()
    _maybe_seed_demo_user()

# include existing routers
app.include_router(auth_routes.router)
app.include_router(history_routes.router)
app.include_router(profile_router)   # profile PUT/GET live here
app.include_router(symptoms_routes.router)
app.include_router(recs_routes.router, prefix="/api/recommendations")
# Legacy chat routes (start/send/history) for compatibility with existing tests/clients
app.include_router(chat_routes.router)
app.include_router(privacy_routes.router)

# --- health check ---
@app.get("/health")
def health_check():
    db_ok = False
    redis_ok = False
    model_ok = False
    model_warning: Optional[str] = None
    try:
        with SessionLocal() as db:
            db.execute(SAtext("SELECT 1"))
            db_ok = True
    except Exception as exc:
        logger.warning("healthcheck_db_failed", extra={"error": str(exc)})

    client = get_redis_client()
    if client:
        try:
            client.ping()
            redis_ok = True
        except Exception as exc:
            logger.warning("healthcheck_redis_failed", extra={"error": str(exc)})

    pipe, model_name, warn = get_ner_pipeline()
    model_ok = pipe is not None
    model_warning = warn

    status = "ok" if db_ok and redis_ok and model_ok else "degraded"
    return {
        "status": status,
        "db": db_ok,
        "redis": redis_ok,
        "ner": {
            "available": model_ok,
            "model": model_name,
            "warning": model_warning,
        },
    }

# --- schemas ---
class ParseRequest(BaseModel):
    text: constr(max_length=5000)


class ChatIn(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatOut(BaseModel):
    conversation_id: Optional[str] = None
    conversation_title: Optional[str] = None
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

class ConversationOut(BaseModel):
    id: str
    title: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class ConversationContextOut(BaseModel):
    conversation_id: str
    system_prompt: Optional[str] = None
    lab_tests: Optional[List[Dict[str, Any]]] = None

class ConversationContextIn(BaseModel):
    system_prompt: Optional[str] = None

class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    created_at: datetime

class ConversationCreate(BaseModel):
    title: Optional[str] = None

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
    from backend.services.lab_parser import parse_lab_report

    return parse_lab_report(text)
# -------------------- Routes --------------------

NER_FALLBACK_MODEL_NAME = (os.getenv("NER_FALLBACK_MODEL") or "dslim/bert-base-NER").strip() or "dslim/bert-base-NER"

def _resolve_device() -> int:
    if NER_DEVICE == "cpu":
        return -1
    if NER_DEVICE == "cuda":
        try:
            import torch  # type: ignore
            return 0 if torch.cuda.is_available() else -1
        except Exception:
            return -1
    try:
        # explicit GPU index
        if NER_DEVICE.isdigit():
            return int(NER_DEVICE)
    except Exception:
        pass
    # auto: prefer GPU if available
    try:
        import torch  # type: ignore
        return 0 if torch.cuda.is_available() else -1
    except Exception:
        return -1

@lru_cache(maxsize=1)
def _load_ner_pipeline():
    """Lazy-load BioBERT with a lightweight fallback model."""
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline as hf_pipeline  # type: ignore
    except Exception as exc:  # pragma: no cover - import errors depend on env
        return None, BIOBERT_MODEL_NAME, f"transformers unavailable: {exc}"

    auth_token = HUGGINGFACE_TOKEN or None

    def _build(model_name: str):
        return hf_pipeline(
            "ner",
            model=AutoModelForTokenClassification.from_pretrained(model_name, token=auth_token),
            tokenizer=AutoTokenizer.from_pretrained(model_name, token=auth_token),
            aggregation_strategy="simple",
            device=_resolve_device(),
        )

    init_warning: Optional[str] = None
    try:
        pipe = _build(NER_MODEL_NAME)
        return pipe, NER_MODEL_NAME, None
    except Exception as exc:  # pragma: no cover - depends on external HF availability
        init_warning = f"BioBERT unavailable: {exc}"
        try:
            pipe = _build(NER_FALLBACK_MODEL_NAME)
            return pipe, NER_FALLBACK_MODEL_NAME, init_warning
        except Exception as exc_fallback:
            warning = f"{init_warning}; fallback failed: {exc_fallback}"
            return None, NER_FALLBACK_MODEL_NAME, warning

def get_ner_pipeline():
    # Return (pipeline, model_name, warning_message)
    return _load_ner_pipeline()

def format_entities(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    for item in predictions or []:
        term = item.get("word") or item.get("text") or ""
        if not term:
            continue
        formatted.append(
            {
                "term": term,
                "start": item.get("start"),
                "end": item.get("end"),
                "score": float(item.get("score", 0.0)),
                "label": item.get("entity_group") or item.get("entity") or "ENTITY",
            }
        )
    return formatted

def _run_ner_with_timeout(pipe, input_text: str, timeout: float) -> List[Dict[str, Any]]:
    """Run pipeline with a hard timeout to avoid blocking workers."""
    try:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            fut = executor.submit(pipe, input_text)
            return fut.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        raise TimeoutError(f"NER inference exceeded {timeout} seconds")
    except Exception:
        raise

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
    if not entities:
        return []

    tests: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str]] = set()
    lines = _line_spans(text)

    for ent in entities:
        term = (ent.get("term") or ent.get("word") or ent.get("text") or "").strip()
        if not term:
            continue
        label = normalize_entity_label(ent.get("label") or ent.get("entity_group") or ent.get("entity") or "")
        if label and label not in NER_LAB_LABELS:
            continue
        start = ent.get("start")
        if start is None:
            continue
        try:
            start_idx = int(start)
        except Exception:
            continue

        line_ctx: Optional[Tuple[str, int, int]] = None
        for line, s, e in lines:
            if s <= start_idx < e:
                line_ctx = (line, s, e)
                break
        if not line_ctx:
            continue

        line_text, s, _e = line_ctx
        candidate = parse_line_into_test(line_text, preferred_name=term)
        if not candidate:
            candidate = _naive_test_from_line(line_text, term, start_idx - s)
        if not candidate:
            continue

        try:
            candidate["value"] = float(str(candidate.get("value")))
        except Exception:
            pass

        key = (candidate["name"].lower(), str(candidate["value"]), candidate.get("unit", ""))
        if key in seen:
            continue
        seen.add(key)
        candidate["source"] = candidate.get("source") or "ner-aligned"
        tests.append(candidate)

    return tests

NER_LAB_LABELS = {
    "CHEMICAL",
    "CHEM",
    "TEST",
    "LAB",
    "LAB_TEST",
    "DISEASE",
    "PROBLEM",
    "SYMPTOM",
    "SIGN",
    "TREATMENT",
    "MEDICATION",
    "ANATOMY",
    "BODY",
    "CELL",
    "PROTEIN",
    "DNA",
    "RNA",
    "PER",
    "ORG",
    "LOC",
    "MISC",
    "ENTITY",
}
VALUE_UNIT_INLINE = re.compile(
    r"(?P<value>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)(?P<flag>[HhLl])?(?:\s*(?P<unit>%|[A-Za-z\u00B5\u03BC\u00B5/%][A-Za-z0-9\u00B5\u03BC/%\^]*))?"
)

def normalize_entity_label(label: str) -> str:
    cleaned = (label or "").strip()
    cleaned = re.sub(r"^[BIO]-", "", cleaned, flags=re.IGNORECASE)
    return cleaned.replace("_", "-").upper()

def _select_value_match(segment: str, from_start: bool = True) -> Optional[re.Match]:
    matches = list(VALUE_UNIT_INLINE.finditer(segment))
    if not matches:
        return None
    return matches[0] if from_start else matches[-1]

def _naive_test_from_line(line: str, name: str, entity_offset: int) -> Optional[Dict[str, Any]]:
    raw = line.strip()
    if not raw or not name:
        return None
    rel = max(0, min(len(raw), entity_offset))
    after = raw[rel:]
    before = raw[:rel]
    match = _select_value_match(after, from_start=True) or _select_value_match(before, from_start=False)
    if not match:
        return None

    value_txt = (match.group("value") or "").replace(",", "")
    try:
        value_num = float(value_txt)
    except Exception:
        return None

    unit = normalize_unit_text((match.group("unit") or "").replace(" ", ""))
    flag = (match.group("flag") or "").upper()
    status = "high" if flag == "H" else "low" if flag == "L" else None
    status = status or detect_status(raw) or "unspecified"

    ref = parse_bracket_range(raw)
    ref_min = ref_max = None
    if ref:
        if ref.get("unit"):
            unit = normalize_unit_text(ref.get("unit")) or unit
        inferred = compare_to_range(value_num, ref)
        if inferred:
            status = inferred
        kind = ref.get("kind")
        if kind == "between":
            ref_min = ref.get("lo")
            ref_max = ref.get("hi")
        elif kind in ("lt", "lte"):
            ref_max = ref.get("v")
        elif kind in ("gt", "gte"):
            ref_min = ref.get("v")

    canon = name.strip()
    canon_low = canon.lower().replace(" ", "")
    if "cholesteroltotal" in canon_low:
        canon = "Cholesterol Total"
    elif canon_low.startswith("triglyceride"):
        canon = "Triglyceride"

    abnormal = status if status in ("high", "low", "normal") else "unknown"
    conf = 0.6 + (0.2 if unit else 0.0) + (0.1 if ref else 0.0)

    return {
        "name": canon,
        "value": value_num,
        "unit": unit,
        "status": status,
        "reference": ref or None,
        "ref_min": ref_min,
        "ref_max": ref_max,
        "abnormal": abnormal,
        "evidence": raw,
        "confidence": round(min(conf, 0.95), 2),
    }


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
                .filter(cast(LabReport.user_id, SAText) == str(user.id))
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
        logger.exception("parse_lab failed")
        raise HTTPException(status_code=500, detail=str(e))


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

# ---- Per-user chat history via Redis ----
MAX_CHAT_HISTORY = 50  # keep last 50 turns (user/assistant messages)

def append_chat_history(user_id: str, role: str, content: str) -> None:
    client = get_redis_client()
    if not client:
        return
    payload = json.dumps({"role": role, "content": content})
    key = f"{REDIS_PREFIX}:chat_history:{user_id}"
    try:
        pipe = client.pipeline()
        pipe.rpush(key, payload)
        pipe.ltrim(key, -MAX_CHAT_HISTORY, -1)
        pipe.expire(key, LAB_CACHE_TTL_SECONDS)
        pipe.execute()
    except Exception as exc:
        logger.debug("append_chat_history failed", extra={"error": str(exc)})

def get_chat_history(user_id: str):
    client = get_redis_client()
    if not client:
        return []
    key = f"{REDIS_PREFIX}:chat_history:{user_id}"
    try:
        raw_items = client.lrange(key, -MAX_CHAT_HISTORY, -1)
        history = []
        for item in raw_items:
            try:
                parsed = json.loads(item)
                if isinstance(parsed, dict) and "role" in parsed and "content" in parsed:
                    history.append(parsed)
            except Exception:
                continue
        return history
    except Exception as exc:
        logger.debug("get_chat_history failed", extra={"error": str(exc)})
        return []

# ---- Conversation persistence helpers ----
def _ensure_conversation(db: Session, user: User, conv_id: Optional[str], first_message: str) -> Conversation:
    """Fetch or create a conversation for this user."""
    if conv_id:
        conv = (
            db.query(Conversation)
            .filter(Conversation.id == conv_id, Conversation.user_id == str(user.id))
            .first()
        )
        if conv:
            return conv
    title = (first_message or "").strip()
    if len(title) > 60:
        title = title[:60] + "..."
    conv = Conversation(user_id=str(user.id), title=title or "New chat")
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv

def _get_conversation_context(db: Session, user_id: str, conversation_id: Optional[str]) -> Optional[ConversationContext]:
    if not conversation_id:
        return None
    try:
        return (
            db.query(ConversationContext)
            .filter(
                ConversationContext.conversation_id == conversation_id,
                ConversationContext.user_id == user_id,
            )
            .first()
        )
    except Exception:
        return None

def _format_lab_tests_for_prompt(tests: List[Dict[str, Any]]) -> str:
    if not tests:
        return "None"
    lines = []
    for t in tests[:12]:
        if not isinstance(t, dict):
            continue
        name = t.get("name") or "Test"
        val = t.get("value")
        unit = t.get("unit") or ""
        try:
            num = float(str(val).replace(",", ""))
            if abs(num - round(num)) < 1e-6:
                val_txt = str(int(round(num)))
            else:
                val_txt = f"{num:.2f}".rstrip("0").rstrip(".")
        except Exception:
            val_txt = str(val)
        ref_min = t.get("ref_min")
        ref_max = t.get("ref_max")
        ref_txt = ""
        if ref_min is not None or ref_max is not None:
            if ref_min is not None and ref_max is not None:
                ref_txt = f"ref {ref_min}-{ref_max}{(' ' + unit) if unit else ''}"
            elif ref_min is not None:
                ref_txt = f"ref >= {ref_min}{(' ' + unit) if unit else ''}"
            else:
                ref_txt = f"ref <= {ref_max}{(' ' + unit) if unit else ''}"
        status = (t.get("status") or t.get("abnormal") or "").lower()
        parts = [f"{name}: {val_txt}{(' ' + unit) if unit else ''}"]
        if ref_txt:
            parts.append(ref_txt)
        if status:
            parts.append(f"flagged {status}")
        lines.append("- " + "; ".join(parts))
    return "\n".join(lines) if lines else "None"

def _merge_lab_table_into_prompt(existing: str, tests: List[Dict[str, Any]]) -> str:
    table = _format_lab_tests_for_prompt(tests)
    if table == "None":
        return existing or ""
    start = "[[LAB_TABLE]]"
    end = "[[/LAB_TABLE]]"
    block = f"{start}\nConversation Lab Table (authoritative):\n{table}\n{end}"
    if not existing:
        return block
    if start in existing and end in existing:
        pre = existing.split(start)[0].rstrip()
        post = existing.split(end)[-1].lstrip()
        return "\n\n".join([p for p in [pre, block, post] if p])
    return "\n\n".join([existing.strip(), block])

async def _extract_tests_from_gemini(text: str) -> List[Dict[str, Any]]:
    if not GEMINI_API_KEY or not text.strip():
        return []
    prompt = (
        "Extract lab tests from the text below. Return ONLY a JSON array. "
        "Each item: {\"name\": str, \"value\": number or string, \"unit\": str, "
        "\"ref_min\": number or null, \"ref_max\": number or null, "
        "\"status\": \"low\"|\"high\"|\"normal\"|\"unspecified\"}. "
        "If a range is shown like 13.0-17.0, set ref_min/ref_max. "
        "If no range, use nulls. Do not add extra keys.\n\n"
        f"{text.strip()}"
    )
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
            text_out = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                or ""
            ).strip()
            if "```" in text_out:
                text_out = re.sub(r"^```(?:json)?", "", text_out.strip(), flags=re.IGNORECASE).strip()
                text_out = re.sub(r"```$", "", text_out.strip()).strip()
            parsed = json.loads(text_out)
            if isinstance(parsed, dict) and isinstance(parsed.get("tests"), list):
                return parsed.get("tests")  # type: ignore[return-value]
            if isinstance(parsed, list):
                return parsed  # type: ignore[return-value]
            return []
    except Exception:
        return []

def _persist_message(db: Session, conv: Conversation, user: User, role: MessageRole, content: str) -> Message:
    try:
        conv.updated_at = datetime.utcnow()
    except Exception:
        pass
    msg = Message(
        conversation_id=conv.id,
        user_id=str(user.id),
        role=role.value if hasattr(role, "value") else str(role).lower(),
        content=content,
    )
    try:
        msg.created_at = datetime.utcnow()
    except Exception:
        pass
    try:
        db.add(msg)
        db.commit()
        db.refresh(msg)
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        raise
    return msg

def _should_auto_title(current_title: Optional[str], first_message: str) -> bool:
    if not current_title or not current_title.strip():
        return True
    title = current_title.strip()
    if title.lower() in {"new chat", "new chat...", "new chat…"}:
        return True
    raw = (first_message or "").strip()
    if not raw:
        return False
    truncated = raw[:60] + ("..." if len(raw) > 60 else "")
    return title == raw or title == truncated

async def _generate_chat_title(message: str) -> str:
    raw = (message or "").strip()
    if not raw:
        return "New chat"
    if not AI_EXPLANATION_ENABLED or not GEMINI_API_KEY:
        return raw[:60] + ("..." if len(raw) > 60 else "")
    prompt = (
        "Create a short chat title (max 5 words) based on this user message. "
        "Return only the title, no quotes or punctuation.\n\n"
        f"{raw}"
    )
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
            title = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                or ""
            ).strip()
            if title:
                return title[:60]
    except Exception:
        pass
    return raw[:60] + ("..." if len(raw) > 60 else "")


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
    # Ensure DB session is clean before we start work (in case a prior request aborted)
    try:
        db.rollback()
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

    # Ensure a conversation exists and persist incoming user message
    conv = _ensure_conversation(db, user, getattr(payload, "conversation_id", None), payload.message or "")
    conv_ctx = _get_conversation_context(db, str(user.id), conv.id)
    conv_system_prompt = (conv_ctx.system_prompt or "").strip() if conv_ctx else ""
    conv_lab_tests = None
    if conv_ctx and conv_ctx.lab_tests_json:
        try:
            conv_lab_tests = json.loads(conv_ctx.lab_tests_json)
        except Exception:
            conv_lab_tests = None
    try:
        append_chat_history(str(user.id), "user", payload.message or "")
    except Exception:
        pass
    try:
        _persist_message(db, conv, user, MessageRole.USER, payload.message or "")
    except Exception:
        logger.warning("db_persist_user_message_failed", exc_info=True)

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

        # Trigger condition: lab-like lines with values/units or reference ranges
        lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        line_count = len(lines)
        has_unit = bool(re.search(r"(?i)\b(ng/mL|mmol/L|mg/dL|g/dL|u/l|u/l|iu/l|%|x?10\^)\b", cleaned))
        has_brackets = ("[" in cleaned and "]" in cleaned)
        has_ref_word = bool(re.search(r"(?i)\bref(?:erence)?\b", cleaned))
        looks_like_table = any(
            re.search(r"\d", ln) and (("[" in ln and "]" in ln) or re.search(r"(?i)\bref\b", ln) or re.search(r"(?i)\b(ng/mL|mmol/L|mg/dL|g/dL|u/l|iu/l|%|x?10\^)\b", ln))
            for ln in lines
        )
        if (line_count >= 2 and (has_unit or has_brackets or has_ref_word or looks_like_table)) or looks_like_table:
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
            normal_n = len((presentation.get("normal") or [])) if isinstance(presentation, dict) else 0
            conf = float(presentation.get("confidence") or 0.0) if isinstance(presentation, dict) else 0.0

            # Build user_view for UI consumption when we have parsed lines
            if parsed_n >= 1 or (abnormal_n + normal_n) >= 2:
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
                        cast(LabReport.user_id, SAText) == str(user.id),
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
    try:
        if isinstance(latest_lab_struct_early, dict):
            tests = latest_lab_struct_early.get("tests")
            if isinstance(tests, list):
                apply_universal_reference_ranges(tests)
            labs_obj = latest_lab_struct_early.get("labs") if isinstance(latest_lab_struct_early.get("labs"), dict) else None
            analytes = labs_obj.get("analytes") if isinstance(labs_obj, dict) else None
            if isinstance(analytes, list):
                apply_universal_reference_ranges(analytes)
    except Exception:
        pass
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
            lt = lab_triage(latest_lab_struct, profile_dict)
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
        # Profile-aware escalations
        profile_risk_reasons: list = []
        try:
            if profile_dict:
                age_val = profile_dict.get("age")
                if isinstance(age_val, (int, float)):
                    if age_val >= 75 and triage_level != "high":
                        triage_level = "high"
                        profile_risk_reasons.append("age >=75")
                    elif age_val >= 65 and triage_level == "low":
                        triage_level = "moderate"
                        profile_risk_reasons.append("age >=65")
                conds = [str(c).lower() for c in (profile_dict.get("conditions") or []) if str(c).strip()]
                meds = [str(m).lower() for m in (profile_dict.get("medications") or []) if str(m).strip()]
                high_risk_conditions = {
                    "heart failure", "coronary", "cad", "cardiomyopathy", "cancer", "immunosuppression",
                    "transplant", "pregnancy", "pregnant", "ckd", "kidney", "pulmonary hypertension"
                }
                hits: list[str] = []
                for c in conds:
                    for label in high_risk_conditions:
                        if label in c:
                            hits.append(label)
                            break
                if hits:
                    if triage_level == "low":
                        triage_level = "moderate"
                    elif triage_level == "moderate" and any(h in ("heart failure", "coronary", "cad", "cardiomyopathy") for h in hits):
                        triage_level = "high"
                    profile_risk_reasons.append(f"condition: {', '.join(sorted(set(hits)))}")
                if meds and any(tag in m for m in meds for tag in ("prednisone", "steroid", "immunosuppress", "chemo")):
                    if triage_level == "low":
                        triage_level = "moderate"
                    profile_risk_reasons.append("immunosuppressive medication")
        except Exception:
            pass
        if profile_risk_reasons:
            triage_reasons.extend(f"profile: {r}" for r in profile_risk_reasons)
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
    conv_lab_block = _format_lab_tests_for_prompt(conv_lab_tests or [])
    conv_lab_prompt = ""
    if conv_lab_tests:
        conv_lab_prompt = f"--- Conversation Lab Table (authoritative) ---\n{conv_lab_block}\n\n"
    conv_system_prompt_block = ""
    if conv_system_prompt:
        conv_system_prompt_block = f"--- Conversation System Prompt (user-edited) ---\n{conv_system_prompt}\n\n"
    prompt = (
        "You are a helpful medical assistant. Use the provided context to answer the user's question. "
        "Personalize guidance using the patient's age, sex, pre-existing conditions, and medications from the profile. "
        "If a profile field is marked N/A, briefly ask for it in one concise sentence. "
        "The context includes the user's profile, their latest lab report, and their latest symptom summary. "
        "Provide general, educational information and you may answer follow-up questions about likely causes of abnormal tests (e.g., high ferritin) and typical next steps. "
        "Avoid diagnosis or prescriptions; include caveats and when to seek care. "
        "Keep responses concise and easy to understand. "
        "If Lab Report Availability says Available: yes, do not say you lack the lab report; use the lab content provided (structured or raw text). "
        "Use the Key Tests block for follow-up questions; quote the exact values, units, and reference ranges as written there. "
        "Do not round or drop decimals; if the value shows a decimal, keep it. "
        "If any test includes reference_source 'universal' or reference_note, clearly state that the range is a general reference and not from the user's lab report.\n\n"
        f"{conv_system_prompt_block}"
        f"{conv_lab_prompt}"
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
    # Ensure a safe fallback if the model returned nothing or failed silently
    if not ai_explanation:
        ai_explanation = "AI explanation unavailable; showing structured recommendations."
        ai_explanation_source = "fallback"

    # Rely on conversation lab table + history for consistency (no auto-injected lab facts).

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
    try:
        _persist_message(db, conv, user, MessageRole.ASSISTANT, ai_explanation or "")
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        logger.warning("db_persist_assistant_message_failed", exc_info=True)

    try:
        if _should_auto_title(getattr(conv, "title", None), payload.message or ""):
            new_title = await _generate_chat_title(payload.message or "")
            if new_title and new_title != conv.title:
                conv.title = new_title
                db.add(conv)
                db.commit()
    except Exception:
        try:
            db.rollback()
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
        conversation_id=str(conv.id),
        conversation_title=conv.title,
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


@app.post("/api/chat/image", response_model=ChatOut)
@limiter.limit("30/minute", key_func=user_rate_key)
async def chat_image(
    request: Request,
    file: UploadFile = File(...),
    conversation_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    try:
        request.state.user_id = str(user.id)
    except Exception:
        pass

    mt = (file.content_type or "").lower()
    if not mt.startswith("image/"):
        raise HTTPException(status_code=415, detail="Unsupported file type. Please upload a JPG or PNG image.")

    data = await file.read()
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise HTTPException(status_code=413, detail=f"File size exceeds the {MAX_FILE_MB}MB limit")

    request_id = str(_uuid.uuid4())
    user_text = "Uploaded lab image."

    conv = _ensure_conversation(db, user, conversation_id, user_text)
    try:
        append_chat_history(str(user.id), "user", user_text)
    except Exception:
        pass
    try:
        _persist_message(db, conv, user, MessageRole.USER, user_text)
    except Exception:
        logger.warning("db_persist_user_message_failed", exc_info=True)

    raw_text = ""
    lab_tests: List[Dict[str, Any]] = []
    try:
        raw_text, source = extract_text_from_bytes(data, file.filename or "upload", file.content_type or "")
        lab_tests = await _extract_tests_from_gemini(raw_text)
    except Exception:
        raw_text = ""
        lab_tests = []
    if lab_tests:
        try:
            apply_universal_reference_ranges(lab_tests)
        except Exception:
            pass
        try:
            ctx = _get_conversation_context(db, str(user.id), conv.id)
            if not ctx:
                ctx = ConversationContext(
                    user_id=str(user.id),
                    conversation_id=conv.id,
                    lab_tests_json=json.dumps(lab_tests, ensure_ascii=False),
                )
                db.add(ctx)
            else:
                ctx.lab_tests_json = json.dumps(lab_tests, ensure_ascii=False)
            ctx.system_prompt = _merge_lab_table_into_prompt(ctx.system_prompt or "", lab_tests)
            db.commit()
        except Exception:
            try:
                db.rollback()
            except Exception:
                pass
        try:
            structured = {"tests": lab_tests, "meta": {"engine": "gemini"}}
            summary = f"Parsed {len(lab_tests)} analytes via Gemini."
            lr = LabReport(
                user_id=str(user.id),
                title=file.filename or "Lab Report",
                raw_text=raw_text,
                structured_json=structured,
                summary=summary,
            )
            db.add(lr)
            db.commit()
            try:
                conv.active_lab_id = str(lr.id)
                db.add(conv)
                db.commit()
            except Exception:
                try:
                    db.rollback()
                except Exception:
                    pass
            set_latest_lab_cache(str(user.id), structured)
        except Exception:
            try:
                db.rollback()
            except Exception:
                pass

    context, _notice = build_chat_context(db, user, user_text)
    lab_table = _format_lab_tests_for_prompt(lab_tests) if lab_tests else "None"
    lab_table_block = f"\n\n--- Extracted Lab Table (from image OCR) ---\n{lab_table}\n" if lab_tests else ""
    prompt = (
        "You are a helpful medical assistant. Analyze the attached lab report image. "
        "Extract all tests with value, unit, reference range (if visible), and flag high/low/normal when possible. "
        "Then provide a concise summary and a plain-language explanation. "
        "Avoid diagnosis or prescriptions; include caveats and when to seek care.\n\n"
        f"{context}{lab_table_block}\n"
    )

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
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
                payload = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {"text": prompt.strip()},
                                {
                                    "inline_data": {
                                        "mime_type": mt,
                                        "data": base64.b64encode(data).decode("ascii"),
                                    }
                                },
                            ],
                        }
                    ]
                }
                r = await client.post(
                    url,
                    params={"key": GEMINI_API_KEY},
                    headers={"Content-Type": "application/json"},
                    json=payload,
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
        except httpx.TimeoutException:
            timed_out = True
            ai_explanation_source = "fallback"
            ai_explanation = "AI explanation unavailable; please try again."
        except Exception:
            timed_out = False
            ai_explanation_source = "fallback"
            ai_explanation = "AI explanation unavailable; please try again."

    if not ai_explanation:
        ai_explanation = "AI explanation unavailable; please try again."
        ai_explanation_source = "fallback"

    try:
        append_chat_history(str(user.id), "assistant", ai_explanation or "")
    except Exception:
        pass
    try:
        _persist_message(db, conv, user, MessageRole.ASSISTANT, ai_explanation or "")
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
        logger.warning("db_persist_assistant_message_failed", exc_info=True)

    try:
        if _should_auto_title(getattr(conv, "title", None), user_text):
            title_seed = ai_explanation if ai_explanation_source == "model" else "Lab image analysis"
            new_title = await _generate_chat_title(title_seed)
            if new_title and new_title != conv.title:
                conv.title = new_title
                db.add(conv)
                db.commit()
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass

    symptom_analysis = {"symptoms": [], "possible_tests": [], "confidence": 0.0, "event_id": None}
    local_recs = {
        "priority": "low",
        "actions": ["Monitor symptoms and rest"],
        "follow_up": "If symptoms persist >48h, worsen, or include red flags (fainting, chest pain), seek medical care.",
        "rationale": "Initial self-care suggestions based on reported symptoms.",
    }
    disclaimer = "For educational purposes only. Consult a medical professional for medical advice."

    return ChatOut(
        conversation_id=str(conv.id),
        conversation_title=conv.title,
        request_id=request_id,
        summary="Image analyzed.",
        symptom_analysis=symptom_analysis,
        local_recommendations=local_recs,
        ai_explanation=ai_explanation,
        ai_explanation_source=ai_explanation_source,
        timed_out=timed_out,
        disclaimer=disclaimer,
        pipeline=None,
        missing_fields=None,
        triage=None,
        user_view=None,
    )



@router.post("/explain", response_model=ExplainOut)
@limiter.limit("2/minute")
async def explain(request: Request, payload: ExplainIn, db: Session = Depends(get_db), user: User = Depends(get_current_user), response: Response = None) -> ExplainOut:
    structured = payload.structured
    try:
        if isinstance(structured, dict):
            tests = structured.get("tests")
            if isinstance(tests, list):
                apply_universal_reference_ranges(tests)
    except Exception:
        pass
    def _format_range(test: Dict[str, Any]) -> str:
        if test.get("ref_min") is not None or test.get("ref_max") is not None:
            ref_min = test.get("ref_min")
            ref_max = test.get("ref_max")
            if ref_min is not None and ref_max is not None:
                base = f"{ref_min}-{ref_max}"
                if test.get("reference_source") == "universal" or test.get("reference_note"):
                    return f"{base} (general reference; not from this lab report)"
                return base
            if ref_min is not None:
                base = f">= {ref_min}"
                if test.get("reference_source") == "universal" or test.get("reference_note"):
                    return f"{base} (general reference; not from this lab report)"
                return base
            if ref_max is not None:
                base = f"<= {ref_max}"
                if test.get("reference_source") == "universal" or test.get("reference_note"):
                    return f"{base} (general reference; not from this lab report)"
                return base
        ref = test.get("reference")
        if isinstance(ref, dict):
            kind = ref.get("kind")
            if kind == "between":
                base = f"{ref.get('lo')}-{ref.get('hi')}"
                if test.get("reference_source") == "universal" or test.get("reference_note"):
                    return f"{base} (general reference; not from this lab report)"
                return base
            if kind in ("lt", "lte"):
                op = "<=" if kind == "lte" else "<"
                base = f"{op} {ref.get('v')}"
                if test.get("reference_source") == "universal" or test.get("reference_note"):
                    return f"{base} (general reference; not from this lab report)"
                return base
            if kind in ("gt", "gte"):
                op = ">=" if kind == "gte" else ">"
                base = f"{op} {ref.get('v')}"
                if test.get("reference_source") == "universal" or test.get("reference_note"):
                    return f"{base} (general reference; not from this lab report)"
                return base
        return "N/A"

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
        "patient-friendly explanation (<=200 words). Not medical advice. "
        "If any test includes reference_source 'universal' or reference_note, say the range is general and not from the lab report.\n\n" +
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
        tests = structured.get("tests") or []
        first = tests[0] if tests else {}
        offline_summary = OFFLINE_SUMMARY_TEMPLATE.format(
            test_name=first.get("name", "Test"),
            value=first.get("value", "?"),
            unit=first.get("unit", ""),
            status=first.get("status", "unspecified"),
            range=_format_range(first),
        )
        return ExplainOut(explanation=offline_summary)

# --- Conversation list/history endpoints ---
@router.get("/chat/conversations", response_model=List[ConversationOut])
def list_conversations(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    items = (
        db.query(Conversation)
        .filter(Conversation.user_id == str(user.id))
        .order_by(Conversation.updated_at.desc())
        .all()
    )
    return items

@router.get("/chat/conversations/{conversation_id}/context", response_model=ConversationContextOut)
def get_conversation_context(conversation_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == str(user.id))
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    ctx = (
        db.query(ConversationContext)
        .filter(
            ConversationContext.conversation_id == conversation_id,
            ConversationContext.user_id == str(user.id),
        )
        .first()
    )
    lab_tests = None
    if ctx and ctx.lab_tests_json:
        try:
            lab_tests = json.loads(ctx.lab_tests_json)
        except Exception:
            lab_tests = None
    return ConversationContextOut(
        conversation_id=conversation_id,
        system_prompt=ctx.system_prompt if ctx else None,
        lab_tests=lab_tests,
    )

@router.put("/chat/conversations/{conversation_id}/context", response_model=ConversationContextOut)
def update_conversation_context(conversation_id: str, payload: ConversationContextIn, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == str(user.id))
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    ctx = (
        db.query(ConversationContext)
        .filter(
            ConversationContext.conversation_id == conversation_id,
            ConversationContext.user_id == str(user.id),
        )
        .first()
    )
    if not ctx:
        ctx = ConversationContext(user_id=str(user.id), conversation_id=conversation_id, system_prompt=payload.system_prompt)
        db.add(ctx)
    else:
        ctx.system_prompt = payload.system_prompt
    db.commit()
    lab_tests = None
    if ctx.lab_tests_json:
        try:
            lab_tests = json.loads(ctx.lab_tests_json)
        except Exception:
            lab_tests = None
    return ConversationContextOut(
        conversation_id=conversation_id,
        system_prompt=ctx.system_prompt,
        lab_tests=lab_tests,
    )

@router.post("/chat/conversations", response_model=ConversationOut)
def create_conversation(payload: ConversationCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    title = (payload.title or "").strip() or "New chat"
    conv = Conversation(user_id=str(user.id), title=title)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv

@router.get("/chat/conversations/{conversation_id}/messages", response_model=List[MessageOut])
def get_conversation_messages(conversation_id: str, limit: int = 50, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == str(user.id))
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    msgs = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(msgs))

@router.delete("/chat/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    conv = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == str(user.id))
        .first()
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.delete(conv)
    db.commit()
    return None

# Register the local /api/explain route after it has been attached to the router
app.include_router(router)

def _sanitize_filename(name: str) -> str:
    name = os.path.basename(name or "file")
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return name or "file"


MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "10"))


@app.get("/api/extract_text")
async def extract_text_info():
    """
    Friendly helper for accidental GET requests. The extractor only supports POST with multipart/form-data.
    """
    return {
        "detail": "Use POST multipart/form-data with `file` (PDF or image).",
        "max_file_mb": MAX_FILE_MB,
        "supported_types": ["application/pdf", "image/*"],
    }


@app.post("/api/extract_text")
@limiter.limit("10/minute", key_func=user_rate_key)
async def extract_text(request: Request, file: UploadFile = File(...), user: User = Depends(get_current_user)):
    # Record user id on request state for per-user rate limiting
    try:
        request.state.user_id = str(user.id)
    except Exception:
        pass
    try:
        logger.info(
            json.dumps(
                {
                    "event": "extract_text_received",
                    "filename": (file.filename or "").strip() or "unnamed",
                    "content_type": (file.content_type or "").lower(),
                    "user": getattr(user, "email", None),
                }
            )
        )
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
            # Fallback to OCR if PDF text is empty or parsing failed
            try:
                images = []
                try:
                    from pdf2image import convert_from_bytes  # type: ignore
                    images = convert_from_bytes(data)
                except Exception:
                    images = []
                if not images:
                    raise e
                ocr_text = []
                for img in images:
                    ocr_text.append(pytesseract.image_to_string(img, lang="eng"))
                joined = "\n".join(ocr_text).strip()
                if not joined:
                    raise ValueError("No text found via OCR")
                return {"text": joined, "source": "pdf_ocr_fallback"}
            except Exception as ocr_exc:
                raise HTTPException(status_code=400, detail=f"PDF extraction failed: {e}; OCR fallback failed: {ocr_exc}")

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
