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
from backend.routes import auth_routes, history_routes, symptoms_routes, recs_routes
from backend.routers.profile import router as profile_router

# --- app & router setup ---
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY", "") or "").strip()
# Allow overriding the Gemini model via env; default to 2.5 flash to match README/services
GEMINI_MODEL = (os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()
HUGGINGFACE_TOKEN = (os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_API_KEY") or "").strip()
BIOBERT_MODEL_NAME = os.getenv("BIOBERT_MODEL_NAME", "d4data/biobert_ner").strip() or "d4data/biobert_ner"

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
app.include_router(recs_routes.router, prefix="/api/recommendations")

# --- schemas ---
class ParseRequest(BaseModel):
    text: constr(max_length=5000)

class ChatIn(BaseModel):
    message: str

class ChatOut(BaseModel):
    response: str
    pipeline: Optional[Dict[str, Any]] = None

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
