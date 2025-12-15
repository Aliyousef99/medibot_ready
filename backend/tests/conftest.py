import os
import sys
from pathlib import Path
import types
import uuid
import io
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Ensure tests use an in-memory SQLite DB and JSON columns stay generic
os.environ.setdefault("DATABASE_URL", "sqlite://")
os.environ.setdefault("FORCE_GENERIC_JSON", "1")

# Ensure the project root is on sys.path so `import backend` works when running
# pytest from the repository root.
ROOT_DIR = Path(__file__).resolve().parents[2]  # repository root
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app import app
from backend.db.session import Base, get_db
from backend.auth.deps import get_current_user
from types import SimpleNamespace


# In-memory SQLite shared across connections
engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    implicit_returning=False,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session", autouse=True)
def create_schema():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def _override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = _override_get_db
app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id="user-1", email="u@example.com")

# Ensure code paths that import SessionLocal directly (legacy chat routes) use the test engine/session
import backend.db.session as session_mod
session_mod.engine = engine
session_mod.SessionLocal = TestingSessionLocal
import backend.models as models_mod
models_mod.engine = engine
# Legacy chat routes import SessionLocal directly; point them to the test session
import backend.routes.chat_routes as chat_routes_mod
chat_routes_mod.SessionLocal = TestingSessionLocal


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    # ensure limiter state fresh each test
    if hasattr(app.state, "limiter") and hasattr(app.state.limiter, "reset"):
        app.state.limiter.reset()
    yield


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_gemini(monkeypatch):
    # Ensure no outbound calls
    monkeypatch.setenv("GEMINI_API_KEY", "")
    class Dummy:
        @staticmethod
        def rewrite_patient_copy(actions):
            return "\n".join(f"- {a}" for a in actions)
    import backend.services.recs as recs
    monkeypatch.setattr(recs, "gemini", Dummy)


@pytest.fixture
def mock_ner(monkeypatch):
    # Deterministic NER pipeline
    def fake_pipeline(text):
        spans = []
        t = text.lower()
        for term in ("chest pain", "fever"):
            i = t.find(term)
            if i >= 0:
                spans.append({
                    "word": term,
                    "entity_group": "SYMPTOM",
                    "score": 0.9,
                    "start": i,
                    "end": i + len(term),
                })
        return spans
    def get_ner_pipeline():
        return fake_pipeline, "mock-biobert", None
    import backend.services.ner as ner
    monkeypatch.setattr(ner, "get_ner_pipeline", get_ner_pipeline)


@pytest.fixture
def mock_ocr(monkeypatch):
    # Tesseract image OCR always returns fixed text
    import backend.services.ocr as ocr
    monkeypatch.setattr("pytesseract.image_to_string", lambda img, lang=None: "mock-ocr-text")
    # PdfReader to return one page with text
    class _Pg:
        def extract_text(self):
            return "pdf text"
    class _Reader:
        def __init__(self, *_a, **_k):
            self.pages = [_Pg()]
    import pypdf
    monkeypatch.setattr(pypdf, "PdfReader", _Reader)


def pytest_configure(config):
    """Allow overriding the coverage floor via environment variable for local runs.

    Example (PowerShell):
      $env:PYTEST_COV_FAIL_UNDER='0'; pytest
    """
    env_floor = os.getenv("PYTEST_COV_FAIL_UNDER") or os.getenv("COV_FAIL_UNDER")
    if env_floor is not None:
        try:
            config.option.cov_fail_under = float(env_floor)
        except Exception:
            # ignore invalid values; keep existing floor
            pass
