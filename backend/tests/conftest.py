import os
import types
import uuid
import io
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

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
