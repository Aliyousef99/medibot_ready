import asyncio
import uuid
from typing import List, Dict, Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app import app
from backend.db.session import SessionLocal
from types import SimpleNamespace
from backend.models.lab_report import LabReport
from backend.models.message import Message, MessageRole


def auth_headers_for() -> dict:
    return {}


@pytest.fixture
def db() -> Session:
    return SessionLocal()


@pytest.fixture(autouse=True)
def override_current_user(monkeypatch):
    # FastAPI dependency override to bypass JWT/DB user lookup
    from backend.auth import deps as deps_mod
    def _fake_user():
        return SimpleNamespace(id="user-1", email="u@example.com")
    monkeypatch.setattr(deps_mod, "get_current_user", lambda: _fake_user())


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def make_lab(db: Session, user, ldl_value: int = 180) -> LabReport:
    lr = LabReport(
        id=str(uuid.uuid4()),
        user_id=str(user.id),
        raw_text="LDL: {} mg/dL".format(ldl_value),
        structured_json={
            "tests": [
                {"name": "ldl", "value": ldl_value, "unit": "mg/dL"},
            ]
        },
        summary="Demo lab",
    )
    db.add(lr)
    db.commit()
    return lr


def test_chat_start_and_send_persists_messages(client: TestClient, db: Session, monkeypatch):
    # Mock LLM to deterministic reply
    async def fake_gen(system: str, messages: List[Dict[str, Any]], timeout_s: int = 20) -> str:
        return "assistant: ok"
    monkeypatch.setattr("backend.services.gemini.generate_chat", fake_gen)

    headers = auth_headers_for()
    # Start conversation
    r = client.post("/api/chat/start", json={"title": "T1"}, headers=headers)
    assert r.status_code == 200, r.text
    conv_id = r.json()["conversation_id"]

    # Send a message
    r2 = client.post("/api/chat/send", json={"conversation_id": conv_id, "message": "hello"}, headers=headers)
    assert r2.status_code == 200, r2.text

    # Verify messages persisted: one user + one assistant
    msgs = db.query(Message).all()
    roles = sorted(m.role.value for m in msgs)
    assert roles == ["assistant", "user"]


def test_chat_uses_active_lab_context(client: TestClient, db: Session, monkeypatch):
    lab = make_lab(db, SimpleNamespace(id="user-1"), ldl_value=180)

    # Mock LLM to assert lab content in system prompt
    async def fake_gen(system: str, messages: List[Dict[str, Any]], timeout_s: int = 20) -> str:
        assert '"ldl"' in system and '180' in system
        return "ack: ldl 180"
    monkeypatch.setattr("backend.services.gemini.generate_chat", fake_gen)

    headers = auth_headers_for()
    r = client.post("/api/chat/start", json={"title": "T2", "active_lab_id": lab.id}, headers=headers)
    conv_id = r.json()["conversation_id"]

    r2 = client.post("/api/chat/send", json={"conversation_id": conv_id, "message": "what about cholesterol?"}, headers=headers)
    assert r2.status_code == 200
    assert "conversation_id" in r2.json()


def test_latest_lab_fallback(client: TestClient, db: Session, monkeypatch):
    lab = make_lab(db, SimpleNamespace(id="user-1"), ldl_value=155)

    async def fake_gen(system: str, messages: List[Dict[str, Any]], timeout_s: int = 20) -> str:
        # No active_lab provided; should fall back to latest lab
        assert '"ldl"' in system and '155' in system
        return "ok"
    monkeypatch.setattr("backend.services.gemini.generate_chat", fake_gen)

    headers = auth_headers_for()
    r = client.post("/api/chat/start", json={"title": "T3"}, headers=headers)
    conv_id = r.json()["conversation_id"]
    r2 = client.post("/api/chat/send", json={"conversation_id": conv_id, "message": "question"}, headers=headers)
    assert r2.status_code == 200


def test_rolling_summary_triggers_and_shrinks_context(client: TestClient, db: Session, monkeypatch):
    # Prepare many messages
    headers = auth_headers_for()
    r = client.post("/api/chat/start", json={"title": "Long"}, headers=headers)
    conv_id = r.json()["conversation_id"]

    # Insert many messages directly to simulate long history
    # Each message is ~200 chars -> 50 tokens; 200 msgs -> 10k tokens
    long_text = "x" * 200
    msgs = []
    for i in range(120):
        m = Message(conversation_id=conv_id, user_id="user-1", role=MessageRole.USER, content=f"{i}-{long_text}")
        msgs.append(m)
    db.add_all(msgs)
    db.commit()

    # Mock summarizer + LLM
    async def fake_sum(messages, previous_summary=None):
        return "SUMMARY"
    monkeypatch.setattr("backend.services.summarizer.summarize_messages", fake_sum)

    async def fake_gen(system: str, messages: List[Dict[str, Any]], timeout_s: int = 20) -> str:
        return "resp"
    monkeypatch.setattr("backend.services.gemini.generate_chat", fake_gen)

    # Send one more message to trigger summarization path
    r2 = client.post("/api/chat/send", json={"conversation_id": conv_id, "message": "trigger"}, headers=headers)
    assert r2.status_code == 200

    # After send, conversation should have a rolling summary and old messages trimmed
    from backend.models.conversation import Conversation
    conv = db.query(Conversation).get(conv_id)
    assert conv.rolling_summary == "SUMMARY"
    remaining = db.query(Message).filter(Message.conversation_id == conv_id).count()
    assert remaining <= 50 + 2  # retention plus latest user/assistant


def test_history_pagination(client: TestClient, db: Session, monkeypatch):
    async def fake_gen(system: str, messages: List[Dict[str, Any]], timeout_s: int = 20) -> str:
        return "ok"
    monkeypatch.setattr("backend.services.gemini.generate_chat", fake_gen)

    headers = auth_headers_for()
    r = client.post("/api/chat/start", json={"title": "P"}, headers=headers)
    conv_id = r.json()["conversation_id"]

    ids = []
    for i in range(5):
        r2 = client.post("/api/chat/send", json={"conversation_id": conv_id, "message": f"m{i}"}, headers=headers)
        assert r2.status_code == 200
        # capture assistant id from response
        ids.append(r2.json()["message"]["id"])  # type: ignore[index]

    # Fetch last 2
    r3 = client.get(f"/api/chat/{conv_id}/history", params={"limit": 2}, headers=headers)
    assert r3.status_code == 200
    page1 = r3.json()["messages"]
    assert len(page1) == 2
    # Fetch previous before first id in page1
    before = page1[0]["id"]
    r4 = client.get(f"/api/chat/{conv_id}/history", params={"limit": 2, "before_id": before}, headers=headers)
    assert r4.status_code == 200
    page2 = r4.json()["messages"]
    # No overlap with page1
    assert not {m["id"] for m in page1} & {m["id"] for m in page2}


def test_trace_id_header_present(client: TestClient, db: Session, monkeypatch):
    async def fake_gen(system: str, messages: List[Dict[str, Any]], timeout_s: int = 20) -> str:
        return "ok"
    monkeypatch.setattr("backend.services.gemini.generate_chat", fake_gen)

    headers = auth_headers_for()
    r = client.post("/api/chat/start", json={"title": "Trace"}, headers=headers)
    assert r.status_code == 200
    assert any(h.lower() == "x-trace-id" for h in r.headers.keys())




from backend.models.conversation import Conversation
conv = None

