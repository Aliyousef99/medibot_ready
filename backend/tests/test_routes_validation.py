import io
import json
import pytest


def test_chat_handles_empty_message_with_safe_response(client):
    r = client.post("/api/chat", json={"message": ""})
    assert r.status_code == 200
    body = r.json()
    # Even with empty content we should respond safely
    assert body.get("disclaimer")


def test_chat_falls_back_when_llm_fails(client, monkeypatch):
    async def boom(system, messages, timeout_s=20):
        raise RuntimeError("llm down")
    # Force generate_chat to fail so the route exercises fallback path
    monkeypatch.setattr("backend.services.gemini.generate_chat", boom)
    r = client.post("/api/chat", json={"message": "hello there"})
    assert r.status_code == 200
    body = r.json()
    # We should still get a safe explanation and disclaimer
    assert body.get("ai_explanation") is not None
    assert body.get("disclaimer")


def test_recommendations_require_payload(client):
    r = client.post("/api/recommendations/generate", json={})
    assert r.status_code == 400
    assert "Either labs or symptoms" in r.text


def test_lab_upload_empty_file(client):
    r = client.post(
        "/api/history/labs/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert r.status_code == 400
    assert "Empty file" in r.text


def test_lab_upload_process_failure(client, monkeypatch):
    # Make process_upload raise to ensure we translate to 400
    def boom(*args, **kwargs):
        raise ValueError("boom")
    monkeypatch.setattr("backend.routes.history_routes.process_upload", boom)
    r = client.post(
        "/api/history/labs/upload",
        files={"file": ("report.txt", b"abc", "text/plain")},
    )
    assert r.status_code == 400
    assert "Upload failed" in r.text
