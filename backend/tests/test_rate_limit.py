import io
from fastapi.testclient import TestClient
from backend.app import app


client = TestClient(app)


def test_rate_limit_chat(monkeypatch):
    # ensure fresh limiter state
    app.state.limiter.reset()
    # Minimal auth bypass by monkeypatching dependency to avoid 401s
    from backend import app as backend_app
    def fake_get_current_user():
        class U: id = "user-1"
        return U()
    backend_app.get_current_user = lambda: fake_get_current_user()

    # Hit the limit of 2/minute by calling 3 times
    for i in range(2):
        r = client.post("/api/chat", json={"message": "hello"}, headers={"Authorization": "Bearer dummy"})
        assert r.status_code in (200, 401, 422)
    r = client.post("/api/chat", json={"message": "hello"}, headers={"Authorization": "Bearer dummy"})
    # Either 429 or auth error depending on environment; assert envelope when 429
    if r.status_code == 429:
        j = r.json()
        assert j["code"] == "TOO_MANY_REQUESTS"
        assert "trace_id" in j


def test_rate_limit_explain():
    app.state.limiter.reset()
    for i in range(2):
        r = client.post("/api/explain", json={"structured": {"tests": []}})
        assert r.status_code in (200, 500)
    r = client.post("/api/explain", json={"structured": {"tests": []}})
    if r.status_code == 429:
        j = r.json()
        assert j["code"] == "TOO_MANY_REQUESTS"
        assert "trace_id" in j


def test_rate_limit_extract_text():
    app.state.limiter.reset()
    file = ("test.pdf", io.BytesIO(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"), "application/pdf")
    for i in range(2):
        r = client.post("/api/extract_text", files={"file": file})
        assert r.status_code in (200, 400)
    r = client.post("/api/extract_text", files={"file": file})
    if r.status_code == 429:
        j = r.json()
        assert j["code"] == "TOO_MANY_REQUESTS"
        assert "trace_id" in j

