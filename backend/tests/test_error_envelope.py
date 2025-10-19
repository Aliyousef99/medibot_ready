from unittest.mock import patch


def test_http_exception_envelope(client):
    # list_models returns 400 when no API key
    with patch("backend.app.GEMINI_API_KEY", ""):
        r = client.get("/api/list_models")
    assert r.status_code == 400
    j = r.json()
    assert j["code"] == "BAD_REQUEST"
    assert "trace_id" in j


def test_unhandled_exception_envelope(client):
    with patch("backend.app.parse_lab_text", side_effect=ValueError("boom")):
        r = client.post("/api/parse_lab", json={"text": "abc"})
    assert r.status_code == 500
    j = r.json()
    assert j["code"] == "INTERNAL_SERVER_ERROR"
    assert j["details"] == "boom"
    assert "trace_id" in j
