from backend.app import app


def test_limits_extract_text(client):
    app.state.limiter.reset()
    file = ("a.pdf", b"%PDF-1.4\n", "application/pdf")
    for _ in range(2):
        client.post("/api/extract_text", files={"file": file})
    r = client.post("/api/extract_text", files={"file": file})
    if r.status_code == 429:
        j = r.json()
        assert j["code"] == "TOO_MANY_REQUESTS"
        assert "trace_id" in j

