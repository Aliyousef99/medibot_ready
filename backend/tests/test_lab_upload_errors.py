def test_upload_empty_file_returns_400(client):
    r = client.post("/api/history/labs/upload", files={"file": ("empty.txt", b"", "text/plain")})
    assert r.status_code == 400
    assert "Empty file" in r.text


def test_upload_bad_content_type_graceful(client, monkeypatch):
    # Force process_upload to raise to exercise 400 path
    monkeypatch.setattr("backend.routes.history_routes.process_upload", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad file")))
    r = client.post("/api/history/labs/upload", files={"file": ("weird.bin", b"abc", "application/octet-stream")})
    assert r.status_code == 400
    assert "Upload failed" in r.text
