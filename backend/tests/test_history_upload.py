import io
from fastapi.testclient import TestClient


def test_upload_lab_persists_metadata(client: TestClient, monkeypatch, tmp_path):
    # Avoid writing to random locations
    import backend.services.storage as storage
    fake_path = tmp_path / "saved.bin"
    monkeypatch.setattr(storage, "store_local_upload", lambda data, filename: (str(fake_path), "saved.bin"))

    content = b"Hemoglobin 10 g/dL"
    files = {"file": ("lab.txt", io.BytesIO(content), "text/plain")}
    resp = client.post("/api/history/labs/upload", files=files)
    assert resp.status_code == 201, resp.text
    payload = resp.json()
    assert payload["structured_json"]["file_meta"]["filename"] == "lab.txt"
    assert payload["structured_json"]["file_meta"]["size_bytes"] == len(content)
    assert payload["raw_text"]
    assert payload["summary"]
