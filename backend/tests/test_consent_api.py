from backend.db.session import SessionLocal
from backend.models.user import UserProfile


def test_consent_get_creates_profile(client):
    db = SessionLocal()
    db.query(UserProfile).delete()
    db.commit()
    db.close()

    r = client.get("/api/privacy/consent")
    assert r.status_code == 200
    body = r.json()
    assert body["consent_given"] is False


def test_consent_set_and_store(client):
    r = client.post("/api/privacy/consent", json={"consent_given": True})
    assert r.status_code == 200
    body = r.json()
    assert body["consent_given"] is True
    assert body.get("consent_at") is not None
