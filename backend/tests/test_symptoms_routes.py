import uuid
from backend.auth.jwt import create_access_token


def _auth_headers_for(user_id: str, email: str = "u@example.com"):
    token = create_access_token({"sub": user_id, "email": email})
    return {"Authorization": f"Bearer {token}"}


def test_symptoms_analyze_200(client, mock_ner):
    # create a user row
    from backend.db.session import SessionLocal
    from backend.models.user import User
    db = SessionLocal()
    uid = str(uuid.uuid4())
    db.add(User(id=uid, email="u@example.com", hashed_password="x"))
    db.commit()
    db.close()

    headers = _auth_headers_for(uid)
    r = client.post("/api/symptoms/analyze", headers=headers, json={"text": "I have fever"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["urgency"] in ("normal", "urgent")
    assert isinstance(j["symptoms"], list)


def test_symptoms_analyze_unauthorized(client, mock_ner):
    r = client.post("/api/symptoms/analyze", json={"text": "I have fever"})
    assert r.status_code == 401
    j = r.json()
    assert j["code"] == "UNAUTHORIZED"
    assert "trace_id" in j

