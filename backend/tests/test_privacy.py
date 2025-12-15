from backend.models.user import UserProfile
from backend.models.lab_report import LabReport
from backend.models.recommendation import RecommendationSet
from backend.models.conversation import Conversation
from backend.models.message import Message
from backend.models.symptom_event import SymptomEvent
from backend.db.session import SessionLocal


def test_consent_roundtrip(client):
    r = client.get("/api/privacy/consent")
    assert r.status_code == 200
    body = r.json()
    assert body["consent_given"] is False

    r2 = client.post("/api/privacy/consent", json={"consent_given": True})
    assert r2.status_code == 200
    assert r2.json()["consent_given"] is True


def test_delete_all_user_data(client):
    db = SessionLocal()
    uid = "user-1"
    # Seed some data
    prof = db.query(UserProfile).filter(UserProfile.user_id == uid).first()
    if not prof:
        prof = UserProfile(user_id=uid, consent_given=True)
        db.add(prof)
    else:
        prof.consent_given = True
    lab = LabReport(user_id=uid, raw_text="hello", structured_json={}, summary="s")
    rec = RecommendationSet(user_id=uid, risk_tier="low", actions=["a"], text="t", rules_version="1.0", llm_used=False)
    conv = Conversation(user_id=uid, title="T")
    db.add_all([prof, lab, rec, conv])
    db.flush()
    msg = Message(conversation_id=conv.id, user_id=uid, role="user", content="hi")
    db.add(msg)
    db.commit()
    db.close()

    r = client.delete("/api/privacy/delete_data")
    assert r.status_code == 204

    db = SessionLocal()
    assert db.query(UserProfile).filter(UserProfile.user_id == uid).count() == 0
    assert db.query(LabReport).filter(LabReport.user_id == uid).count() == 0
    assert db.query(RecommendationSet).filter(RecommendationSet.user_id == uid).count() == 0
    assert db.query(Conversation).filter(Conversation.user_id == uid).count() == 0
    assert db.query(Message).filter(Message.user_id == uid).count() == 0
    assert db.query(SymptomEvent).filter(SymptomEvent.user_id == uid).count() == 0
    db.close()
