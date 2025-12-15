from backend.db.session import SessionLocal
from backend.models.lab_report import LabReport


def test_recommendations_generate_with_lab_id_handles_string_values(client):
    # Seed a lab with string values so comparisons coerce safely
    db = SessionLocal()
    lab = LabReport(
        user_id="user-1",
        raw_text="Creat 1.2",
        structured_json={"tests": [{"name": "creatinine", "value": "1.2"}]},
        summary="",
    )
    db.add(lab)
    db.commit()
    db.refresh(lab)
    db.close()

    r = client.post("/api/recommendations/generate", json={"lab_id": lab.id})
    assert r.status_code == 201
    body = r.json()
    assert body["risk_tier"] in ("low", "moderate", "high")
    assert isinstance(body.get("actions"), list)
