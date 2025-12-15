from backend.db.session import SessionLocal
from backend.models.user import User, UserProfile
from backend.models.lab_report import LabReport
from backend.models.recommendation import RecommendationSet


def test_user_profile_lab_and_recommendations_relationships():
    db = SessionLocal()
    try:
        user = User(email="rel@example.com", hashed_password="hashed")
        db.add(user)
        db.flush()

        profile = UserProfile(
            user_id=user.id,
            age=42,
            sex="female",
            conditions=["htn"],
            medications=["lisinopril"],
        )
        db.add(profile)

        lab = LabReport(
            user_id=user.id,
            raw_text="Hemoglobin 12.0",
            structured_json={"tests": [{"name": "Hemoglobin", "value": 12.0}]},
            summary="ok",
        )
        db.add(lab)

        rec = RecommendationSet(
            user_id=user.id,
            risk_tier="low",
            actions=["Monitor blood pressure at home"],
            text="Keep monitoring",
            rules_version="1.0",
            llm_used=False,
        )
        db.add(rec)

        db.commit()
        db.refresh(user)

        loaded = db.query(User).filter_by(id=user.id).one()
        assert loaded.profile.age == 42
        assert loaded.profile.sex == "female"
        assert len(loaded.lab_reports) == 1
        assert loaded.lab_reports[0].raw_text.startswith("Hemoglobin")
        assert len(loaded.recommendations) == 1
        assert loaded.recommendations[0].risk_tier == "low"
        # reverse relations
        assert loaded.lab_reports[0].user.id == user.id
        assert loaded.recommendations[0].user.id == user.id
    finally:
        db.close()
