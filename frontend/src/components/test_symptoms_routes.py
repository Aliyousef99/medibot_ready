# backend/tests/test_symptoms_routes.py
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from unittest.mock import patch

from backend.app import app
from backend.db.session import get_db
from backend.models.user import User
from backend.auth.jwt import create_access_token


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture
def auth_headers(db: Session):
    """Fixture to create a test user and get auth headers."""
    user = db.query(User).filter(User.email == "test@example.com").first()
    if not user:
        user = User(email="test@example.com", hashed_password="password", name="Test User")
        db.add(user)
        db.commit()
        db.refresh(user)
    token = create_access_token({"sub": str(user.id), "email": user.email})
    return {"Authorization": f"Bearer {token}"}


@patch("backend.services.symptoms.analyze")
def test_analyze_symptoms_success(mock_analyze, client: TestClient, auth_headers: dict, db: Session):
    """Test successful symptom analysis."""
    mock_analyze.return_value = {
        "symptoms": [],
        "urgency": "normal",
        "summary": "Analysis complete.",
        "engine": "mock-ner"
    }

    response = client.post(
        "/api/symptoms/analyze",
        json={"text": "I have a headache"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["urgency"] == "normal"
    assert "Analysis complete" in data["summary"]
    mock_analyze.assert_called_once()


def test_analyze_symptoms_unauthorized(client: TestClient):
    """Test that the endpoint requires authentication."""
    response = client.post("/api/symptoms/analyze", json={"text": "I have a headache"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"