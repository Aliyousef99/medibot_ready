import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import uuid

from backend.app import app
from backend.db.session import Base, get_db
from backend.auth.jwt import create_access_token
from backend.models.user import User
from backend.models.recommendation import RecommendationSet

# Setup test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

@pytest.fixture(scope="function")
def test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def test_user(test_db):
    db = TestingSessionLocal()
    user_id = str(uuid.uuid4())
    user = User(id=user_id, email="test@example.com", hashed_password="password", name="Test User")
    db.add(user)
    db.commit()
    db.refresh(user)
    yield user
    db.close()

@pytest.fixture(scope="function")
def auth_headers(test_user):
    token = create_access_token(data={"sub": test_user.email})
    return {"Authorization": f"Bearer {token}"}

def test_generate_recs_high_risk(auth_headers, test_user):
    response = client.post(
        "/api/recommendations/generate",
        headers=auth_headers,
        json={"symptoms": [{"description": "severe chest pain"}]},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["risk_tier"] == "high"
    assert data["user_id"] == test_user.id

def test_generate_recs_moderate_risk(auth_headers, test_user):
    response = client.post(
        "/api/recommendations/generate",
        headers=auth_headers,
        json={
            "labs": [{"name": "ldl", "value": 180}],
            "patient_info": {"bmi": 31}
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["risk_tier"] == "moderate"
    assert data["user_id"] == test_user.id