import pytest
from fastapi.testclient import TestClient
from backend.app import app
from unittest.mock import patch
import io

client = TestClient(app)

# Fixture to reset the limiter state for each test
@pytest.fixture(autouse=True)
def reset_limiter():
    app.state.limiter.reset()

def test_upload_file_too_large():
    """
    Tests that uploading a file larger than the limit is rejected.
    """
    # 10MB limit
    large_content = b"a" * (10 * 1024 * 1024 + 1)
    response = client.post(
        "/api/extract_text",
        files={"file": ("large.txt", io.BytesIO(large_content), "text/plain")}
    )
    assert response.status_code == 413
    data = response.json()
    assert data["code"] == "PAYLOAD_TOO_LARGE"
    assert "File size exceeds the 10MB limit" in data["message"]    
    assert "trace_id" in data

def test_upload_invalid_mime_type():
    """
    Tests that uploading a file with an unsupported MIME type is rejected.
    """
    response = client.post(
        "/api/extract_text",
        files={"file": ("test.zip", b"content", "application/zip")}
    )
    assert response.status_code == 415
    data = response.json()
    assert data["code"] == "UNSUPPORTED_MEDIA_TYPE"
    assert "Unsupported file type" in data["message"]
    assert "trace_id" in data

def test_unhandled_exception_handler():
    """
    Tests the global exception handler for unhandled errors.
    """
    # To trigger an unhandled exception, we can patch a function to raise one.
    with patch("backend.app.parse_lab_text", side_effect=ValueError("A test error")):
        response = client.post("/api/parse_lab", json={"text": "some text"})
        assert response.status_code == 500
        data = response.json()
        assert data["code"] == "INTERNAL_SERVER_ERROR"
        assert "An unexpected error occurred" in data["message"]
        assert data["details"] == "A test error"
        assert "trace_id" in data

def test_http_exception_handler():
    """
    Tests that HTTPExceptions are formatted into the standard error envelope.
    """
    # The /api/list_models route throws a 400 HTTPException if GEMINI_API_KEY is not set.
    with patch.dict("os.environ", {"GEMINI_API_KEY": ""}):
        response = client.get("/api/list_models")
        assert response.status_code == 400
        data = response.json()
        assert data["code"] == "BAD_REQUEST"
        assert "No GEMINI_API_KEY set" in data["message"]
        assert "trace_id" in data
