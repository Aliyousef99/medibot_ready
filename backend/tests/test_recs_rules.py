import pytest
import os
from unittest.mock import patch
from backend.services.recs import score_risk, get_recommendations, render_patient_copy

@pytest.fixture
def high_risk_glucose():
    return {"labs": [{"name": "glucose", "value": 250}], "symptoms": []}

@pytest.fixture
def high_risk_symptom():
    return {"labs": [], "symptoms": [{"description": "I have chest pain"}]}

@pytest.fixture
def moderate_risk_obese_lipids():
    return {
        "labs": [{"name": "ldl", "value": 170}],
        "symptoms": [],
        "patient_info": {"bmi": 32}
    }

@pytest.fixture
def low_risk_normal():
    return {
        "labs": [{"name": "glucose", "value": 90}, {"name": "ldl", "value": 120}],
        "symptoms": [{"description": "feeling tired"}],
        "patient_info": {"bmi": 22}
    }

def test_score_risk_high_glucose(high_risk_glucose):
    assert score_risk(labs=high_risk_glucose["labs"]) == "high"

def test_score_risk_high_symptom(high_risk_symptom):
    assert score_risk(symptoms=high_risk_symptom["symptoms"]) == "high"

def test_score_risk_moderate(moderate_risk_obese_lipids):
    assert score_risk(
        labs=moderate_risk_obese_lipids["labs"],
        patient_info=moderate_risk_obese_lipids["patient_info"]
    ) == "moderate"

def test_score_risk_low(low_risk_normal):
    assert score_risk(
        labs=low_risk_normal["labs"],
        symptoms=low_risk_normal["symptoms"],
        patient_info=low_risk_normal["patient_info"]
    ) == "low"

def test_get_recommendations_high(high_risk_symptom):
    risk = score_risk(symptoms=high_risk_symptom["symptoms"])
    recs = get_recommendations(risk, symptoms=high_risk_symptom["symptoms"])
    assert "Urgent" in recs["actions"][0]
    assert "red flag" in recs["actions"][1]

def test_get_recommendations_moderate(moderate_risk_obese_lipids):
    risk = score_risk(
        labs=moderate_risk_obese_lipids["labs"],
        patient_info=moderate_risk_obese_lipids["patient_info"]
    )
    recs = get_recommendations(risk, labs=moderate_risk_obese_lipids["labs"])
    assert "moderate health risk" in recs["actions"][0]
    assert "LDL cholesterol is high" in recs["actions"][2]

def test_get_recommendations_low(low_risk_normal):
    risk = score_risk(
        labs=low_risk_normal["labs"],
        symptoms=low_risk_normal["symptoms"],
        patient_info=low_risk_normal["patient_info"]
    )
    recs = get_recommendations(risk)
    assert "within the normal range" in recs["actions"][0]

@patch('backend.services.recs.gemini.rewrite_patient_copy')
def test_render_patient_copy_with_gemini(mock_rewrite):
    """Test that LLM is called when API key is set and returns rewritten text."""
    actions = ["Action 1", "Action 2"]
    mock_rewrite.return_value = "This is a rewritten message."
    
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
        copy, llm_used = render_patient_copy(actions)

    mock_rewrite.assert_called_once_with(actions)
    assert llm_used is True
    assert copy == "This is a rewritten message."

@patch('backend.services.recs.gemini.rewrite_patient_copy')
def test_render_patient_copy_gemini_fallback(mock_rewrite):
    """Test fallback to template when Gemini fails."""
    actions = ["Action 1", "Action 2"]
    mock_rewrite.return_value = None # Simulate Gemini failure

    with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
        copy, llm_used = render_patient_copy(actions)

    mock_rewrite.assert_called_once_with(actions)
    assert llm_used is False
    assert "Here are your health recommendations" in copy
    assert "- Action 1" in copy

def test_render_patient_copy_no_api_key():
    """Test fallback to template when no API key is present."""
    actions = ["Action 1", "Action 2"]
    
    # Ensure GEMINI_API_KEY is not set
    with patch.dict(os.environ, {}, clear=True):
        copy, llm_used = render_patient_copy(actions)

    assert llm_used is False
    assert "Here are your health recommendations" in copy