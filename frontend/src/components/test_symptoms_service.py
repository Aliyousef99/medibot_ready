# backend/tests/test_symptoms_service.py
import pytest
from unittest.mock import patch, MagicMock

from backend.services import symptoms as symptoms_service
from backend.schemas.symptoms import SymptomSpan


@pytest.fixture
def mock_ner_pipeline():
    """Mocks the Hugging Face NER pipeline."""
    with patch('backend.services.symptoms.get_ner_pipeline') as mock_get_ner:
        mock_pipeline = MagicMock()
        # Simulate BioBERT-style output
        mock_pipeline.return_value = [
            {'entity_group': 'SIGN', 'score': 0.98, 'word': 'fever', 'start': 12, 'end': 17},
            {'entity_group': 'SYMPTOM', 'score': 0.95, 'word': 'headache', 'start': 22, 'end': 30},
            {'entity_group': 'SIGN', 'score': 0.89, 'word': 'chest pain', 'start': 45, 'end': 55},
            {'entity_group': 'DISEASE', 'score': 0.80, 'word': 'flu', 'start': 60, 'end': 63},
        ]
        mock_get_ner.return_value = (mock_pipeline, "mock-biobert", None)
        yield mock_pipeline


def test_extract_symptoms(mock_ner_pipeline):
    """Tests that symptoms are extracted correctly."""
    text = "Patient has a fever and a headache. Also reports chest pain and flu."
    entities, model, warning = symptoms_service.extract_symptoms(text)
    assert len(entities) == 4
    assert model == "mock-biobert"
    assert entities[0]['word'] == 'fever'


def test_detect_negation():
    """Tests negation detection logic."""
    text = "Patient reports no cough, but has a fever. Denies shortness of breath."
    spans = [
        SymptomSpan(text="cough", start=15, end=20, score=0.9, label="SYMPTOM"),
        SymptomSpan(text="fever", start=31, end=36, score=0.9, label="SIGN"),
        SymptomSpan(text="shortness of breath", start=45, end=64, score=0.9, label="SYMPTOM"),
    ]
    result = symptoms_service.detect_negation(text, spans)
    assert result[0].negated is True
    assert result[1].negated is False
    assert result[2].negated is True


def test_classify_urgency():
    """Tests urgency classification logic."""
    urgent_symptoms = [SymptomSpan(text="severe chest pain", start=0, end=17, score=0.99, label="SYMPTOM")]
    normal_symptoms = [SymptomSpan(text="mild headache", start=0, end=13, score=0.9, label="SYMPTOM")]
    negated_urgent = [SymptomSpan(text="chest pain", start=0, end=10, score=0.99, label="SYMPTOM", negated=True)]

    assert symptoms_service.classify_urgency(urgent_symptoms, None) == "urgent"
    assert symptoms_service.classify_urgency(normal_symptoms, None) == "normal"
    assert symptoms_service.classify_urgency(negated_urgent, None) == "normal"


def test_analyze_pipeline(mock_ner_pipeline):
    """Tests the full analysis pipeline orchestration."""
    text = "Patient has a fever and a headache. Denies chest pain."
    result = symptoms_service.analyze(text, None)
    assert result['urgency'] == 'normal'
    assert len(result['symptoms']) == 3
    assert result['symptoms'][0]['text'] == 'fever'
    assert result['symptoms'][0]['negated'] is False
    assert result['symptoms'][2]['text'] == 'chest pain'
    assert result['symptoms'][2]['negated'] is True
    assert "Analysis complete" in result['summary']