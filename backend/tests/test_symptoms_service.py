from backend.services import symptoms as svc
from backend.schemas.profile import UserProfileOut
from backend.schemas.symptoms import SymptomSpan


def test_negation_detection(mock_ner):
    text = "No chest pain but I do have fever."
    result = svc.analyze(text, profile=None)
    spans = result["symptoms"]
    # chest pain should be negated
    chest = next(s for s in spans if s["text"].lower() == "chest pain")
    assert chest["negated"] is True
    # fever should not
    fever = next(s for s in spans if s["text"].lower() == "fever")
    assert fever["negated"] is False


def test_urgency_classification(mock_ner):
    res = svc.analyze("I have chest pain", profile=None)
    assert res["urgency"] == "urgent"
    res2 = svc.analyze("no chest pain", profile=None)
    assert res2["urgency"] == "normal"


def test_confidence_bounds(mock_ner):
    res = svc.analyze("fever", profile=None)
    for s in res["symptoms"]:
        assert 0.0 <= s["score"] <= 1.0

