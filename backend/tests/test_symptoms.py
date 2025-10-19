from backend.services import symptoms as svc


def test_negation_no_headache():
    text = "I have cough but no headache"
    res = svc.summarize_to_json(text)
    items = res.symptoms
    # find headache
    h = next((i for i in items if i.canonical == "headache"), None)
    assert h is not None
    assert h.negated is True


def test_duration_parsing_two_days():
    text = "Fever for two days and sore throat"
    res = svc.summarize_to_json(text)
    fev = next((i for i in res.symptoms if i.canonical == "fever"), None)
    assert fev is not None
    assert fev.duration is not None
    assert "2" in fev.duration and "day" in fev.duration


def test_confidence_threshold():
    text = "fever and cough and dizziness"
    res = svc.summarize_to_json(text)
    assert res.overall_confidence >= 0.5

