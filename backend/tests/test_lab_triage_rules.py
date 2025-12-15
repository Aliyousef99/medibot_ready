from backend.services.recommendations import lab_triage


def test_lab_triage_critical_troponin_flags_high():
    data = {"tests": [{"name": "troponin", "value": 0.2, "unit": "ng/mL"}]}
    tri = lab_triage(data, profile=None)
    assert tri["level"] == "high"
    assert any("troponin" in r.lower() for r in tri["reasons"])


def test_lab_triage_egfr_reduced_is_moderate():
    data = {"tests": [{"name": "egfr", "value": 40, "unit": "mL/min/1.73m2"}]}
    tri = lab_triage(data, profile=None)
    assert tri["level"] in ("moderate", "high")
    assert any("egfr" in r.lower() for r in tri["reasons"])
