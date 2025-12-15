from backend.services.recommendations import recommend


def test_age_escalates_priority_for_older_adults():
    rec = recommend({"age": 70}, None, ["cough"])
    assert rec["priority"] in ("moderate", "high")
    assert any("over 65" in a.lower() for a in rec["actions"])


def test_female_low_hemoglobin_uses_sex_specific_threshold():
    rec = recommend({"sex": "female"}, {"hemoglobin": 11.5}, ["tired"])
    assert rec["priority"] == "high"
    assert any("anemia" in a.lower() for a in rec["actions"])


def test_diabetes_adds_glucose_guidance():
    rec = recommend({"conditions": ["Diabetes"]}, None, [])
    assert any("glucose" in a.lower() for a in rec["actions"])
