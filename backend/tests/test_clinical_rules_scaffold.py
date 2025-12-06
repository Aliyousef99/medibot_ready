from backend.services import recs


def test_rules_scaffold_has_sections():
    rules = recs.load_rules()
    for key in (
        "glucose",
        "lipids",
        "blood_pressure",
        "renal",
        "hematology",
        "inflammatory",
        "cardiac",
        "symptoms",
        "triage",
    ):
        assert key in rules, f"missing top-level section: {key}"


def test_glucose_and_lipids_keep_legacy_keys():
    rules = recs.load_rules()
    assert rules["glucose"]["very_high_mg_dl"] > 0
    assert rules["lipids"]["ldl"]["elevated_mg_dl"] > 0
    assert rules["lipids"]["hdl"]["low_mg_dl"] > 0


def test_symptom_flags_tiers_present():
    rules = recs.load_rules()
    rf = rules["symptoms"]
    assert isinstance(rf.get("red_flags"), list) and len(rf["red_flags"]) >= 3
    assert isinstance(rf.get("moderate_flags"), list) and len(rf["moderate_flags"]) >= 3
    durations = rf.get("duration_modifiers", {})
    assert durations.get("acute_days") and durations.get("chronic_days")
