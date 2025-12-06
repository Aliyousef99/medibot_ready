import backend.services.symptom_analysis as sa


def test_symptom_analysis_uses_model(monkeypatch):
    def fake_get():
        def fake_pipe(_text):
            return [
                {
                    "word": "chest pain",
                    "entity_group": "PROBLEM",
                    "score": 0.93,
                    "start": 0,
                    "end": 10,
                }
            ]

        return fake_pipe, "mock-model", None

    monkeypatch.setattr(sa, "_get_clinical_pipeline", fake_get)

    res = sa.analyze_text("patient reports chest pain and dizziness")
    assert res["engine"] == "transformers"
    assert "chest pain" in res["symptoms"]
    assert res["confidence"] > 0.5


def test_symptom_analysis_falls_back_to_heuristics(monkeypatch):
    monkeypatch.setattr(sa, "_get_clinical_pipeline", lambda: (None, "mock-model", "disabled"))

    res = sa.analyze_text("Feeling dizzy and weak lately")
    assert res["engine"] == "heuristic"
    assert "dizziness" in res["symptoms"]
    assert "weakness" in res["symptoms"]
    assert any("glucose" in t for t in res["possible_tests"])
