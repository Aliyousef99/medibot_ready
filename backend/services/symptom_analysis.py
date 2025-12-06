"""
Clinical NER-powered symptom extraction with heuristic fallback.

Behavior:
- Try a small clinical NER model (BioClinicalBERT family) via transformers.
- Normalize to canonical symptom strings and map to suggested labs/tests.
- If the model is unavailable or returns nothing, fall back to the legacy
  keyword-based heuristics to keep functionality intact.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

try:  # transformers is optional; keep graceful fallback
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline  # type: ignore
except Exception:  # pragma: no cover - missing optional dep
    AutoModelForTokenClassification = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    pipeline = None  # type: ignore

logger = logging.getLogger("medibot")

CLINICAL_MODEL_ENV = "CLINICAL_NER_MODEL"
DEFAULT_CLINICAL_MODEL = "samrawal/bert-base-uncased_clinical-ner"  # compact clinical NER
CLINICAL_SYMPTOM_LABELS = {"PROBLEM", "SYMPTOM", "DISEASE", "SIGN", "OBS"}

# --- Mapping from symptoms to likely tests --------------------------------
SYMPTOM_TEST_MAP: Dict[str, List[str]] = {
    # Symptoms mapped to example lab categories (parentheses clarify panel)
    "fatigue": ["hemoglobin (CBC)"],
    "weakness": ["hemoglobin (CBC)"],
    "dizziness": ["blood sugar (glucose)"],
    "lightheadedness": ["blood sugar (glucose)"],
    "infection": ["WBC (CBC)"],
    "fever": ["WBC (CBC)"],
    "cough": ["WBC (CBC)"],
    "chest pain": ["troponin (cardiac enzymes)"],
    "shortness of breath": ["arterial blood gas"],
    "dyspnea": ["arterial blood gas"],
    "orthopnea": ["arterial blood gas"],
    "palpitations": ["ECG"],
    "edema": ["BMP (electrolytes, renal)"],
    "headache": ["ESR/CRP"],
    "nausea": ["CMP"],
    "vomiting": ["CMP"],
    "diarrhea": ["CMP"],
    "abdominal pain": ["lipase", "CMP"],
    "syncope": ["ECG", "hemoglobin (CBC)"],
    "hematuria": ["urinalysis"],
    "dysuria": ["urinalysis"],
}

# Synonym normalization to canonical keys we expose in output
CANON_SYNONYMS: Dict[str, str] = {
    "dizzy": "dizziness",
    "lightheaded": "lightheadedness",
    "light-headed": "lightheadedness",
    "light headed": "lightheadedness",
    "tired": "fatigue",
    "tiredness": "fatigue",
    "weak": "weakness",
    "sob": "shortness of breath",
    "dyspnea": "shortness of breath",
    "short of breath": "shortness of breath",
    "breathless": "shortness of breath",
    "chest tightness": "chest pain",
    "cp": "chest pain",
    "gi upset": "abdominal pain",
    "stomach pain": "abdominal pain",
    "belly pain": "abdominal pain",
    "loose stools": "diarrhea",
    "constipation": "constipation",
    "passing out": "syncope",
    "fainting": "syncope",
    "swelling": "edema",
    "ankle swelling": "edema",
    "leg swelling": "edema",
    "urine blood": "hematuria",
    "burning urination": "dysuria",
    "painful urination": "dysuria",
}


def _canonicalize(term: str) -> str:
    t = (term or "").strip().lower()
    return CANON_SYNONYMS.get(t, t)


@lru_cache(maxsize=1)
def _load_clinical_pipeline() -> Tuple[Optional[Any], str, Optional[str]]:
    """Return (pipeline_or_None, model_name, warning_or_None)."""
    if pipeline is None or AutoTokenizer is None or AutoModelForTokenClassification is None:
        return None, DEFAULT_CLINICAL_MODEL, "transformers not available"

    model_name = os.getenv(CLINICAL_MODEL_ENV, DEFAULT_CLINICAL_MODEL)
    try:
        pipe = pipeline(
            "token-classification",
            model=AutoModelForTokenClassification.from_pretrained(model_name),
            tokenizer=AutoTokenizer.from_pretrained(model_name),
            aggregation_strategy="simple",
            device=-1,  # CPU friendly
        )
        return pipe, model_name, None
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        logger.warning("clinical NER model failed to load: %s", exc)
        return None, model_name, str(exc)


def _get_clinical_pipeline() -> Tuple[Optional[Any], str, Optional[str]]:
    """Wrapper to make monkeypatching easier in tests."""
    return _load_clinical_pipeline()


def _normalize_predictions(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert HF pipeline output to a canonical symptom list with scores."""
    normalized: List[Dict[str, Any]] = []
    for item in predictions or []:
        label = (item.get("entity_group") or item.get("entity") or "").upper()
        if label not in CLINICAL_SYMPTOM_LABELS:
            continue
        text = item.get("word") or item.get("text") or ""
        if not text.strip():
            continue
        normalized.append(
            {
                "text": text,
                "canonical": _canonicalize(text),
                "score": float(item.get("score", 0.0)),
                "start": item.get("start"),
                "end": item.get("end"),
            }
        )
    return normalized


def _heuristic_candidates(text: str) -> List[Dict[str, Any]]:
    """Legacy keyword matcher used as fallback or augmentation."""
    text_low = (text or "").lower()
    patterns = [
        "light-headed", "light headed", "lightheaded", "lightheadedness",
        "dizzy", "dizziness",
        "weak", "weakness",
        "tired", "tiredness", "fatigue",
        "infection", "fever", "cough",
        "chest pain", "shortness of breath", "dyspnea", "orthopnea",
        "palpitations", "edema", "swelling",
        "headache", "nausea", "vomiting", "diarrhea", "abdominal pain",
        "stomach pain", "belly pain",
        "syncope", "fainting", "passing out",
        "hematuria", "urine blood",
        "dysuria", "burning urination", "painful urination",
    ]
    hits: List[Dict[str, Any]] = []
    for p in patterns:
        idx = text_low.find(p)
        if idx == -1:
            continue
        if _is_negated(text_low, idx):
            continue
        hits.append({
            "text": p,
            "canonical": _canonicalize(p),
            "score": 0.6 + (0.05 if _has_temporal_cue(text_low, idx + len(p)) else 0.0),
        })
    return hits


def _dedupe(symptoms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep first occurrence per canonical form."""
    seen: set[str] = set()
    unique: List[Dict[str, Any]] = []
    for item in symptoms:
        canon = item.get("canonical") or ""
        if not canon or canon in seen:
            continue
        seen.add(canon)
        unique.append(item)
    return unique


def _is_negated(text: str, start_idx: int, window: int = 32) -> bool:
    """Detect simple negation cues close to the mention."""
    if start_idx <= 0:
        return False
    left = text[max(0, start_idx - window):start_idx]
    # Reduce scope to the fragment after the last contrastive conjunction
    for conj in (" but ", " however ", " yet "):
        pos = left.rfind(conj)
        if pos != -1:
            left = left[pos + len(conj):]
    neg_terms = ("no ", "not ", "without ", "denies ", "denied ", "absence of ", "free of ")
    for nt in neg_terms:
        pos = left.rfind(nt)
        if pos != -1 and (len(left) - pos) <= window:
            return True
    return False


def _has_temporal_cue(text: str, anchor: int, window: int = 32) -> bool:
    """Look for duration cues near the mention to boost certainty."""
    seg = text[max(0, anchor - window):anchor + window]
    cues = ("for ", "x ", "× ", "since ", "for the last", "past ", "over the")
    if not any(c in seg for c in cues):
        return False
    import re  # local import to keep module import light
    patterns = [
        r"for\s+\d+\s+(day|days|wk|wks|week|weeks|month|months)",
        r"for\s+(a|one|two|three|four|five|six|seven|eight|nine|ten)\s+(day|days|week|weeks)",
        r"since\s+(yesterday|last night|last week)",
        r"past\s+\d+\s+(day|days|week|weeks)",
    ]
    return any(re.search(p, seg) for p in patterns)


def _compute_confidence(symptoms: List[Dict[str, Any]], engine: str) -> float:
    if not symptoms:
        return 0.2
    scores = [float(s.get("score", 0.5)) for s in symptoms]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    base = 0.55 if engine == "transformers" else 0.4
    conf = min(0.98, base + 0.35 * avg_score + 0.05 * min(len(symptoms), 4))
    return round(conf, 2)


def analyze_text(text: str) -> Dict[str, Any]:
    """Extract symptoms with clinical NER and map to exams, with heuristic fallback."""
    pipe, model_name, init_warning = _get_clinical_pipeline()
    engine = "transformers" if pipe else "heuristic"
    warning: Optional[str] = init_warning
    symptoms: List[Dict[str, Any]] = []

    if pipe:
        try:
            predictions = pipe(text)
            symptoms = _normalize_predictions(predictions)
            # drop negated entities if location is available
            filtered: List[Dict[str, Any]] = []
            text_low = (text or "").lower()
            for s in symptoms:
                st = s.get("start")
                if isinstance(st, int) and st >= 0 and _is_negated(text_low, st):
                    continue
                if isinstance(st, int) and st >= 0 and _has_temporal_cue(text_low, s.get("end", st)):
                    s["score"] = min(1.0, float(s.get("score", 0.5)) + 0.05)
                filtered.append(s)
            symptoms = filtered
        except Exception as exc:  # pragma: no cover - runtime dependent
            warning = f"inference failed: {exc}"
            engine = "heuristic"

    if not symptoms:  # fallback to heuristics
        symptoms = _heuristic_candidates(text)
        engine = "heuristic"

    symptoms = _dedupe(symptoms)
    symptom_names = [s["canonical"] for s in symptoms]

    # Map to tests (dedup)
    tests: List[str] = []
    tseen: set[str] = set()
    for name in symptom_names:
        for t in SYMPTOM_TEST_MAP.get(name, []):
            if t not in tseen:
                tseen.add(t)
                tests.append(t)

    confidence = _compute_confidence(symptoms, engine)

    result = {
        "symptoms": symptom_names,
        "possible_tests": tests,
        "confidence": confidence,
        "engine": engine,
        "model": model_name if engine == "transformers" else None,
    }
    if warning:
        result["warning"] = warning

    try:
        logger.info({
            "function": "analyze_text",
            "symptoms": result["symptoms"],
            "tests": result["possible_tests"],
            "confidence": result["confidence"],
            "engine": engine,
        })
    except Exception:
        pass

    return result


__all__ = ["analyze_text"]
