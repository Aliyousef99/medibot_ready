"""
Pure, lightweight symptom analysis heuristics and mapping.

Intentional design:
- No external NLP dependencies; purely string/heuristics based.
- Easy to swap later with spaCy/HF while preserving the return shape.
- Provides a simple confidence score to indicate heuristic certainty.

API:
    analyze_text(text: str) -> {
        "symptoms": ["dizziness", "weakness"],
        "possible_tests": ["blood sugar (glucose)", "hemoglobin (CBC)"],
        "confidence": 0.78,
    }
"""
from __future__ import annotations

from typing import Dict, List, Any
import logging

logger = logging.getLogger("medibot")


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
}

# Synonym normalization to canonical keys we expose in output
CANON_SYNONYMS: Dict[str, str] = {
    "dizzy": "dizziness",
    "lightheaded": "lightheadedness",
    "light-headed": "lightheadedness",
    "tired": "fatigue",
    "tiredness": "fatigue",
    "weak": "weakness",
}


def _canonicalize(term: str) -> str:
    t = term.strip().lower()
    return CANON_SYNONYMS.get(t, t)


def analyze_text(text: str) -> Dict[str, Any]:
    """Extract symptoms heuristically and map to exams. Pure function.

    Heuristics:
    - Simple substring/word matching for common symptom variants.
    - Canonicalize via a small synonym map.
    - Confidence scaled by number of matched unique symptoms.
    """
    text_low = (text or "").lower()

    candidates: List[str] = []
    # order matters; longer forms first
    patterns = [
        "light-headed", "lightheaded", "lightheadedness",
        "dizzy", "dizziness",
        "weak", "weakness",
        "tired", "tiredness", "fatigue",
        "infection", "fever", "cough",
    ]
    for p in patterns:
        if p in text_low:
            candidates.append(p)

    # dedupe and canonicalize
    seen: set[str] = set()
    symptoms: List[str] = []
    for c in candidates:
        canon = _canonicalize(c)
        if canon not in seen:
            seen.add(canon)
            symptoms.append(canon)

    # Map to tests (dedup)
    tests: List[str] = []
    tseen: set[str] = set()
    for s in symptoms:
        for t in SYMPTOM_TEST_MAP.get(s, []):
            if t not in tseen:
                tseen.add(t)
                tests.append(t)

    # Confidence: base 0.4 + 0.2 per symptom up to 0.95
    confidence = min(0.95, 0.4 + 0.2 * len(symptoms)) if symptoms else 0.2

    result = {"symptoms": symptoms, "possible_tests": tests, "confidence": round(confidence, 2)}

    # Robust logging line as requested
    try:
        logger.info({
            "function": "analyze_text",
            "symptoms": result["symptoms"],
            "tests": result["possible_tests"],
            "confidence": result["confidence"],
        })
    except Exception:
        pass

    return result


__all__ = ["analyze_text"]
