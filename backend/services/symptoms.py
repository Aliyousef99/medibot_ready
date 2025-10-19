# backend/services/symptoms.py
import logging
from typing import List, Dict, Any, Optional, Tuple

from backend.schemas.symptoms import SymptomSpan
from backend.schemas.profile import UserProfileOut
from backend.services.ner import get_ner_pipeline, format_entities

logger = logging.getLogger("medibot")

# Labels from BioBERT that we consider as symptoms
SYMPTOM_LABELS = {"SIGN", "SYMPTOM", "DISEASE", "PROBLEM"}

# Keywords to detect negation in a window around the symptom
NEGATION_WORDS = {"no", "denies", "without", "not", "negative for", "absence of"}

# Keywords for urgency classification
URGENT_SYMPTOMS = {
    "chest pain", "shortness of breath", "difficulty breathing",
    "severe bleeding", "uncontrolled bleeding", "new neuro deficits",
    "neurological deficit", "sudden weakness", "sudden numbness",
    "difficulty speaking", "loss of consciousness"
}


def _extract_symptoms_with_meta(text: str) -> Tuple[List[SymptomSpan], str, Optional[str]]:
    """Extract symptoms using BioBERT when available, with heuristic fallback.

    Returns (spans, engine, warning).
    - engine is one of: "biobert", "heuristic", or an error code string
    - warning is None or a brief note about fallback
    """
    pipeline, model_name, init_warning = get_ner_pipeline()
    if pipeline is None:
        # Heuristic fallback: scan for a small set of symptom phrases
        logger.warning(f"Symptom NER pipeline unavailable: {init_warning}")
        spans = _heuristic_symptom_spans(text)
        return spans, "heuristic", init_warning

    try:
        raw = pipeline(text)
        spans: List[SymptomSpan] = []
        for ent in format_entities(raw):
            label = (ent.get("label") or "").upper()
            if label in SYMPTOM_LABELS:
                try:
                    spans.append(
                        SymptomSpan(
                            text=ent["term"],
                            start=int(ent["start"]),
                            end=int(ent["end"]),
                            score=float(ent.get("score", 0.0)),
                            label=label,
                        )
                    )
                except Exception:
                    logger.debug("Skipping malformed NER entity in extract_symptoms: %s", ent)
        return spans, "biobert", init_warning
    except Exception as e:
        logger.error(
            f"Symptom NER inference failed with model {model_name}: {e}",
            exc_info=True,
        )
        # Fallback to heuristic on runtime error as well
        spans = _heuristic_symptom_spans(text)
        return spans, "heuristic", str(e)


def extract_symptoms(text: str) -> List[SymptomSpan]:
    """Public API: return detected SymptomSpan list (CPU-only).

    Uses BioBERT NER when available; falls back to a lightweight
    heuristic phrase matcher. Confidence scores are preserved when
    provided by the model; heuristics default to 0.5.
    """
    spans, _engine, _warn = _extract_symptoms_with_meta(text)
    return spans


def _heuristic_symptom_spans(text: str) -> List[SymptomSpan]:
    """Very lightweight, CPU-only heuristic matcher for common symptoms.

    This is intentionally simple and conservative. It prioritizes
    high-signal symptom phrases used elsewhere in rules and tests.
    """
    lower = text.lower()
    phrases = set(
        list(URGENT_SYMPTOMS)
        + [
            "fever",
            "cough",
            "headache",
            "nausea",
            "vomiting",
            "diarrhea",
            "dizziness",
        ]
    )

    spans: List[SymptomSpan] = []
    for term in sorted(phrases, key=len, reverse=True):
        start = 0
        while True:
            idx = lower.find(term, start)
            if idx < 0:
                break
            spans.append(
                SymptomSpan(
                    text=text[idx : idx + len(term)],
                    start=idx,
                    end=idx + len(term),
                    score=0.5,
                    label="SYMPTOM",
                )
            )
            start = idx + len(term)
    return spans


def detect_negation(text: str, spans: List[SymptomSpan]) -> List[SymptomSpan]:
    """
    Detects negation for a list of symptom spans.
    """
    lower_text = text.lower()
    for span in spans:
        # Define a window of characters around the symptom
        window_start = max(0, span.start - 30)
        window_end = min(len(lower_text), span.end + 10)
        window = lower_text[window_start:window_end]

        if any(word in window for word in NEGATION_WORDS):
            span.negated = True
    return spans


def classify_urgency(symptoms: List[SymptomSpan], profile: Optional[UserProfileOut]) -> str:
    """
    Classifies the urgency based on detected symptoms and user profile.
    Currently uses a simple keyword-based rule set.
    """
    # Profile is passed for future enhancements (e.g., age-specific or condition-specific urgency)
    for symptom in symptoms:
        if symptom.negated:
            continue
        symptom_text_lower = symptom.text.lower()
        if any(urgent_term in symptom_text_lower for urgent_term in URGENT_SYMPTOMS):
            return "urgent"

    return "normal"


def analyze(text: str, profile: Optional[UserProfileOut]) -> Dict[str, Any]:
    """
    Orchestrates the symptom analysis pipeline.
    """
    symptom_spans, engine, warning = _extract_symptoms_with_meta(text)

    # Step 2: Detect negation
    symptoms_with_negation = detect_negation(text, symptom_spans)

    # Step 3: Classify urgency
    urgency = classify_urgency(symptoms_with_negation, profile)

    # Step 4: Generate a summary
    summary = f"Analysis complete. Detected {len(symptoms_with_negation)} potential symptom(s). Urgency level: {urgency}."
    if urgency == "urgent":
        summary += " Urgent symptoms detected. Please seek medical attention promptly."

    return {
        "symptoms": [s.model_dump() for s in symptoms_with_negation],
        "urgency": urgency,
        "summary": summary,
        "engine": engine,
    }
