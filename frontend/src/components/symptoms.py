# backend/services/symptoms.py
import re
import logging
from typing import List, Dict, Any, Optional

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


def extract_symptoms(text: str) -> tuple[list[dict[str, Any]], str, Optional[str]]:
    """
    Extracts symptom-related entities from text using a NER pipeline.
    """
    pipeline, model_name, init_warning = get_ner_pipeline()
    if pipeline is None:
        logger.warning(f"Symptom NER pipeline unavailable: {init_warning}")
        # In a real scenario, you might have a heuristic fallback here.
        # For now, we return empty if the primary tool is missing.
        return [], "ner-unavailable", init_warning

    try:
        entities = pipeline(text)
        return entities, model_name, init_warning
    except Exception as e:
        logger.error(f"Symptom NER inference failed with model {model_name}: {e}", exc_info=True)
        return [], "ner-inference-error", str(e)


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
    raw_entities, engine, warning = extract_symptoms(text)

    # Filter for relevant symptom labels and format into Pydantic models
    symptom_spans = []
    formatted_entities = format_entities(raw_entities)
    for entity in formatted_entities:
        if entity.get("label") in SYMPTOM_LABELS:
            try:
                symptom_spans.append(SymptomSpan(
                    text=entity["term"],
                    start=entity["start"],
                    end=entity["end"],
                    score=entity["score"],
                    label=entity["label"],
                ))
            except (KeyError, TypeError):
                logger.warning(f"Skipping malformed entity: {entity}")

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