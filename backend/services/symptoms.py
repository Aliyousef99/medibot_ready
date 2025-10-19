import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from backend.schemas.symptoms import (
    SymptomSpan,
    SymptomParseResult,
    SymptomParsedItem,
)
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


# ---------- Deterministic pipeline (lexicon-based) ----------
@lru_cache(maxsize=1)
def _load_lexicon() -> Dict[str, Any]:
    try:
        # resources dir at repo root
        base = Path(__file__).resolve().parents[2]
        path = base / "resources" / "symptoms_lexicon.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return data
    except Exception as e:
        logger.warning("symptoms lexicon missing; using built-in default", extra={"error": str(e)})
        return {
            "symptoms": [
                {"canonical": c, "variants": [c]} for c in [
                    "headache", "fever", "nausea", "cough", "chest pain",
                    "shortness of breath", "dizziness", "fatigue", "sore throat",
                    "abdominal pain", "diarrhea", "vomiting", "back pain", "joint pain", "rash"
                ]
            ]
        }


def _tokenize(text: str) -> List[str]:
    # simple tokenization on non-letters/digits, keep alphanumerics
    return [t for t in re.split(r"[^A-Za-z0-9']+", text.lower()) if t]


def _find_phrase_spans(text: str, phrases: List[str]) -> List[Tuple[int, int, str]]:
    low = text.lower()
    spans: List[Tuple[int, int, str]] = []
    for p in sorted(phrases, key=len, reverse=True):
        start = 0
        while True:
            i = low.find(p.lower(), start)
            if i < 0:
                break
            spans.append((i, i + len(p), p))
            start = i + len(p)
    # de-duplicate overlapping smaller matches
    spans.sort(key=lambda x: (x[0], -(x[1]-x[0])))
    filtered: List[Tuple[int, int, str]] = []
    last_end = -1
    for s in spans:
        if s[0] >= last_end:
            filtered.append(s)
            last_end = s[1]
    return filtered


def normalize(symptoms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    lex = _load_lexicon()
    canon_map: Dict[str, str] = {}
    for item in lex.get("symptoms", []):
        canonical = item.get("canonical")
        for v in item.get("variants", []) or []:
            canon_map[v.lower()] = canonical
    for s in symptoms:
        name = s.get("name", "").lower()
        s["canonical"] = canon_map.get(name, name)
    return symptoms


def extract_symptoms(text: str) -> List[SymptomSpan]:
    # Prefer deterministic lexicon; still provide BioBERT path for backward compat
    spans: List[SymptomSpan] = []
    # build phrase list from lexicon variants
    lex = _load_lexicon()
    phrases: List[str] = []
    for item in lex.get("symptoms", []):
        phrases.extend([v.lower() for v in (item.get("variants") or [])])
    for start, end, phrase in _find_phrase_spans(text, phrases):
        spans.append(
            SymptomSpan(text=text[start:end], start=start, end=end, score=0.8, label="SYMPTOM")
        )
    if spans:
        return spans
    # fallback to previous NER if nothing found
    pipe, _model, _warn = get_ner_pipeline()
    if pipe is None:
        return []
    try:
        raw = pipe(text)
        for ent in format_entities(raw):
            label = (ent.get("label") or "").upper()
            if label in SYMPTOM_LABELS:
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
        return spans
    return spans


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


def _window_tokens(tokens: List[str], idx: int, window: int = 5) -> List[str]:
    lo = max(0, idx - window)
    hi = min(len(tokens), idx + window + 1)
    return tokens[lo:hi]


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
    """Mark span.negated if a negation token is within a 5-token window."""
    tokens = _tokenize(text)
    for span in spans:
        # approximate index of symptom start token
        prefix = _tokenize(text[: span.start])
        idx = len(prefix)
        window = _window_tokens(tokens, idx, 5)
        if any(n in window for n in ["no", "not", "denies", "without"]):
            span.negated = True
    return spans


def _parse_onset_duration_severity(text: str, span: SymptomSpan) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    window = text[max(0, span.start - 50): min(len(text), span.end + 50)]
    wlow = window.lower()
    # onset keywords
    onset = None
    m = re.search(r"since\s+(yesterday|today|last night|this morning|\b\w+day\b)", wlow)
    if m:
        onset = m.group(0)
    # duration patterns
    duration = None
    # e.g., "for two days", "for 2 days", "2d" near symptom
    m2 = re.search(r"for\s+((?:\d+|one|two|three|four|five|six|seven|eight|nine|ten))\s+(day|days|week|weeks|hour|hours)", wlow)
    if m2:
        num = m2.group(1)
        unit = m2.group(2)
        nums = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
        duration = f"{nums.get(num, num)} {unit}"
    else:
        m3 = re.search(r"\b(\d+)\s*(d|day|days|w|week|weeks|h|hour|hours)\b", wlow)
        if m3:
            n = m3.group(1)
            unit = m3.group(2)
            mp = {"d": "days", "w": "weeks", "h": "hours"}
            duration = f"{n} {mp.get(unit, unit)}"
    # severity
    severity = None
    for sev in ("mild", "moderate", "severe", "worse", "worsening"):
        if sev in wlow:
            severity = sev
            break
    return onset, duration, severity


def score_confidence(spans: List[SymptomSpan]) -> Tuple[List[float], float]:
    per: List[float] = []
    pos = sum(1 for s in spans if not s.negated)
    neg = sum(1 for s in spans if s.negated)
    for s in spans:
        base = 0.6 if not s.negated else 0.3
        per.append(max(0.0, min(1.0, base)))
    overall = 0.0
    if spans:
        overall = sum(per) / len(per)
    # adjust overall down if negative outweighs positive
    if pos + neg > 0 and neg > pos:
        overall *= 0.8
    return per, max(0.0, min(1.0, overall))


def summarize_to_json(text: str) -> SymptomParseResult:
    spans = extract_symptoms(text)
    spans = detect_negation(text, spans)
    per_scores, overall = score_confidence(spans)
    items: List[SymptomParsedItem] = []
    # normalization map by text
    normd: List[Dict[str, Any]] = normalize([{ "name": s.text } for s in spans])
    for i, s in enumerate(spans):
        onset, duration, severity = _parse_onset_duration_severity(text, s)
        items.append(
            SymptomParsedItem(
                name=s.text,
                canonical=normd[i].get("canonical") or s.text.lower(),
                negated=s.negated,
                onset=onset,
                duration=duration,
                severity=severity,
                confidence=per_scores[i] if i < len(per_scores) else 0.5,
            )
        )
    return SymptomParseResult(symptoms=items, overall_confidence=overall)


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
    # Use deterministic pipeline primarily, with engine label 'lexicon'
    symptom_spans = extract_symptoms(text)
    engine = "lexicon"

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
