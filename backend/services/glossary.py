"""Dynamic glossary helpers leveraging Gemini for fallbacks."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import httpx

_STATIC_GLOSSARY: Dict[str, str] = {
    "tenosynovitis": "An inflammation of the sheath that surrounds a tendon, often causing pain or difficulty moving the affected joint.",
    "anemia": "A lower than normal level of red blood cells, which can lead to tiredness or shortness of breath.",
    "hyperlipidemia": "Higher than normal fats such as cholesterol in the blood.",
    "hypertension": "High blood pressure.",
    "diabetes": "A condition where the body has trouble controlling blood sugar levels.",
    "creatinine": "A waste product filtered by the kidneys; higher levels can mean the kidneys need attention.",
    "wbc": "White blood cells, which are part of the immune system and fight infections.",
    "platelets": "Cells that help the blood clot when you have a cut or injury.",
    "hdl": "High-density lipoprotein, sometimes called 'good' cholesterol.",
    "ldl": "Low-density lipoprotein, often called 'bad' cholesterol when levels are high.",
}

_DYNAMIC_CACHE: Dict[str, Tuple[str, str]] = {}
_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)


@lru_cache(maxsize=1)
def _gemini_api_key() -> str:
    return (os.getenv("GEMINI_API_KEY") or "").strip()


def _fetch_term_from_gemini(term: str) -> Optional[str]:
    api_key = _gemini_api_key()
    if not api_key:
        return None

    prompt = (
        "You are a medical assistant. Provide a concise, patient-friendly "
        "definition (<=40 words) for the medical term: '{term}'. Explain in plain "
        "language and avoid medical jargon when possible."
    ).format(term=term)

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                ]
            }
        ]
    }

    try:
        with httpx.Client(timeout=15) as client:
            response = client.post(
                _GEMINI_ENDPOINT,
                params={"key": api_key},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError:
        return None

    text = (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
        .strip()
    )
    return text or None


def lookup_term(term: str, *, allow_fetch: bool = True) -> Tuple[Optional[str], Optional[str]]:
    """Return (definition, source) for a term, optionally fetching via Gemini."""
    normalized = term.strip().lower()
    if not normalized:
        return None, None

    static = _STATIC_GLOSSARY.get(normalized)
    if static:
        return static, "glossary"

    if normalized in _DYNAMIC_CACHE:
        definition, source = _DYNAMIC_CACHE[normalized]
        return definition, source

    if not allow_fetch:
        return None, None

    definition = _fetch_term_from_gemini(normalized)
    if definition:
        _DYNAMIC_CACHE[normalized] = (definition, "gemini")
        return definition, "gemini"

    return None, None


def enrich_entities(entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    cache: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    enriched: List[Dict[str, str]] = []
    for item in entities:
        term = item.get("text") or item.get("term") or ""
        if not term:
            continue
        key = term.strip().lower()
        if key not in cache:
            cache[key] = lookup_term(term)
        definition, source = cache[key]
        enriched.append(
            {
                "term": term,
                "label": item.get("label") or item.get("entity"),
                "explanation": definition,
                "source": source,
            }
        )
    return enriched


def has_local_definition(term: str) -> bool:
    return term.strip().lower() in _STATIC_GLOSSARY


__all__ = ["lookup_term", "enrich_entities", "has_local_definition"]
