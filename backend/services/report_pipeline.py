"""End-to-end lab report processing helpers."""
from __future__ import annotations

from typing import Any, Dict

from backend.services.glossary import enrich_entities
from backend.services.lab_parser import parse_lab_text
from backend.services.ner import detect_medical_entities
from backend.services.ocr import extract_text_from_bytes
from backend.services.storage import store_local_upload


def _detect_language(text: str) -> str:
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    ratio = ascii_chars / max(1, len(text))
    return "en" if ratio > 0.9 else "unknown"


def _generate_explanation(structured: Dict[str, Any], enriched_entities: list[dict]) -> str:
    lines: list[str] = []
    tests = structured.get("tests") or []
    if tests:
        lines.append("Key lab readings:")
        for item in tests[:5]:
            name = item.get("name", "Test")
            value = item.get("value", "?")
            unit = item.get("unit", "")
            status = item.get("status", "unspecified")
            unit_part = f" {unit}" if unit else ""
            lines.append(f"- {name}: {value}{unit_part} ({status})")
    explained = [e for e in enriched_entities if e.get("explanation")]
    if explained:
        lines.append("What the medical terms mean:")
        for item in explained:
            lines.append(f"- {item['term']}: {item['explanation']}")
    if not lines:
        lines.append("No clear clinical entities were detected. Please review the raw text with a clinician.")
    lines.append("This summary is informational and not a diagnosis.")
    return "\n".join(lines)


def process_text(text: str) -> Dict[str, Any]:
    structured = parse_lab_text(text)
    ner_out = detect_medical_entities(text)
    enriched_entities = enrich_entities(ner_out.get("entities", []))
    structured["entities"] = enriched_entities
    explanation = _generate_explanation(structured, enriched_entities)
    return {
        "raw_text": text,
        "language": _detect_language(text),
        "structured": structured,
        "entities": enriched_entities,
        "ner_meta": ner_out.get("meta", {}),
        "explanation": explanation,
    }


def process_upload(data: bytes, filename: str, content_type: str) -> Dict[str, Any]:
    stored_path, stored_name = store_local_upload(data, filename)
    text, source = extract_text_from_bytes(data, filename, content_type)
    result = process_text(text)
    result["source"] = source
    result["stored_path"] = stored_path
    result["stored_name"] = stored_name
    return result


__all__ = ["process_text", "process_upload"]
