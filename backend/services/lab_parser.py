"""Lab report text parsing helpers."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from backend.services.ner import detect_medical_entities
from backend.services.reference_ranges import apply_universal_reference_ranges

# Patterns for common lab tests when table parsing misses them.
TEST_PATTERNS = [
    (
        "Hemoglobin",
        r"(?:\bhemoglobin\b|\bhb\b)\s*[:\-]?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>g/?d[il]|g\s?per\s?d[il])?",
    ),
    (
        "WBC",
        r"(?:\bwbc\b|white\s+blood\s+cells?)\s*[:\-]?\s*(?P<value>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(?P<unit>(?:/uL|/ul|x?10\^3/?uL|x?10\^9/?L)?)",
    ),
    (
        "Platelets",
        r"(?:\bplatelets?\b|\bplt\b)\s*[:\-]?\s*(?P<value>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(?P<unit>(?:/uL|/ul|x?10\^3/?uL|x?10\^9/?L)?)",
    ),
    (
        "Creatinine",
        r"(?:\bcreatinine\b|\bcreat\b)\s*[:\-]?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mg/?d[il]|micromol/?l|umol/?l|mg/?dL)",
    ),
    (
        "Glucose",
        r"(?:\bglucose\b|fasting\s+glucose|\bfbs\b)\s*[:\-]?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mg/?d[il]|mmol/?l|mg/?dL)",
    ),
    (
        "ALT",
        r"(?:\balt\b|\bsgpt\b)\s*[:\-]?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>u/?l|U/?L)",
    ),
    (
        "AST",
        r"(?:\bast\b|\bsgot\b)\s*[:\-]?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>u/?l|U/?L)",
    ),
    (
        "HDL",
        r"(?:\bhdl\b)\s*[:\-]?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mg/?d[il]|mg/?dL)",
    ),
    (
        "LDL",
        r"(?:\bldl\b)\s*[:\-]?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mg/?d[il]|mg/?dL)",
    ),
]

STATUS_WORDS = {
    "high": [" high ", " elevated ", " above ", " high)", "(high", "high,"],
    "low": [" low ", " decreased ", " below ", " low)", "(low", "low,"],
    "normal": [" normal ", " within range ", " in range "],
}

CONDITION_WORDS = [
    "anemia",
    "infection",
    "inflammation",
    "diabetes",
    "kidney",
    "liver",
    "thyroid",
    "cholesterol",
    "triglycerides",
    "cardiac",
    "covid",
    "vitamin d",
    "iron",
    "b12",
    "urinary",
    "pregnancy",
    "hepatitis",
]


def normalize_unit_text(unit: str) -> str:
    if not unit:
        return ""
    cleaned = unit.replace(" ", "")
    cleaned = cleaned.replace("/uL", "/uL").replace("/ul", "/uL").replace("/ULL", "/uL")
    cleaned = cleaned.replace("mg/dl", "mg/dL").replace("g/dl", "g/dL")
    cleaned = cleaned.replace("U/l", "U/L").replace("u/l", "U/L")
    cleaned = cleaned.replace("gperdL", "g/dL").replace("gperdl", "g/dL")
    cleaned = cleaned.replace("micromol", "umol")
    return cleaned


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def detect_status(window: str) -> Optional[str]:
    window_norm = f" {normalize_text(window)} "
    for status, words in STATUS_WORDS.items():
        if any(word in window_norm for word in words):
            return status
    return None


def detect_conditions(text: str) -> List[str]:
    low = normalize_text(text)
    return sorted({word for word in CONDITION_WORDS if word in low})


RANGE_BRACKET = re.compile(r"\[(?P<body>[^\]]+)\]")
NUM = r"\d+(?:\.\d+)?"
NAME_UNIT_IN_PARENS = re.compile(r"^(?P<name>.+?)\((?P<unit>[^)]+)\)\s*$", re.IGNORECASE)
ROW_PATTERN = re.compile(
    r"""^
    (?P<proc>[A-Za-z][A-Za-z0-9\s/_\-]+(?:\([^)]+\))?)
    \s+
    (?P<valtok>[-+]?""" + NUM + r"""(?:[HhLl](?:\.)?)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:[HhLl](?:\.)?)?)
    \s*\.?
    \s+
    (?P<unit>[^\[\]\s]+(?:/[^\[\]\s]+)?)?
    (?P<rest>.*)$
    """,
    re.VERBOSE,
)


def clean_value_token(token: str) -> str:
    text = token.strip()
    text = text.replace("*", "")
    text = re.sub(r"(?i)\b(" + NUM + r")[HhLl](?:\.)?", r"\1", text)
    text = re.sub(r"[.,]+$", "", text)
    match = re.match(r"^\s*(" + NUM + ")", text)
    return match.group(1) if match else ""


def detect_flag(token_or_line: str) -> Optional[str]:
    lowered = token_or_line.lower()
    if re.search(r"\bhigh\b|\(high\)|(^|\s|\*)h(\b|\s|\*|$)", lowered):
        return "high"
    if re.search(r"\blow\b|\(low\)|(^|\s|\*)l(\b|\s|\*|$)", lowered):
        return "low"
    return None


def _looks_like_flag_token(token: str) -> bool:
    t = token.strip().lower().replace(".", "")
    return t in {"*", "h", "l", "*h", "*l", "h*", "l*"}


def _extract_unit_from_rest(rest: str) -> str:
    if not rest:
        return ""
    cleaned = rest.replace("(", " ").replace(")", " ")
    for raw in cleaned.split():
        tok = raw.strip()
        if not tok:
            continue
        if _looks_like_flag_token(tok):
            continue
        if re.search(r"[A-Za-z/]", tok):
            return normalize_unit_text(tok)
    return ""


def parse_bracket_range(line: str) -> Optional[Dict[str, Any]]:
    match = RANGE_BRACKET.search(line)
    if not match:
        return None
    body = match.group("body").replace(" ", "")
    op_match = re.match(r"^(<=|<|>=|>)(" + NUM + r")(.*)$", body)
    if op_match:
        op, val, unit_tail = op_match.groups()
        unit = normalize_unit_text(unit_tail)
        value = float(val)
        mapping = {"<=": "lte", "<": "lt", ">=": "gte", ">": "gt"}
        return {"kind": mapping[op], "v": value, "unit": unit}
    between_match = re.match(r"^(" + NUM + r")-(" + NUM + r")(.*)$", body)
    if between_match:
        lo, hi, unit_tail = between_match.groups()
        unit = normalize_unit_text(unit_tail)
        return {"kind": "between", "lo": float(lo), "hi": float(hi), "unit": unit}
    return None


INLINE_RANGE = re.compile(r"(?P<lo>" + NUM + r")\s*-\s*(?P<hi>" + NUM + r")\s*(?P<unit>[A-Za-z/%^0-9]+)?")
INLINE_THRESH = re.compile(r"(?P<op><=|<|>=|>)\s*(?P<val>" + NUM + r")\s*(?P<unit>[A-Za-z/%^0-9]+)?")

def parse_inline_range(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    # normalize unicode dashes in-place
    cleaned = text.replace("\u2013", "-").replace("\u2014", "-")
    m = INLINE_THRESH.search(cleaned)
    if m:
        op = m.group("op")
        val = m.group("val")
        unit = normalize_unit_text(m.group("unit") or "")
        mapping = {"<=": "lte", "<": "lt", ">=": "gte", ">": "gt"}
        return {"kind": mapping[op], "v": float(val), "unit": unit}
    m = INLINE_RANGE.search(cleaned)
    if m:
        lo = m.group("lo")
        hi = m.group("hi")
        unit = normalize_unit_text(m.group("unit") or "")
        return {"kind": "between", "lo": float(lo), "hi": float(hi), "unit": unit}
    return None


def compare_to_range(value: float, reference: Dict[str, Any]) -> Optional[str]:
    kind = reference.get("kind") if reference else None
    if not kind:
        return None
    if kind == "lte":
        return "high" if value > reference["v"] else "normal"
    if kind == "lt":
        return "high" if value >= reference["v"] else "normal"
    if kind == "gte":
        return "low" if value < reference["v"] else "normal"
    if kind == "gt":
        return "low" if value <= reference["v"] else "normal"
    if kind == "between":
        lo, hi = reference["lo"], reference["hi"]
        if value < lo:
            return "low"
        if value > hi:
            return "high"
        return "normal"
    return None


def _format_value(val: float) -> str:
    if abs(val - round(val)) < 1e-6:
        return str(int(round(val)))
    return f"{val:.2f}".rstrip("0").rstrip(".")


def _maybe_rescale_value(value_str: str, value_num: float, reference: Dict[str, Any]) -> tuple[float, str]:
    if not reference or "." in value_str:
        return value_num, value_str
    kind = reference.get("kind")
    ref_min = reference.get("lo") if kind == "between" else None
    ref_max = reference.get("hi") if kind == "between" else None
    if kind in ("lt", "lte"):
        ref_max = reference.get("v")
    if kind in ("gt", "gte"):
        ref_min = reference.get("v")
    if not ref_max or ref_max <= 0:
        return value_num, value_str
    ratio = value_num / float(ref_max)
    if ratio < 3:
        return value_num, value_str
    for factor in (10, 100):
        cand = value_num / factor
        low = float(ref_min) * 0.5 if ref_min is not None else 0.0
        high = float(ref_max) * 1.5
        if low <= cand <= high:
            return cand, _format_value(cand)
    return value_num, value_str


def _heuristic_confidence(tests: List[Dict[str, Any]]) -> float:
    if not tests:
        return 0.0
    valid = [t for t in tests if t.get("name") and t.get("value") not in (None, "")]
    if not valid:
        return 0.0
    units = sum(1 for t in valid if t.get("unit"))
    statuses = sum(
        1 for t in valid if (t.get("status") or "").lower() in ("high", "low", "normal")
    )
    base = (len(valid) / max(1, len(tests))) * 0.7
    unit_bonus = (units / max(1, len(valid))) * 0.2
    status_bonus = (statuses / max(1, len(valid))) * 0.1
    return round(min(1.0, base + unit_bonus + status_bonus), 2)


def _has_key_fields(tests: List[Dict[str, Any]]) -> bool:
    return any(t.get("name") and t.get("value") not in (None, "") for t in tests)


VALUE_UNIT_INLINE = re.compile(
    r"(?P<value>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s*(?P<unit>[A-Za-z/%\^][A-Za-z0-9/%\^]*))?"
)


def _extract_tests_from_entities(text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not entities:
        return []
    terms = []
    for item in entities:
        term = (item.get("text") or item.get("term") or item.get("word") or "").strip()
        if term:
            terms.append(term)

    tests: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for line in text.splitlines():
        if not line.strip():
            continue
        lowered = line.lower()
        for term in terms:
            if term.lower() not in lowered:
                continue
            match = VALUE_UNIT_INLINE.search(line)
            if not match:
                continue
            value = (match.group("value") or "").replace(",", "")
            if not value:
                continue
            unit = normalize_unit_text((match.group("unit") or "").strip())
            status = detect_status(line) or "unspecified"
            key = (term.lower(), value, unit)
            if key in seen:
                continue
            seen.add(key)
            tests.append(
                {
                    "name": term,
                    "value": value,
                    "unit": unit,
                    "status": status,
                    "reference": None,
                    "source": "ner",
                }
            )
    return tests


def parse_lab_heuristics(text: str) -> Dict[str, Any]:
    tests: List[Dict[str, Any]] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines:
        line = line.replace("\u2013", "-").replace("\u2014", "-")
        match = ROW_PATTERN.match(line)
        if not match:
            continue

        proc_raw = match.group("proc").strip()
        val_tok = match.group("valtok").strip()
        unit_tok = (match.group("unit") or "").strip()
        rest = match.group("rest") or ""

        name = proc_raw
        name_unit = None
        name_unit_match = NAME_UNIT_IN_PARENS.match(proc_raw)
        if name_unit_match:
            name = name_unit_match.group("name").strip()
            name_unit = normalize_unit_text(name_unit_match.group("unit").strip())

        value_str = clean_value_token(val_tok).replace(",", "")
        if not value_str:
            continue
        try:
            value_num = float(value_str)
        except ValueError:
            continue

        if _looks_like_flag_token(unit_tok):
            rest = f"{unit_tok} {rest}".strip()
            unit_tok = ""

        unit = normalize_unit_text(unit_tok) or name_unit or ""
        if not unit:
            unit = _extract_unit_from_rest(rest)
        status = detect_flag(val_tok) or detect_flag(rest) or detect_status(rest) or "unspecified"

        reference = parse_bracket_range(line) or parse_inline_range(rest) or parse_inline_range(line)
        if reference:
            if reference.get("unit"):
                unit = normalize_unit_text(reference["unit"]) or unit
            value_num, value_str = _maybe_rescale_value(value_str, value_num, reference)
            inferred = compare_to_range(value_num, reference)
            if inferred:
                status = inferred

        canon = name
        canon_low = name.lower().replace(" ", "")
        if "cholesteroltotal" in canon_low:
            canon = "Cholesterol Total"
        elif canon_low.startswith("triglyceride"):
            canon = "Triglyceride"

        tests.append(
            {
                "name": canon,
                "value": value_str,
                "unit": unit,
                "status": status,
                "reference": reference,
            }
        )

    lowered_text = text
    lowered_norm = normalize_text(text)
    for name, pattern in TEST_PATTERNS:
        regex = re.compile(pattern, re.IGNORECASE)
        for match in regex.finditer(lowered_text):
            gd = match.groupdict()
            value = (gd.get("value") or "").strip().replace(",", "")
            if not value:
                continue
            start, end = match.span()
            window = lowered_norm[max(0, start - 30) : min(len(lowered_norm), end + 30)]
            status = detect_status(window) or "unspecified"
            unit = normalize_unit_text((gd.get("unit") or "").strip())
            tests.append(
                {
                    "name": name,
                    "value": value,
                    "unit": unit,
                    "status": status,
                    "reference": None,
                }
            )

    conditions = detect_conditions(text)

    return {
        "tests": tests,
        "conditions": conditions,
        "meta": {
            "engine": "heuristic",
            "overall_lab_confidence": _heuristic_confidence(tests),
        },
    }


def parse_lab_report(text: str) -> Dict[str, Any]:
    heuristics = parse_lab_heuristics(text)
    tests = heuristics.get("tests") or []
    confidence = heuristics.get("meta", {}).get("overall_lab_confidence", 0.0)
    needs_ner = (not _has_key_fields(tests)) or confidence < 0.5

    if not needs_ner:
        apply_universal_reference_ranges(tests)
        return heuristics

    ner_out = detect_medical_entities(text)
    entities = ner_out.get("entities") or []
    ner_tests = _extract_tests_from_entities(text, entities)

    merged: List[Dict[str, Any]] = list(tests)
    seen = {(t.get("name", "").lower(), str(t.get("value", "")), t.get("unit", "")) for t in tests}
    for candidate in ner_tests:
        key = (candidate["name"].lower(), str(candidate.get("value", "")), candidate.get("unit", ""))
        if key in seen:
            continue
        seen.add(key)
        merged.append(candidate)

    apply_universal_reference_ranges(merged)

    combined_conf = _heuristic_confidence(merged)
    meta = dict(heuristics.get("meta") or {})
    meta.update(
        {
            "engine": "heuristic+ner",
            "overall_lab_confidence": combined_conf,
            "ner_meta": ner_out.get("meta", {}),
        }
    )
    return {
        "tests": merged,
        "conditions": heuristics.get("conditions") or [],
        "entities": entities,
        "meta": meta,
    }


def parse_lab_text(text: str) -> Dict[str, Any]:
    return parse_lab_report(text)


__all__ = ["parse_lab_report", "parse_lab_text", "parse_lab_heuristics", "normalize_unit_text", "normalize_text", "detect_conditions"]
