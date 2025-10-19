"""Lab report text parsing helpers."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

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
    text = re.sub(r"(?i)\b(" + NUM + r")[HhLl](?:\.)?", r"\1", text)
    text = re.sub(r"[.,]+$", "", text)
    match = re.match(r"^\s*(" + NUM + ")", text)
    return match.group(1) if match else ""


def detect_flag(token_or_line: str) -> Optional[str]:
    lowered = token_or_line.lower()
    if re.search(r"\bhigh\b|\(high\)|\bH\b", lowered):
        return "high"
    if re.search(r"\blow\b|\(low\)|\bL\b", lowered):
        return "low"
    return None


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


def parse_lab_text(text: str) -> Dict[str, Any]:
    tests: List[Dict[str, Any]] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines:
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

        unit = normalize_unit_text(unit_tok) or name_unit or ""
        status = detect_flag(val_tok) or detect_status(rest) or "unspecified"

        reference = parse_bracket_range(line)
        if reference:
            if reference.get("unit"):
                unit = normalize_unit_text(reference["unit"]) or unit
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
        },
    }


__all__ = [
    "parse_lab_text",
    "normalize_unit_text",
    "normalize_text",
    "detect_conditions",
]
