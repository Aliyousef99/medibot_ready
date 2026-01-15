"""Extract structured lab tests from AI explanation text."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


def _parse_ref_range(ref_text: str) -> Tuple[Optional[float], Optional[float]]:
    if not ref_text:
        return None, None
    cleaned = ref_text.replace("\u2013", "-").replace("\u2014", "-")
    nums = re.findall(r"\d+(?:\.\d+)?", cleaned)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    if len(nums) == 1:
        return None, float(nums[0])
    return None, None


def _norm_unit(unit: str) -> str:
    return (unit or "").strip()


def _add_test(
    out: List[Dict[str, Any]],
    name: str,
    value: str,
    unit: str,
    ref_text: str,
    flag: str,
) -> None:
    ref_min, ref_max = _parse_ref_range(ref_text)
    status = (flag or "").strip().lower() or "unspecified"
    out.append(
        {
            "name": name.strip(),
            "value": value.strip(),
            "unit": _norm_unit(unit),
            "ref_min": ref_min,
            "ref_max": ref_max,
            "status": status,
            "abnormal": status if status in ("high", "low", "normal") else "unknown",
            "source": "ai_extraction",
        }
    )


def _extract_from_table(text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header_idx = None
    for i, ln in enumerate(lines):
        if "|" in ln and "test" in ln.lower() and "value" in ln.lower():
            header_idx = i
            break
    if header_idx is None:
        return rows
    for ln in lines[header_idx + 1:]:
        if "|" not in ln:
            break
        parts = [p.strip() for p in ln.strip("|").split("|")]
        if len(parts) < 4:
            continue
        name = parts[0]
        value = parts[1]
        unit = parts[2]
        ref = parts[3]
        flag = parts[4] if len(parts) > 4 else ""
        if not name or not value:
            continue
        _add_test(rows, name, value, unit, ref, flag)
    return rows


def _extract_from_blocks(text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pattern = re.compile(
        r"(?:\*\*|\s)?(?P<name>[A-Za-z][A-Za-z0-9 /()%\-]+)(?:\*\*)?\s*:\s*"
        r"(?:\r?\n)+\s*(?:\*\*|\s)?Value(?:\*\*)?\s*:\s*(?P<value>[\d.,]+)\s*(?P<unit>[A-Za-z/%^]+)?\s*"
        r"(?:\r?\n)+\s*(?:\*\*|\s)?Reference Range(?:\*\*)?\s*:\s*(?P<ref>[\d.\-–— ]+)\s*(?P<ref_unit>[A-Za-z/%^]+)?\s*"
        r"(?:\r?\n)+\s*(?:\*\*|\s)?Flag(?:\*\*)?\s*:\s*(?P<flag>Low|High|Normal)",
        re.IGNORECASE | re.MULTILINE,
    )
    inline_pattern = re.compile(
        r"(?:\*\*|\s)?(?P<name>[A-Za-z][A-Za-z0-9 /()%\-]+)(?:\*\*)?\s*:\s*"
        r"(?P<value>[\d.,]+)\s*(?P<unit>[A-Za-z/%^]+)?"
        r"(?:\s*\(ref\s*(?P<ref>[\d.\-–— ]+)\s*(?P<ref_unit>[A-Za-z/%^]+)?\))?",
        re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        name = m.group("name")
        value = m.group("value")
        unit = m.group("unit") or m.group("ref_unit") or ""
        ref = m.group("ref")
        flag = m.group("flag")
        _add_test(rows, name, value, unit, ref, flag)
    if rows:
        return rows
    for m in inline_pattern.finditer(text):
        name = m.group("name")
        value = m.group("value")
        unit = m.group("unit") or m.group("ref_unit") or ""
        ref = m.group("ref") or ""
        _add_test(rows, name, value, unit, ref, "")
    return rows


def extract_tests_from_ai(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    tests = _extract_from_table(text)
    if tests:
        return tests
    return _extract_from_blocks(text)


__all__ = ["extract_tests_from_ai"]
