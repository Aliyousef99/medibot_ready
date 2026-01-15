"""Universal reference ranges for common labs when report-specific ranges are missing."""
from __future__ import annotations

from typing import Any, Dict, Optional

RangeSpec = Dict[str, Any]


_NAME_ALIASES = {
    "whitebloodcells": "wbc",
    "whitebloodcell": "wbc",
    "wbc": "wbc",
    "leukocytes": "wbc",
    "platelets": "platelets",
    "platelet": "platelets",
    "plt": "platelets",
    "glucose": "glucose",
    "fastingglucose": "glucose",
    "fbs": "glucose",
    "bloodglucose": "glucose",
}

_UNIT_ALIASES = {
    "x10^3/ul": "x10^3/ul",
    "10^3/ul": "x10^3/ul",
    "x10^9/l": "x10^3/ul",
    "10^9/l": "x10^3/ul",
    "mg/dl": "mg/dl",
}

_UNIVERSAL_RANGES: Dict[str, RangeSpec] = {
    # Common adult ranges when labs don't provide a reference interval.
    "wbc": {"unit": "x10^3/uL", "ref_min": 4.0, "ref_max": 11.0},
    "platelets": {"unit": "x10^3/uL", "ref_min": 150.0, "ref_max": 450.0},
    "glucose": {"unit": "mg/dL", "ref_min": 70.0, "ref_max": 99.0, "label": "fasting"},
}


def _normalize_name(name: str) -> str:
    cleaned = "".join(ch for ch in (name or "").lower() if ch.isalnum())
    return _NAME_ALIASES.get(cleaned, "")


def _normalize_unit(unit: str) -> str:
    cleaned = (unit or "").strip().lower().replace(" ", "")
    return _UNIT_ALIASES.get(cleaned, cleaned)


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


def get_universal_range(test_name: str, unit: str | None) -> Optional[RangeSpec]:
    key = _normalize_name(test_name or "")
    if not key:
        return None
    spec = _UNIVERSAL_RANGES.get(key)
    if not spec:
        return None
    if unit:
        unit_norm = _normalize_unit(unit)
        spec_unit = _normalize_unit(spec.get("unit", ""))
        if unit_norm and spec_unit and unit_norm != spec_unit:
            return None
    return spec


def apply_universal_reference_ranges(tests: list[Dict[str, Any]]) -> None:
    if not tests:
        return
    for t in tests:
        if not isinstance(t, dict):
            continue
        if t.get("reference") or t.get("ref_min") is not None or t.get("ref_max") is not None:
            continue
        name = t.get("name") or ""
        unit = t.get("unit") or ""
        spec = get_universal_range(name, unit)
        if not spec:
            continue
        ref_min = spec.get("ref_min")
        ref_max = spec.get("ref_max")
        spec_unit = spec.get("unit") or unit
        if not unit and spec_unit:
            t["unit"] = spec_unit
        t["ref_min"] = ref_min
        t["ref_max"] = ref_max
        t["reference"] = {"kind": "between", "lo": ref_min, "hi": ref_max, "unit": spec_unit}
        t["reference_source"] = "universal"
        t["reference_note"] = "Reference range is general and not from this lab report."
        try:
            val = float(str(t.get("value") or "").replace(",", ""))
        except Exception:
            val = None
        if val is not None:
            inferred = compare_to_range(val, t["reference"])
            if inferred:
                if (t.get("status") or "").lower() in ("", "unspecified", "unknown"):
                    t["status"] = inferred
                if (t.get("abnormal") or "").lower() in ("", "unspecified", "unknown"):
                    t["abnormal"] = inferred

