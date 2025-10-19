import re
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


UNIT_ALIASES = {
    "mmol/l": "mmol/L",
    "mg/dl": "mg/dL",
    "ng/ml": "ng/mL",
    "pg/ml": "pg/mL",
    "ug/ml": "µg/mL",
    "µg/ml": "µg/mL",
    "u/l": "U/L",
    "iu/l": "IU/L",
    "ku/l": "kU/L",
    "g/dl": "g/dL",
}


KNOWN_UNITS = tuple(set(UNIT_ALIASES.values()))


def normalize_unit(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    u = u.strip()
    low = u.lower()
    if low in UNIT_ALIASES:
        return UNIT_ALIASES[low]
    # Try normalizing separators
    low = low.replace("\\/", "/").replace("//", "/")
    return UNIT_ALIASES.get(low, u)


def html_unescape(s: str) -> str:
    # Minimal replacements as requested
    repl = {
        "&lt;=": "<=",
        "&gt;=": ">=",
        "&le;": "<=",
        "&ge;": ">=",
        "&lt;": "<",
        "&gt;": ">",
        "&nbsp;": " ",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def collapse_dots_and_spaces(s: str) -> str:
    # Collapse sequences of 2+ dots to a single dot
    s = re.sub(r"\.\.+", ".", s)
    # Remove trailing dot after a numeric token like "4.7." -> "4.7"
    s = re.sub(r"(?<=\d)\.(?=\s|$|\])", "", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def standardize_decimals(s: str) -> str:
    # Convert decimal comma to dot only for patterns like 5,2 (not thousands)
    s = re.sub(r"(\d),(?=\d{1,2}(?:\D|$))", r"\1.", s)
    return s


def preclean(text: str) -> str:
    # Keep line boundaries; perform minimal global transforms here.
    text = html_unescape(text)
    text = standardize_decimals(text)

    # Normalize multiplication sign and scientific notation with unicode superscripts
    try:
        # Replace unicode multiplication with ASCII 'x'
        text = text.replace("\u00D7", "x")  # × -> x
        # Superscript digits mapping
        sup_map = str.maketrans({
            "\u2070": "0",  # ⁰
            "\u00B9": "1",  # ¹
            "\u00B2": "2",  # ²
            "\u00B3": "3",  # ³
            "\u2074": "4",  # ⁴
            "\u2075": "5",  # ⁵
            "\u2076": "6",  # ⁶
            "\u2077": "7",  # ⁷
            "\u2078": "8",  # ⁸
            "\u2079": "9",  # ⁹
        })
        # Convert occurrences like 10⁹ or 10⁶ to 10^9 / 10^6
        import re as _re
        def _repl_pow(m):
            digits = m.group(1).translate(sup_map)
            return "10^" + digits
        text = _re.sub(r"10([\u2070\u00B9\u00B2\u00B3\u2074-\u2079]+)", _repl_pow, text)
    except Exception:
        pass
    return text


def split_lines(text: str) -> List[str]:
    # Split on newline boundaries, then collapse spaces/dots per-line
    parts = re.split(r"[\r\n]+", text)
    cleaned = []
    for p in parts:
        p2 = p.strip()
        if not p2:
            continue
        p2 = collapse_dots_and_spaces(p2)
        if p2:
            cleaned.append(p2)
    return cleaned


def camel_to_spaces(name: str) -> str:
    # Insert space before capital letters in camel case words (e.g., CholesterolTotal)
    name = re.sub(r"(?<=[a-z])([A-Z])", r" \1", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def canonicalize_name(name: str) -> str:
    base = name
    # Remove unit parentheses from the end of name if present
    base = re.sub(r"\([^)]*\)$", "", base).strip()
    # Normalize separators
    base = base.replace("_", " ").replace("-", " ")
    # Expand camel case
    base = camel_to_spaces(base)
    # Build slug for mapping
    slug = re.sub(r"[^a-z0-9]", "", base.lower())
    mapping = {
        "cholesteroltotal": "Total Cholesterol",
        "totalcholesterol": "Total Cholesterol",
        "triglyceride": "Triglycerides",
        "triglycerides": "Triglycerides",
    }
    if slug in mapping:
        return mapping[slug]
    # Title-case words otherwise
    return " ".join(w.capitalize() for w in re.split(r"\s+", base) if w)


def is_non_analyte_heading(name: str) -> bool:
    """Heuristic to drop legend/threshold headings like 'Desirable', 'Normal', 'Borderline High', 'High:>' etc.
    This prevents reference legend rows from being shown as analyte results.
    """
    low = (name or "").strip().lower()
    if not low:
        return True
    # early reject if starts with known status headings
    if re.match(r"^(desirable|normal|borderline\s*high?|very\s*high|high:?|low:?)(\b|:|\s)", low):
        return True
    # generic reference words without an analyte
    slug = re.sub(r"[^a-z0-9]+", "", low)
    if slug in {"desirable", "normal", "borderline", "borderlinehigh", "veryhigh", "high", "low", "reference", "ref"}:
        return True
    return False


def extract_bracket_ref(line: str) -> Tuple[str, Optional[str]]:
    m = re.search(r"\[(.*?)\]", line)
    if not m:
        return line, None
    ref = m.group(1).strip()
    # Normalize en/em dashes used in ranges to a standard hyphen for downstream parsing
    try:
        ref = ref.replace("\u2013", "-").replace("\u2014", "-")
    except Exception:
        pass
    # Remove that portion from the line
    new_line = (line[:m.start()] + line[m.end():]).strip()
    return new_line, ref


def parse_ref_range(ref: Optional[str]) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    if not ref:
        return None, None, None
    s = ref.strip()
    # Normalize dashes
    s = s.replace("–", "-").replace("—", "-")
    # <= or >= thresholds
    m = re.match(r"^(<=|<|>=|>)\s*(\d+(?:\.\d+)?)$", s)
    if m:
        op, num = m.group(1), float(m.group(2))
        if op in ("<=", "<"):
            return None, num, op
        else:
            return num, None, op
    # Range a-b
    m = re.match(r"^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)$", s)
    if m:
        return float(m.group(1)), float(m.group(2)), None
    # Single number (treat as max by default)
    m = re.match(r"^(\d+(?:\.\d+)?)$", s)
    if m:
        return None, float(m.group(1)), "<="
    return None, None, None


def find_value_and_flag(line: str) -> Optional[Tuple[float, Optional[str], Tuple[int, int]]]:
    # Pattern: digits, optionally [H|L].fraction OR .fraction, optional trailing dot
    pat = re.compile(r"(\d+)(?:([HL])\.(\d+)|\.(\d+))?\.?", re.IGNORECASE)
    for m in pat.finditer(line):
        # Exclude numbers that are clearly part of units inside parentheses before the number
        start, end = m.span()
        # Heuristic: we expect a space before number or start of line
        if start > 0 and line[start - 1].isalnum():
            continue
        int_part = m.group(1)
        flag = m.group(2)
        frac1 = m.group(3)
        frac2 = m.group(4)
        if flag:
            val = float(f"{int_part}.{frac1}")
            return val, flag.upper(), (start, end)
        elif frac2:
            val = float(f"{int_part}.{frac2}")
            return val, None, (start, end)
        else:
            val = float(int_part)
            return val, None, (start, end)
    return None


def find_unit(after: str, name_hint: str) -> Optional[str]:
    # Look after the value first
    m = re.search(r"(mmol\s*/\s*L|mg\s*/\s*dL|ng\s*/\s*mL|pg\s*/\s*mL|µg\s*/\s*mL|ug\s*/\s*mL|U\s*/\s*L|IU\s*/\s*L|kU\s*/\s*L|g\s*/\s*dL)", after, re.IGNORECASE)
    if m:
        return normalize_unit(re.sub(r"\s+", "", m.group(1)))
    # Fallback: look inside parentheses in name
    m = re.search(r"\(([^)]*?/[^)]*?)\)$", name_hint)
    if m:
        return normalize_unit(re.sub(r"\s+", "", m.group(1)))
    return None


def extract_name(prefix: str) -> str:
    # Trim trailing punctuation and units in parentheses
    prefix = prefix.strip(" -:")
    # Remove duplicated spaces
    prefix = re.sub(r"\s+", " ", prefix)
    return prefix


def compute_status(value: float, ref_min: Optional[float], ref_max: Optional[float], flag: Optional[str], threshold_op: Optional[str], analyte_name: Optional[str]) -> Tuple[str, bool]:
    if flag:
        if flag.upper() == "H":
            return "high", True
        elif flag.upper() == "L":
            return "low", True
    if ref_min is not None and ref_max is not None:
        if value < ref_min:
            return "low", True
        if value > ref_max:
            return "high", True
        return "normal", False
    if ref_max is not None and (threshold_op in ("<=", "<") or threshold_op is None):
        # For ceiling thresholds, treat lipids as 'desirable' when within threshold; otherwise 'normal'.
        if value <= ref_max:
            nm = (analyte_name or "").lower()
            if any(k in nm for k in ("cholesterol", "triglyceride")):
                return "desirable", False
            return "normal", False
        return "high", True
    if ref_min is not None and threshold_op in (">=", ">"):
        if value >= ref_min:
            return "normal", False
        return "low", True
    return "normal", False


def parse_lab_text(text: str, received_at: Optional[str] = None) -> Dict[str, Any]:
    cleaned = preclean(text)
    lines = split_lines(cleaned)

    analytes: List[Dict[str, Any]] = []
    abnormal_items: List[Dict[str, Any]] = []
    normal_items: List[Dict[str, Any]] = []

    candidate_lines = [ln for ln in lines if re.search(r"\d", ln)]

    for line in candidate_lines:
        working, ref_str = extract_bracket_ref(line)
        ref_min, ref_max, threshold_op = parse_ref_range(ref_str)

        vf = find_value_and_flag(working)
        if not vf:
            continue
        value, flag, span = vf
        name_part = extract_name(working[: span[0]])
        unit_part = find_unit(working[span[1] :], name_part)
        if not unit_part:
            # Additional fallbacks: percent, femtoliters, and scientific counts like x10^9/L
            after = working[span[1] :]
            # Note: "%" is not a word character, so do not use word boundaries
            m_unit = re.search(r"(%|fL)", after, re.IGNORECASE)
            if m_unit:
                unit_part = m_unit.group(1)
            else:
                m_sc = re.search(r"((?:x|×)\s*10\^\d+\s*/\s*L)", after, re.IGNORECASE)
                if m_sc:
                    unit_part = re.sub(r"\s+", "", m_sc.group(1)).replace("×", "x")
        unit_part = normalize_unit(unit_part)
        # Require a recognizable unit to reduce false positives (e.g., dates)
        if not unit_part:
            continue

        if not name_part:
            # If name is missing, skip
            continue
        name = canonicalize_name(name_part)
        if is_non_analyte_heading(name):
            continue

        status, is_abnormal = compute_status(value, ref_min, ref_max, flag, threshold_op, name)

        item = {
            "name": name,
            "value": value,
            "unit": unit_part,
        }
        if ref_min is not None:
            item["ref_min"] = ref_min
        if ref_max is not None:
            item["ref_max"] = ref_max
        item["status"] = status

        analytes.append(item)
        if is_abnormal:
            abnormal_items.append(item)
        else:
            normal_items.append(item)

    # Confidence heuristic: clamp(round_to_0_05(max(0.2, min(0.95, parsed/candidate + 0.2))), 0.2, 0.95)
    parsed_lines = len(analytes)
    denom = max(1, len(candidate_lines))
    raw_conf = (parsed_lines / denom) + 0.2
    raw_conf = max(0.2, min(0.95, raw_conf))

    def _round_to_step(x: float, step: float = 0.05) -> float:
        try:
            return round(round(x / step) * step, 2)
        except Exception:
            return round(x, 2)

    confidence = _round_to_step(raw_conf, 0.05)

    presentation = {
        "abnormal": abnormal_items,
        "normal": normal_items,
        "confidence": round(confidence, 2),
    }

    structured_json = {
        "analytes": analytes,
        "received_at": received_at if received_at else None,
    }

    return {
        "presentation": presentation,
        "structured_json": structured_json,
    }


def cli():
    import sys

    if len(sys.argv) > 1:
        # Read file path
        path = sys.argv[1]
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    else:
        raw = sys.stdin.read()

    now_iso = None
    try:
        now_iso = datetime.utcnow().isoformat() + "Z"
    except Exception:
        now_iso = None

    result = parse_lab_text(raw, received_at=now_iso)
    pres = result["presentation"]

    # One-line log as requested
    log_obj = {
        "function": "lab_parse_summary",
        "parsed_count": len(result["structured_json"]["analytes"]),
        "abnormal_count": len(pres["abnormal"]),
        "overall_lab_confidence": pres["confidence"],
    }
    print(json.dumps(log_obj, ensure_ascii=False))

    # Emit presentation and structured_json JSON blocks
    print(json.dumps(result["presentation"], ensure_ascii=False, indent=2))
    print(json.dumps(result["structured_json"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
