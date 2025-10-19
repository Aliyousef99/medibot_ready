from typing import Optional, List, Dict, Tuple


RED_FLAG_KEYWORDS: List[Tuple[str, List[str]]] = [
    # canonical reason, variants
    ("syncope", [
        "fainted", "fainting", "syncope", "passed out", "pass out", "black out", "blacked out", "blackout", "lost consciousness", "collapsed"
    ]),
    ("chest pain", [
        "chest pain", "pressure in chest", "tightness in chest", "chest tightness"
    ]),
    ("shortness of breath", [
        "shortness of breath", "trouble breathing", "difficulty breathing", "breathless", "sob"
    ]),
    ("severe headache", [
        "worst headache", "severe headache", "thunderclap"
    ]),
    ("confusion", [
        "confusion", "confused", "disoriented"
    ]),
    ("neuro deficits", [
        "one-sided weakness", "one sided weakness", "hemiparesis", "weakness on one side", "facial droop", "slurred speech"
    ]),
    ("uncontrolled bleeding", [
        "uncontrolled bleeding", "heavy bleeding", "bleeding that won't stop", "bleeding won't stop"
    ]),
    ("gi bleeding", [
        "blood in stool", "bloody stool", "blood in my stool", "blood in vomit", "bloody vomit", "hematemesis", "melena"
    ]),
    ("vision loss", [
        "vision loss", "lost vision", "sudden vision loss", "can't see", "cannot see", "blindness in one eye", "amaurosis", "vision went dark"
    ]),
    ("severe neck pain", [
        "new severe neck pain", "severe neck pain", "sudden neck pain"
    ]),
]

# Concerning but non-urgent patterns map to watch level
WATCH_PATTERNS: List[Tuple[str, List[str]]] = [
    ("persistent symptoms", [
        "for days", "for weeks", "for months", "persistent", "ongoing"
    ]),
    ("orthostatic symptoms", [
        "orthostatic", "standing up makes me dizzy", "lightheaded when standing", "dizzy when standing", "near fainting", "near syncope"
    ]),
]


def red_flag_triage(symptoms: List[str], raw_text: Optional[str]) -> Dict:
    """Return triage level and reasons based on red-flag keywords found.

    - level: 'urgent' if any red flags; otherwise 'ok'.
    - reasons: list of canonical red-flag labels that matched.
    """
    text = (raw_text or "").lower()
    sset = set((x or "").lower() for x in (symptoms or []))
    reasons: List[str] = []
    for label, variants in RED_FLAG_KEYWORDS:
        if any(v in text for v in variants) or any(v in sset for v in variants):
            reasons.append(label)
    if reasons:
        return {"level": "urgent", "reasons": reasons}
    # Check watch-level patterns if not urgent
    watch_reasons: List[str] = []
    for label, variants in WATCH_PATTERNS:
        if any(v in text for v in variants) or any(v in sset for v in variants):
            watch_reasons.append(label)
    if watch_reasons:
        return {"level": "watch", "reasons": watch_reasons}
    return {"level": "ok", "reasons": []}


def recommend(profile: Optional[dict], latest_lab: Optional[dict], symptoms: List[str], raw_text: Optional[str] = None) -> Dict:
    """Rule-based initial recommendations based on symptoms and optional labs.

    - dizziness -> hydrate, check glucose, stand up slowly
    - weakness  -> consider CBC/hemoglobin, rest, nutrition
    - if labs suggest low hemoglobin, elevate priority and add clinician follow-up
    """
    priority = "low"
    actions: List[str] = []
    rationale_bits: List[str] = []

    s = set((x or "").lower() for x in (symptoms or []))
    if "dizziness" in s:
        actions += [
            "Hydrate regularly",
            "Check blood sugar (glucose)",
            "Stand up slowly / avoid sudden posture changes",
        ]
        rationale_bits.append(
            "Dizziness can relate to glucose changes, dehydration, or orthostatic effects."
        )
        priority = "moderate"
    if "weakness" in s:
        actions += [
            "Consider CBC / hemoglobin check",
            "Ensure rest and adequate nutrition",
        ]
        rationale_bits.append(
            "Weakness may relate to anemia or low energy availability."
        )
        priority = "moderate"

    # Optional: inspect latest_lab for low hemoglobin if structured
    try:
        if latest_lab and isinstance(latest_lab, dict):
            hgb = latest_lab.get("hemoglobin") or latest_lab.get("HGB")
            if hgb is not None and float(hgb) < 11.0:
                actions.append("Discuss potential anemia with a clinician")
                rationale_bits.append("Low hemoglobin can cause dizziness/weakness.")
                priority = "high"
    except Exception:
        # Be resilient to parsing issues; leave actions/priority as-is
        pass

    # Red-flag triage elevates priority and adds urgent action
    triage = red_flag_triage(symptoms or [], raw_text)
    if triage.get("level") == "urgent":
        priority = "high"
        actions = [
            "Seek medical attention promptly",
            *actions,
        ]
    elif triage.get("level") == "watch" and priority == "low":
        priority = "moderate"
    
    # De-duplicate while preserving order
    actions = list(dict.fromkeys(actions))

    return {
        "priority": priority,
        "actions": actions or ["Monitor symptoms and rest"],
        "follow_up": (
            "Seek urgent care or emergency services now."
            if triage.get("level") == "urgent"
            else "If symptoms persist >48h, worsen, or include red flags (fainting, chest pain), seek medical care."
        ),
        "rationale": " ".join(rationale_bits)
        or "Initial self-care suggestions based on reported symptoms.",
    }


def lab_triage(structured_lab: Optional[dict]) -> Dict:
    """Inspect structured labs and return a triage with level and reasons.

    Expected structured_lab shape includes a list under 'tests' where each test may have:
      - name, value (float), unit, abnormal ('high'|'low'|'normal'|'unknown'), ref_min/ref_max.
    """
    if not structured_lab or not isinstance(structured_lab, dict):
        return {"level": "low", "reasons": [], "suggested_window": "routine follow-up"}

    tests = structured_lab.get("tests") or []
    reasons: List[str] = []
    level: str = "low"

    for t in tests:
        try:
            name = str(t.get("name") or "").strip().lower()
            val = float(t.get("value")) if t.get("value") is not None else None
            unit = (t.get("unit") or "").lower()
            abnormal = (t.get("abnormal") or t.get("status") or "unknown").lower()

            # Ferritin severe elevation
            if name == "ferritin" and val is not None and unit in ("ng/ml",) and val >= 1000:
                level = "high"
                reasons.append("Ferritin markedly elevated vs ref 10–120 ng/mL")

            # Any abnormal value elevates to at least moderate
            if abnormal in ("high", "low") and level == "low":
                level = "moderate"
        except Exception:
            continue

    suggested_window = "as soon as practical" if level in ("moderate", "high") else "routine follow-up"
    return {"level": level, "reasons": reasons, "suggested_window": suggested_window}


__all__ = ["recommend", "red_flag_triage", "lab_triage"]
