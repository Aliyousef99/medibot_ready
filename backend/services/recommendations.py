from typing import Optional, List, Dict, Tuple
from backend.services.recs import RULES


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

def _profile_meta(profile: Optional[dict]) -> dict:
    if not profile or not isinstance(profile, dict):
        return {"age": None, "sex": None, "conditions": set(), "medications": set()}
    sex = str(profile.get("sex") or "").strip().lower() or None
    try:
        age_val = int(profile.get("age")) if profile.get("age") is not None else None
    except Exception:
        age_val = None
    conds = set((str(c) or "").strip().lower() for c in profile.get("conditions") or [])
    conds = {c for c in conds if c}
    meds = set((str(m) or "").strip().lower() for m in profile.get("medications") or [])
    meds = {m for m in meds if m}
    return {"age": age_val, "sex": sex, "conditions": conds, "medications": meds}


def recommend(profile: Optional[dict], latest_lab: Optional[dict], symptoms: List[str], raw_text: Optional[str] = None) -> Dict:
    """Rule-based initial recommendations based on symptoms and optional labs.

    - dizziness -> hydrate, check glucose, stand up slowly
    - weakness  -> consider CBC/hemoglobin, rest, nutrition
    - if labs suggest low hemoglobin, elevate priority and add clinician follow-up
    - age/sex/conditions/meds from profile raise/lower thresholds and actions
    """
    priority = "low"
    actions: List[str] = []
    rationale_bits: List[str] = []
    meta = _profile_meta(profile)

    def bump_priority(level: str):
        order = {"low": 0, "moderate": 1, "high": 2}
        nonlocal priority
        if order.get(level, 0) > order.get(priority, 0):
            priority = level

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

    # Optional: inspect latest_lab for low hemoglobin if structured (sex-aware threshold)
    hgb_threshold = 11.0
    if meta["sex"] == "female":
        hgb_threshold = 12.0
    elif meta["sex"] == "male":
        hgb_threshold = 13.0
    try:
        if latest_lab and isinstance(latest_lab, dict):
            hgb = latest_lab.get("hemoglobin") or latest_lab.get("HGB")
            if hgb is not None and float(hgb) < hgb_threshold:
                actions.append("Discuss potential anemia with a clinician")
                rationale_bits.append("Low hemoglobin can cause dizziness/weakness.")
                bump_priority("high")
    except Exception:
        # Be resilient to parsing issues; leave actions/priority as-is
        pass

    # Profile-based nudges
    if meta["age"] is not None:
        if meta["age"] >= 75:
            bump_priority("high")
            actions.append("Because you are over 75, seek care sooner for new or worsening symptoms.")
            rationale_bits.append("Older age increases risk from acute symptoms.")
        elif meta["age"] >= 65:
            bump_priority("moderate")
            actions.append("Because you are over 65, monitor hydration and balance, and seek care promptly if symptoms persist.")
            rationale_bits.append("Older adults benefit from closer follow-up for new symptoms.")

    conds = meta["conditions"]
    meds = meta["medications"]
    if any("pregnan" in c for c in conds):
        bump_priority("moderate")
        actions.append("If pregnant, coordinate with your obstetric provider before making changes.")
    if any("diab" in c for c in conds):
        actions.append("Track blood sugars/glucose closely and note any hypoglycemia or hyperglycemia symptoms.")
        rationale_bits.append("Diabetes can worsen dizziness or infection risk.")
        bump_priority("moderate")
    if any(c in conds for c in ["hypertension", "htn"]) and ("chest pain" in s or "shortness of breath" in s):
        bump_priority("high")
        actions.append("Hypertension with chest symptoms warrants prompt medical review.")
    if any("heart failure" in c or "cardiac" in c or "coronary" in c or "cad" == c for c in conds):
        bump_priority("high" if priority == "moderate" else "moderate")
        actions.append("Monitor for chest pain, shortness of breath, or swelling; seek urgent care if present.")
    if any("kidney" in c or "ckd" in c for c in conds):
        bump_priority("moderate")
        actions.append("Stay hydrated and avoid NSAIDs unless advised by your clinician due to kidney history.")
    if meds and any(m for m in meds if any(tag in m for tag in ["prednisone", "steroid", "immunosuppress", "chemo"])):
        bump_priority("high" if priority == "moderate" else "moderate")
        actions.append("You may be immunosuppressed; monitor for fever or infection and seek care early.")
    if len(meds) >= 3:
        actions.append("Review medications with your clinician to rule out side effects.")

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


def lab_triage(structured_lab: Optional[dict], profile: Optional[dict] = None) -> Dict:
    """Inspect structured labs and return a triage with level and reasons.

    Expected structured_lab shape includes a list under 'tests' where each test may have:
      - name, value (float), unit, abnormal ('high'|'low'|'normal'|'unknown'), ref_min/ref_max.
    """
    if not structured_lab or not isinstance(structured_lab, dict):
        return {"level": "low", "reasons": [], "suggested_window": "routine follow-up"}

    meta = _profile_meta(profile)
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
                reasons.append("Ferritin markedly elevated vs ref 10-120 ng/mL")

            # Sex-aware hemoglobin threshold
            if name in ("hemoglobin", "hgb") and val is not None:
                low_cutoff = 11.0
                if meta["sex"] == "male":
                    low_cutoff = 13.0
                elif meta["sex"] == "female":
                    low_cutoff = 12.0
                if val < low_cutoff:
                    if level == "low":
                        level = "moderate"
                    reasons.append("Hemoglobin below expected range for sex")

            # Kidney disease + elevated creatinine
            if "kidney" in meta["conditions"] and name in ("creatinine",) and val is not None:
                if val >= 2.0:
                    level = "high"
                    reasons.append("Creatinine elevated with known kidney condition")

            # BP crisis
            if name in ("sbp", "systolic") and val is not None and val >= RULES["blood_pressure"]["systolic_mm_hg"]["crisis"]:
                level = "high"
                reasons.append("Systolic BP in crisis range")
            if name in ("dbp", "diastolic") and val is not None and val >= RULES["blood_pressure"]["diastolic_mm_hg"]["crisis"]:
                level = "high"
                reasons.append("Diastolic BP in crisis range")

            # Renal
            if name == "egfr" and val is not None:
                if val < RULES["renal"]["egfr_ml_min_1_73m2"]["severe_ckd"]:
                    level = "high"
                    reasons.append("eGFR in severe CKD range")
                elif val < RULES["renal"]["egfr_ml_min_1_73m2"]["moderate_ckd"]:
                    if level == "low":
                        level = "moderate"
                    reasons.append("eGFR reduced")
            if name == "creatinine" and val is not None and val >= RULES["renal"]["creatinine_mg_dl"]["critical"]:
                level = "high"
                reasons.append("Creatinine markedly elevated")

            # Cardiac
            if name == "troponin" and val is not None and val >= RULES["cardiac"]["troponin_ng_ml"]["critical"]:
                level = "high"
                reasons.append("Troponin critical")

            # Inflammatory
            if name == "crp" and val is not None:
                if val >= RULES["inflammatory"]["crp_mg_L"]["high"]:
                    if level != "high":
                        level = "moderate"
                    reasons.append("CRP elevated")

            # Any abnormal value elevates to at least moderate
            if abnormal in ("high", "low"):
                if level == "low":
                    level = "moderate"
                pretty_name = (t.get("name") or name).strip() or name
                if abnormal == "high":
                    reason = f"{pretty_name} above reference"
                else:
                    reason = f"{pretty_name} below reference"
                if reason not in reasons:
                    reasons.append(reason)
        except Exception:
            continue

    suggested_window = "as soon as practical" if level == "high" else "routine follow-up"
    return {"level": level, "reasons": reasons, "suggested_window": suggested_window}


__all__ = ["recommend", "red_flag_triage", "lab_triage"]
