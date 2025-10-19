from typing import Optional, List, Dict


def recommend(profile: Optional[dict], latest_lab: Optional[dict], symptoms: List[str]) -> Dict:
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

    # De-duplicate while preserving order
    actions = list(dict.fromkeys(actions))

    return {
        "priority": priority,
        "actions": actions or ["Monitor symptoms and rest"],
        "follow_up": (
            "If symptoms persist >48h, worsen, or include red flags (fainting, chest pain), seek medical care."
        ),
        "rationale": " ".join(rationale_bits)
        or "Initial self-care suggestions based on reported symptoms.",
    }


__all__ = ["recommend"]

