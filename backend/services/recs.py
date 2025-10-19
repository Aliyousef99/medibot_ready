import yaml
from pathlib import Path
import os
try:
    from . import gemini  # may fail if google-generativeai is not installed
except Exception:
    class _GeminiStub:
        @staticmethod
        def rewrite_patient_copy(actions):
            return None
    gemini = _GeminiStub()  # type: ignore

CONFIG_PATH = Path(__file__).parent.parent / "config" / "clinical_rules.yaml"

def load_rules():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

RULES = load_rules()

def score_risk(labs: list = None, symptoms: list = None, patient_info: dict = None):
    """
    Evaluates clinical risk based on lab results, symptoms, and patient info.
    """
    labs = labs or []
    symptoms = symptoms or []
    patient_info = patient_info or {}

    # Critical flags
    for lab in labs:
        if lab.get("name") == "glucose" and lab.get("value") > RULES["glucose"]["very_high_mg_dl"]:
            return "high"
    for symptom in symptoms:
        if any(flag in symptom.get("description", "").lower() for flag in RULES["symptoms"]["red_flags"]):
            return "high"

    # Moderate flags
    is_obese = patient_info.get("bmi", 0) > RULES["obesity"]["bmi_threshold"]
    elevated_lipids = False
    for lab in labs:
        if lab.get("name") == "ldl" and lab.get("value") > RULES["lipids"]["ldl"]["elevated_mg_dl"]:
            elevated_lipids = True
        if lab.get("name") == "hdl" and lab.get("value") < RULES["lipids"]["hdl"]["low_mg_dl"]:
            elevated_lipids = True

    if is_obese and elevated_lipids:
        return "moderate"

    return "low"

def get_recommendations(risk_tier: str, labs: list = None, symptoms: list = None):
    """
    Generates recommendations based on risk tier and specific findings.
    """
    labs = labs or []
    symptoms = symptoms or []
    
    base_recs = {
        "high": [
            "Urgent: Please consult a healthcare provider immediately. Your results indicate a potentially serious condition."
        ],
        "moderate": [
            "Your results indicate a moderate health risk. We recommend scheduling a consultation with your doctor to discuss these results.",
            "Lifestyle modifications including diet and exercise are encouraged."
        ],
        "low": [
            "Your results are within the normal range. Continue to maintain a healthy lifestyle.",
            "Regular check-ups are recommended."
        ]
    }

    actions = base_recs.get(risk_tier, [])

    # Lab-specific advice
    for lab in labs:
        if lab.get("name") == "ldl" and lab.get("value") > RULES["lipids"]["ldl"]["elevated_mg_dl"]:
            actions.append("Your LDL cholesterol is high. Focus on a diet low in saturated fats and consider consulting a nutritionist.")
        if lab.get("name") == "hdl" and lab.get("value") < RULES["lipids"]["hdl"]["low_mg_dl"]:
            actions.append("Your HDL cholesterol is low. Regular exercise can help to increase it.")
        if lab.get("name") == "glucose" and lab.get("value") > RULES["glucose"]["very_high_mg_dl"]:
            actions.append("Your glucose level is very high. This requires immediate medical attention.")

    # Symptom-specific advice
    for symptom in symptoms:
        if any(flag in symptom.get("description", "").lower() for flag in RULES["symptoms"]["red_flags"]):
            actions.append(f"The symptom '{symptom.get('description')}' is a red flag. Please seek medical advice promptly.")

    return {"risk_tier": risk_tier, "actions": list(dict.fromkeys(actions))} # Remove duplicates

def render_patient_copy(actions: list[str]) -> tuple[str, bool]:
    """
    Produces patient-friendly copy for a set of actions.
    Uses Gemini if available, otherwise falls back to a simple template.
    Returns (copy, llm_used_boolean).
    """
    if os.getenv("GEMINI_API_KEY"):
        rewritten_text = gemini.rewrite_patient_copy(actions)
        if rewritten_text:
            return rewritten_text, True

    # Fallback template
    template = (
        "Here are your health recommendations:\n\n" +
        "\n".join(f"- {a}" for a in actions) +
        "\n\nPlease discuss these with your healthcare provider."
    )
    return template, False
