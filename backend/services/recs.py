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

    def lab_val(name: str):
        for lab in labs:
            if (lab.get("name") or "").lower() == name.lower():
                return lab
        return None

    def as_float(val):
        try:
            if isinstance(val, str):
                val = val.strip()
            return float(val)
        except Exception:
            return None

    risk = "low"

    # Critical flags
    glu = lab_val("glucose")
    if glu:
        gv = as_float(glu.get("value"))
        if gv is not None and gv > RULES["glucose"]["thresholds"]["critical_mg_dl"]:
            return "high"
    troponin = lab_val("troponin")
    if troponin:
        tv = as_float(troponin.get("value"))
        if tv is not None and tv >= RULES["cardiac"]["troponin_ng_ml"]["critical"]:
            return "high"
    # Red-flag symptoms
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

    # Blood pressure stages
    sys_bp = lab_val("systolic")
    dia_bp = lab_val("diastolic")
    if sys_bp and sys_bp.get("value") is not None:
        sv = as_float(sys_bp.get("value"))
        if sv is not None:
            if sv >= RULES["blood_pressure"]["systolic_mm_hg"]["crisis"]:
                return "high"
            if sv >= RULES["blood_pressure"]["systolic_mm_hg"]["stage2"]:
                risk = "moderate"
            elif sv >= RULES["blood_pressure"]["systolic_mm_hg"]["stage1"]:
                risk = "moderate"
    if dia_bp and dia_bp.get("value") is not None:
        dv = as_float(dia_bp.get("value"))
        if dv is not None:
            if dv >= RULES["blood_pressure"]["diastolic_mm_hg"]["crisis"]:
                return "high"
            if dv >= RULES["blood_pressure"]["diastolic_mm_hg"]["stage2"]:
                risk = "moderate"
            elif dv >= RULES["blood_pressure"]["diastolic_mm_hg"]["stage1"]:
                risk = "moderate"

    # Renal function
    egfr = lab_val("egfr")
    creat = lab_val("creatinine")
    if egfr and egfr.get("value") is not None:
        ev = as_float(egfr.get("value"))
        if ev is not None:
            if ev < RULES["renal"]["egfr_ml_min_1_73m2"]["severe_ckd"]:
                return "high"
            if ev < RULES["renal"]["egfr_ml_min_1_73m2"]["mild_ckd"]:
                risk = "moderate"
    if creat:
        cv = as_float(creat.get("value"))
        if cv is not None and cv >= RULES["renal"]["creatinine_mg_dl"]["critical"]:
            return "high"

    # Inflammatory markers
    crp = lab_val("crp")
    if crp and crp.get("value") is not None:
        cv = as_float(crp.get("value"))
        if cv is not None:
            if cv >= RULES["inflammatory"]["crp_mg_L"]["high"]:
                risk = "high"
            elif cv >= RULES["inflammatory"]["crp_mg_L"]["mild"]:
                risk = "moderate"

    # Profile modifiers
    age = patient_info.get("age")
    conditions = [str(c).lower() for c in (patient_info.get("conditions") or [])]
    if isinstance(age, (int, float)):
        if age >= RULES["profile_modifiers"]["age_high_risk"] and risk == "moderate":
            risk = "high"
        elif age >= RULES["profile_modifiers"]["age_moderate_risk"] and risk == "low":
            risk = "moderate"
    if any(k in c for c in conditions for k in RULES["profile_modifiers"]["cardiac_history"]["keywords"]):
        if risk == "moderate":
            risk = "high"
        elif risk == "low":
            risk = "moderate"
    if any(k in c for c in conditions for k in RULES["profile_modifiers"]["renal_history"]["keywords"]):
        if risk == "low":
            risk = "moderate"

    return risk

def get_recommendations(risk_tier: str, labs: list = None, symptoms: list = None):
    """
    Generates recommendations based on risk tier and specific findings.
    """
    labs = labs or []
    symptoms = symptoms or []

    def as_float(val):
        try:
            if isinstance(val, str):
                val = val.strip()
            return float(val)
        except Exception:
            return None
    
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
        name = (lab.get("name") or "").lower()
        val = as_float(lab.get("value"))
        if lab.get("name") == "ldl" and val is not None and val > RULES["lipids"]["ldl"]["elevated_mg_dl"]:
            actions.append("Your LDL cholesterol is high. Focus on a diet low in saturated fats and consider consulting a nutritionist.")
        if lab.get("name") == "hdl" and val is not None and val < RULES["lipids"]["hdl"]["low_mg_dl"]:
            actions.append("Your HDL cholesterol is low. Regular exercise can help to increase it.")
        if lab.get("name") == "glucose" and val is not None and val > RULES["glucose"]["very_high_mg_dl"]:
            actions.append("Your glucose level is very high. This requires immediate medical attention.")
        if name == "egfr" and val is not None:
            if val < RULES["renal"]["egfr_ml_min_1_73m2"]["moderate_ckd"]:
                actions.append("Kidney function is reduced; arrange nephrology follow-up.")
            elif val < RULES["renal"]["egfr_ml_min_1_73m2"]["mild_ckd"]:
                actions.append("Mildly reduced kidney function; monitor hydration and avoid NSAIDs unless advised.")
        if name == "creatinine" and val is not None and val >= RULES["renal"]["creatinine_mg_dl"]["critical"]:
            actions.append("Creatinine is significantly elevated; seek urgent evaluation.")
        if name == "troponin" and val is not None and val >= RULES["cardiac"]["troponin_ng_ml"]["elevated"]:
            actions.append("Elevated troponin can indicate cardiac injury; seek emergency care if you have chest pain or shortness of breath.")
        if name == "crp" and val is not None and val >= RULES["inflammatory"]["crp_mg_L"]["high"]:
            actions.append("High CRP suggests significant inflammation; discuss with your clinician.")

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
        try:
            if hasattr(gemini, "rewrite_patient_copy"):
                rewritten_text = gemini.rewrite_patient_copy(actions)
                if rewritten_text:
                    return rewritten_text, True
        except Exception:
            # fall through to template
            pass

    # Fallback template
    template = (
        "Here are your health recommendations:\n\n" +
        "\n".join(f"- {a}" for a in actions) +
        "\n\nPlease discuss these with your healthcare provider."
    )
    return template, False
