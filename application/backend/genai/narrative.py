"""
Layer 7 — Clinical narrative (SSE) and chat stream via OpenAI.
Falls back to a deterministic template when OpenAI is unavailable.
"""
import os
import re
import json
import asyncio
from dataclasses import asdict, dataclass
from typing import AsyncGenerator

from dotenv import load_dotenv

# backend/.env provides GEMINI_API_KEY / OPENAI_API_KEY (keys are read per call by genai.llm_provider).
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

NARRATIVE_PROMPT = """You are CardiacSense AI, a clinical documentation assistant embedded in a hospital vitals monitoring system.

PATIENT VITALS:
- Heart Rate: {hr} bpm
- Blood Pressure: {bp_systolic}/{bp_diastolic} mmHg
- SpO2: {spo2}%
- Respiratory Rate: {rr} breaths/min
- Temperature: {temp}°C
- Age: {age}, Sex: {sex}

ML PREDICTION RESULTS:
Primary condition: {primary_condition} (probability: {primary_prob})
All conditions: {conditions_text}
Overall risk: {risk_label}

KEY DRIVERS (SHAP analysis):
{shap_text}

VITAL SIGN TRENDS:
{trend_text}

Write a clinical summary in exactly 3 paragraphs:
Paragraph 1: Current patient status (2-3 sentences). Plain English. State what the vitals show overall.
Paragraph 2: Key risk factors (2-3 sentences). Name specific vital values that are driving the predictions.
Paragraph 3: Recommended actions (2-3 sentences). What the care team should do now.

Tone: clinical, clear, direct. No jargon. No disclaimers. No "I am an AI" statements.
Length: 150-250 words total."""

CHAT_SYSTEM = (
    "You are CardiacSense AI, a clinical assistant. "
    "You have access to the patient's current vitals and ML predictions. "
    "Answer questions clearly, cite specific vital values when relevant, "
    "and always recommend consulting a physician for diagnosis."
)


def _build_narrative_prompt(vitals: dict, conditions: list, shap: list, trend: list, patient_context: dict) -> str:
    primary = conditions[0] if conditions else {}
    conditions_text = ", ".join(
        f"{c.get('name','?')}: {c.get('probability',0):.0%}" for c in conditions
    )
    shap_text = "\n".join(
        f"- {s.get('display_name','?')}: SHAP={s.get('shap_score',0):+.3f}" for s in shap[:4]
    )
    trend_text = "\n".join(
        f"- {t.get('display_name','?')}: {t.get('direction','stable')} (Δ{t.get('delta',0):+.1f})"
        for t in trend
    )
    risk_prob = primary.get("probability", 0)
    risk_label = "High" if risk_prob > 0.70 else "Moderate" if risk_prob > 0.40 else "Low"

    return NARRATIVE_PROMPT.format(
        hr=vitals.get("hr", "?"),
        bp_systolic=vitals.get("bp_systolic", "?"),
        bp_diastolic=vitals.get("bp_diastolic", "?"),
        spo2=vitals.get("spo2", "?"),
        rr=vitals.get("rr", "?"),
        temp=vitals.get("temp", "?"),
        age=patient_context.get("age", "unknown"),
        sex=patient_context.get("sex", "unknown"),
        primary_condition=primary.get("name", "Unknown"),
        primary_prob=f"{risk_prob:.0%}",
        conditions_text=conditions_text,
        risk_label=risk_label,
        shap_text=shap_text or "No SHAP data available.",
        trend_text=trend_text or "No trend data available.",
    )


_VITAL_RANGES = {
    # key: (label, unit, low, high)
    "hr": ("Heart rate", "bpm", 60, 100),
    "bp_systolic": ("Systolic BP", "mmHg", 90, 129),
    "bp_diastolic": ("Diastolic BP", "mmHg", 60, 79),
    "spo2": ("SpO2", "%", 95, 100),
    "rr": ("Respiratory rate", "/min", 12, 20),
    "temp": ("Temperature", "°C", 36.1, 37.5),
}


def _vital_findings(vitals: dict) -> tuple[list[str], list[str]]:
    """(abnormal, normal) sentences computed from the submitted values and normal ranges."""
    abnormal, normal = [], []
    for key, (label, unit, lo, hi) in _VITAL_RANGES.items():
        try:
            v = float(vitals.get(key))
        except (TypeError, ValueError):
            continue
        if v > hi:
            abnormal.append(f"{label} {v:g} {unit} is above the normal {lo}–{hi} {unit} (by {v - hi:g})")
        elif v < lo:
            abnormal.append(f"{label} {v:g} {unit} is below the normal {lo}–{hi} {unit} (by {lo - v:g})")
        else:
            normal.append(f"{label} {v:g} {unit}")
    return abnormal, normal


def _deterministic_narrative(vitals: dict, conditions: list, shap: list | None = None) -> str:
    """Quick-vitals summary built only from the submitted numbers, model probabilities and SHAP drivers."""
    abnormal, normal = _vital_findings(vitals or {})
    ranked = sorted(conditions or [], key=lambda c: c.get("probability", 0), reverse=True)
    out = []
    if ranked:
        top = ranked[0]
        out.append(f"The model's highest-probability finding is {top.get('name', 'unknown')} at {top.get('probability', 0):.0%}"
                   + (", followed by " + ", ".join(f"{c.get('name')} {c.get('probability', 0):.0%}" for c in ranked[1:3]) if len(ranked) > 1 else "")
                   + ".")
    if abnormal:
        out.append("Out-of-range values: " + "; ".join(abnormal) + ".")
    if normal:
        out.append("Within range: " + ", ".join(normal) + ".")
    drivers = sorted(shap or [], key=lambda s: abs(s.get("shap_score", 0)), reverse=True)[:3]
    if drivers:
        out.append("Largest model drivers: " + "; ".join(
            f"{d.get('display_name', d.get('feature'))} ({'raises' if d.get('shap_score', 0) > 0 else 'lowers'} risk, {d.get('shap_score', 0):+.2f})"
            for d in drivers) + ".")
    if abnormal:
        out.append(f"Re-check the {len(abnormal)} out-of-range value{'s' if len(abnormal) > 1 else ''} and review with the care team.")
    return " ".join(out) or "No vitals were provided to summarize."


# ═══════════════════════════════════════════════════════════════════════════
# Longitudinal context (patient profile + Visit 1 vs Visit 2) and fallbacks
# ═══════════════════════════════════════════════════════════════════════════
FALLBACK_TOKEN_DELAY = 0.015

CODE_NAMES = {
    "1dAVb": "1st-degree AV block", "RBBB": "right bundle branch block",
    "LBBB": "left bundle branch block", "SB": "sinus bradycardia (slow heart rhythm)",
    "AF": "atrial fibrillation (irregular heartbeat)", "ST": "sinus tachycardia (fast heart rhythm)",
}

PATIENT_RULES = (
    "Audience: the PATIENT. Use plain, warm, non-alarming language and short sentences. "
    "Never use bare abbreviations like AF or LBBB without the plain name. Give practical daily guidance on "
    "food (state the daily sodium target in mg) and activity (state minutes and the heart-rate ceiling). "
    "Never suggest medication or dose changes; tell the patient to talk to their care team for diagnosis "
    "or medication questions."
)
STAFF_RULES = (
    "Audience: HOSPITAL STAFF / DIETITIAN. Be concise and clinical. Report every biomarker delta with "
    "units, the severity-tier transition, present CODE-15% codes, DSP intervals (HR, QRS, PR, ST with "
    "units), the plan's clinical rule IDs, exercise contraindications and items needing sign-off."
)


@dataclass(frozen=True)
class LongitudinalContext:
    patient_id: str | None = None
    patient_profile: dict | None = None
    current_visit: dict | None = None
    previous_visit: dict | None = None
    deltas: dict | None = None
    diet_plan: dict | None = None
    is_staff_mode: bool = False
    no_visits: bool = False


def mode_instructions(staff: bool) -> str:
    return STAFF_RULES if staff else PATIENT_RULES


def _f(value, nd: int = 1) -> str:
    try:
        return f"{float(value):.{nd}f}"
    except (TypeError, ValueError):
        return "n/a"


def _ctx_parts(ctx) -> tuple[dict, dict, dict | None, dict, dict, bool]:
    if isinstance(ctx, LongitudinalContext):
        ctx = asdict(ctx)
    ctx = ctx or {}
    patient = ctx.get("patient_profile") or {}
    current = ctx.get("current_visit") or {}
    previous = ctx.get("previous_visit") or None
    deltas = ctx.get("deltas") or current.get("delta_summary") or {}
    plan = ctx.get("diet_plan") or current.get("diet_plan") or {}
    return patient, current, previous, deltas, plan, bool(ctx.get("is_staff_mode"))


def _present(visit: dict) -> list[str]:
    flags = visit.get("code15_flags") or {}
    probs = (visit.get("predictions") or {}).get("arrhythmias") or {}
    return [c for c in CODE_NAMES if flags.get(c) or (probs.get(c) or 0) >= 0.5]


def build_longitudinal_narrative(ctx: dict) -> str:
    """Deterministic Before-vs-Now summary built from the patient's exact visit numbers."""
    patient, current, previous, deltas, plan, staff = _ctx_parts(ctx)
    name = (patient.get("name") or "the patient").split(" ")[0] if not staff else (patient.get("name") or "Patient")
    pred = current.get("predictions") or {}
    risk, tier = pred.get("overall_risk_score"), pred.get("severity_tier") or "unknown"
    targets = plan.get("daily_targets") or {}
    ex = plan.get("exercise_plan") or {}
    present = _present(current)
    out: list[str] = []

    if staff:
        out.append(f"{name} ({patient.get('age', '?')} y, {patient.get('gender', '?')}, MRN {patient.get('mrn', 'n/a')}) - "
                   f"visit {current.get('visit_date', 'n/a')}: overall risk {_f(risk)}% ({tier}); "
                   f"ECG heart age {_f(pred.get('predicted_heart_age'))} y (gap {_f(pred.get('heart_age_gap'))} y).")
        if previous:
            pp = previous.get("predictions") or {}
            out.append(f"Tier transition: {pp.get('severity_tier', 'n/a')} ({_f(pp.get('overall_risk_score'))}%) -> {tier} "
                       f"({_f(risk)}%), delta {_f(deltas.get('risk_delta'))} points since {previous.get('visit_date')}.")
            lines = [f"{b['label']} {b['previous']} -> {b['current']} {b['unit']} ({b['status']})"
                     for b in deltas.get("biomarker_deltas") or []]
            if lines:
                out.append("Biomarker deltas: " + "; ".join(lines) + ".")
            new = [a["code"] for a in deltas.get("new_arrhythmias") or []]
            if new:
                out.append("New CODE-15% findings: " + ", ".join(new) + ".")
        else:
            out.append("Baseline visit; no earlier visit to compare.")
        if present:
            out.append("Present codes: " + ", ".join(present) + ".")
        dsp = (current.get("ecg_metadata") or {}).get("dsp_measurements") or {}
        dsp_bits = []
        if dsp.get("heart_rate") is not None:
            dsp_bits.append(f"HR {_f(dsp['heart_rate'], 0)} bpm")
        if dsp.get("qrs_ms") is not None:
            dsp_bits.append(f"QRS {_f(dsp['qrs_ms'], 0)} ms")
        if dsp.get("pr_ms") is not None:
            dsp_bits.append(f"PR {_f(dsp['pr_ms'], 0)} ms")
        if dsp.get("st_amp") is not None:
            dsp_bits.append(f"ST {_f(dsp['st_amp'], 2)} mV")
        if dsp_bits:
            out.append("DSP: " + ", ".join(dsp_bits) + ".")
        if targets:
            out.append(f"Plan targets: sodium {targets.get('sodium_mg')} mg, protein {targets.get('protein_g')} g, "
                       f"fluid {targets.get('fluid_liters')} L, potassium {targets.get('potassium_mg')} mg.")
        if ex:
            out.append(f"Exercise: {ex.get('allowed_activity')}, {ex.get('duration_mins')} min, "
                       f"max HR {ex.get('max_hr_ceiling')} bpm.")
            if ex.get("warnings"):
                out.append("Contraindications: " + " ".join(ex["warnings"]))
        if plan.get("clinical_rules_applied"):
            out.append("Rules applied: " + ", ".join(plan["clinical_rules_applied"]) + ".")
        out.append("Plan status: " + ("approved by staff." if current.get("approved_by_staff") else "pending staff sign-off."))
        return "\n".join(out)

    # Patient mode
    if previous:
        pp = previous.get("predictions") or {}
        direction = "gone up" if (deltas.get("risk_delta") or 0) > 0 else "come down"
        out.append(f"Hi {name}. Since your visit on {previous.get('visit_date')}, your heart risk score has {direction} "
                   f"from {_f(pp.get('overall_risk_score'))}% to {_f(risk)}% ({tier} risk).")
        worse = [b for b in deltas.get("biomarker_deltas") or [] if b.get("status") == "worsened"]
        if worse:
            out.append("What got worse: " + "; ".join(f"{b['label']} went from {b['previous']} to {b['current']} {b['unit']}"
                                                       for b in worse) + ".")
        better = [b for b in deltas.get("biomarker_deltas") or [] if b.get("status") == "improved"]
        if better:
            out.append("What improved: " + "; ".join(f"{b['label']} went from {b['previous']} to {b['current']} {b['unit']}"
                                                      for b in better) + ".")
        new = [CODE_NAMES.get(a["code"], a["name"]) for a in deltas.get("new_arrhythmias") or []]
        if new:
            out.append("Your new heart tracing also shows " + " and ".join(new) + ".")
        resolved = [CODE_NAMES.get(a["code"], a["name"]) for a in deltas.get("resolved_arrhythmias") or []]
        if resolved:
            out.append("Good news: " + " and ".join(resolved) + " is no longer seen.")
        if current.get("adherence_notes"):
            out.append(f"You told us: \"{current['adherence_notes']}\"")
    else:
        out.append(f"Hi {name}. This is your baseline visit, so there is nothing earlier to compare yet. "
                   f"Your heart risk score today is {_f(risk)}% ({tier} risk).")
        if present:
            out.append("Your heart tracing shows " + " and ".join(CODE_NAMES[c] for c in present) + ".")
    if targets:
        out.append(f"Food today: keep salt (sodium) under {targets.get('sodium_mg')} mg a day and drink about "
                   f"{targets.get('fluid_liters')} L of fluid in total.")
    if ex:
        sessions = ex.get("sessions_per_day") or 1
        out.append(f"Activity: {ex.get('allowed_activity', 'gentle walking').lower()} for {ex.get('duration_mins')} minutes"
                   f"{f', {sessions} times a day' if sessions and sessions > 1 else ''}, keeping your heart rate below "
                   f"{ex.get('max_hr_ceiling')} beats per minute.")
    if plan:
        out.append("Your plan is approved by your hospital care team." if current.get("approved_by_staff")
                   else "Your plan is waiting for your hospital dietitian's review.")
    out.append("Please talk to your care team before any change in diagnosis or medication.")
    return " ".join(out)


_TOPICS = {
    "risk": ("risk", "score", "worse", "better", "change", "improve", "why"),
    "diet": ("diet", "food", "eat", "meal", "sodium", "salt", "paneer", "pakoda", "drink"),
    "exercise": ("exercise", "walk", "workout", "activity", "heart rate", "gym", "run"),
    "medication": ("medication", "medicine", "drug", "pill", "dose", "tablet"),
}


def _last_user(messages: list) -> str:
    for m in reversed(messages or []):
        if (m.get("role") if isinstance(m, dict) else getattr(m, "role", "")) == "user":
            return (m.get("content") if isinstance(m, dict) else getattr(m, "content", "")) or ""
    return ""


# Biomarker questions: keyword -> (section, key, label, unit)
_BIOMARKER_TOPICS = [
    (("creatinine", "kidney", "renal"), "blood", "creatinine", "Serum creatinine", "mg/dL"),
    (("glucose", "sugar", "diabet"), "blood", "glucose", "Fasting glucose", "mg/dL"),
    (("cholesterol", "lipid"), "blood", "cholesterol", "Total cholesterol", "mg/dL"),
    (("blood pressure", "bp", "pressure", "hypertension"), "vitals", "bp", "Blood pressure", "mmHg"),
    (("heart rate", "pulse"), "vitals", "hr", "Heart rate", "bpm"),
]
# Food words -> (reason, plan-condition that makes it a problem or None = always)
_FOOD_FLAGS = [
    (("pakoda", "pakora", "samosa", "fried", "bhaji", "vada", "puri", "chips"), "deep-fried and salty", None),
    (("pickle", "achar", "papad", "namkeen"), "very high in salt", None),
    (("sweet", "mithai", "jalebi", "gulab", "cake", "ice cream", "sugar", "dessert"), "high in sugar", "glucose"),
    (("banana", "coconut water", "potato", "tomato", "orange"), "high in potassium", "renal"),
    (("paneer", "cheese", "ghee", "butter", "cream", "red meat", "mutton"), "high in saturated fat", "cholesterol"),
]


def _n(x):
    """Numbers as patients read them: 250 not 250.0, 1.7 stays 1.7."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return x
    return f"{f:g}"


def _value(visit: dict | None, section: str, key: str):
    src = (visit or {}).get(section) or {}
    if key == "bp":
        s, d = src.get("bp_sys"), src.get("bp_dia")
        return f"{_n(s)}/{_n(d)}" if s is not None and d is not None else None
    v = src.get(key)
    return None if v is None else (f"{float(v):.1f}" if key == "creatinine" else _n(v))


def _biomarker_answer(q: str, current: dict, previous: dict | None, deltas: dict, plan: dict) -> list[str]:
    out = []
    for words, section, key, label, unit in _BIOMARKER_TOPICS:
        if not any(re.search(rf"\b{re.escape(w)}", q) for w in words):
            continue
        now = _value(current, section, key)
        if now is None:
            continue
        line = f"{label} today is {now} {unit}"
        if previous is not None and _value(previous, section, key) is not None:
            line += f"; at your visit on {previous.get('visit_date')} it was {_value(previous, section, key)} {unit}"
        bkey = "bp_sys" if key == "bp" else key
        b = next((x for x in (deltas or {}).get("biomarker_deltas") or [] if x.get("key") == bkey), None)
        if b:
            line += f" — {b.get('status')}" + (f" ({b.get('clinical_note')})" if b.get("clinical_note") else "")
        line += "."
        t = plan.get("daily_targets") or {}
        if key == "creatinine":
            line += f" Your plan keeps protein to {t.get('protein_g')} g, potassium to {t.get('potassium_mg')} mg and fluid to {t.get('fluid_liters')} L a day."
        elif key == "glucose":
            line += f" Meals are {'low-glycemic (no sugar, jaggery or white bread)' if 'diabetic_low_gi_meals' in (plan.get('clinical_rules_applied') or []) else 'balanced'} with {t.get('fiber_g')} g fiber a day."
        elif key in ("cholesterol",):
            line += f" Saturated fat is limited to {t.get('saturated_fat_g', 'a low amount of')} g a day."
        elif key == "bp":
            line += f" Sodium is limited to {t.get('sodium_mg')} mg a day."
        out.append(line)
    return out


def _food_answer(q: str, current: dict, plan: dict) -> str | None:
    rules = plan.get("clinical_rules_applied") or []
    blood = current.get("blood") or {}
    active = {
        "glucose": (blood.get("glucose") or 0) >= 100 or "low_glycemic_index" in rules,
        "renal": "renal_restriction" in rules,
        "cholesterol": (blood.get("cholesterol") or 0) >= 240,
    }
    problems = []
    for words, reason, cond in _FOOD_FLAGS:
        hit = next((w for w in words if re.search(rf"\b{re.escape(w)}", q)), None)
        if hit and (cond is None or active.get(cond)):
            why = {"glucose": f" and your fasting glucose is {_n(blood.get('glucose'))} mg/dL",
                   "renal": f" and your creatinine is {_n(blood.get('creatinine'))} mg/dL",
                   "cholesterol": f" and your cholesterol is {_n(blood.get('cholesterol'))} mg/dL"}.get(cond or "", "")
            problems.append(f"{hit} is {reason}{why}")
    pref = (plan.get("diet_preference") or "").replace("_", "-")
    try:
        from genai.diet_planner import EXCLUSION_LISTS, find_excluded
        bad = find_excluded(q, EXCLUSION_LISTS.get(plan.get("diet_preference") or "", []), vegan=plan.get("diet_preference") == "vegan")
        if bad:
            problems.append(f"{', '.join(bad)} is not part of your {pref} diet")
    except Exception:
        pass
    if not problems:
        return None
    return "It is best to skip that: " + "; ".join(problems) + "."


def build_longitudinal_chat_answer(messages: list, ctx: dict) -> str:
    patient, current, previous, deltas, plan, staff = _ctx_parts(ctx)
    question = _last_user(messages).lower()
    pred = current.get("predictions") or {}
    prev_pred = (previous or {}).get("predictions") or {}
    targets = plan.get("daily_targets") or {}
    ex = plan.get("exercise_plan") or {}
    paragraphs: list[str] = []
    matched = [t for t, kws in _TOPICS.items() if any(re.search(rf"\b{re.escape(k)}", question) for k in kws)]

    if "risk" in matched:
        if previous and deltas:
            head = (f"Your risk was {_f(prev_pred.get('overall_risk_score'))}% ({prev_pred.get('severity_tier')}) on "
                    f"{previous.get('visit_date')} and is {_f(pred.get('overall_risk_score'))}% ({pred.get('severity_tier')}) "
                    f"on {current.get('visit_date')}.")
            worse = [b for b in deltas.get("biomarker_deltas") or [] if b.get("status") == "worsened"]
            new = [CODE_NAMES.get(a["code"], a["name"]) if not staff else a["code"] for a in deltas.get("new_arrhythmias") or []]
            reasons = []
            if new:
                reasons.append("the new ECG shows " + " and ".join(new))
            reasons += [f"{b['label']} went from {b['previous']} to {b['current']} {b['unit']}" for b in worse[:4]]
            if reasons:
                head += (" It went up mainly because " if (deltas.get("risk_delta") or 0) > 0 else " Changes: ") + "; ".join(reasons) + "."
            better = [b for b in deltas.get("biomarker_deltas") or [] if b.get("status") == "improved"]
            if better:
                head += " Improved: " + "; ".join(f"{b['label']} {b['previous']} → {b['current']} {b['unit']}" for b in better[:3]) + "."
            if current.get("adherence_notes"):
                head += f" You also told us: \"{current['adherence_notes']}\""
            paragraphs.append(head)
        else:
            paragraphs.append(f"Your current risk score is {_f(pred.get('overall_risk_score'))}% "
                              f"({pred.get('severity_tier', 'unknown')}) on {current.get('visit_date')}. This is the first visit on record.")
    paragraphs += _biomarker_answer(question, current, previous, deltas, plan)
    if "diet" in matched and targets:
        verdict = _food_answer(question, current, plan)
        text = (verdict + " " if verdict else "") + (
            f"Your plan limits sodium to {targets.get('sodium_mg')} mg a day, protein to about "
            f"{targets.get('protein_g')} g and fluid to {targets.get('fluid_liters')} L.")
        meals = plan.get("meals") or {}
        slot = "dinner" if any(w in question for w in ("dinner", "tonight", "night")) else \
               "breakfast" if "breakfast" in question else "lunch" if "lunch" in question else \
               "snack" if "snack" in question else "dinner"
        if meals.get(slot):
            text += f" For {slot} today your plan suggests: " + ", ".join(meals[slot].get("items", [])) + "."
        paragraphs.append(text)
    if "exercise" in matched and ex:
        calc = ex.get("hr_calculation") or {}
        text = (f"Your exercise is now {ex.get('allowed_activity', '').lower()} for {ex.get('duration_mins')} minutes "
                f"{(ex.get('frequency') or '').lower()}, keeping your heart rate under {ex.get('max_hr_ceiling')} bpm")
        if calc.get("resting_ecg_hr"):
            text += (f" (worked out from your age {calc.get('age')} and today's resting ECG heart rate of "
                     f"{calc.get('resting_ecg_hr')} bpm)")
        text += "."
        prev_ex = ((previous or {}).get("diet_plan") or {}).get("exercise_plan") or {}
        if prev_ex and (prev_ex.get("allowed_activity") != ex.get("allowed_activity")
                        or prev_ex.get("max_hr_ceiling") != ex.get("max_hr_ceiling")):
            reasons = []
            if prev_pred.get("overall_risk_score") is not None:
                reasons.append(f"risk {_f(prev_pred.get('overall_risk_score'))}% → {_f(pred.get('overall_risk_score'))}%")
            reasons += [CODE_NAMES.get(a["code"], a["name"]) + " on the new ECG" for a in (deltas or {}).get("new_arrhythmias") or []]
            text += (f" On {previous.get('visit_date')} it was {prev_ex.get('allowed_activity', '').lower()} for "
                     f"{prev_ex.get('duration_mins')} minutes with a {prev_ex.get('max_hr_ceiling')} bpm limit; it changed because of "
                     + (", ".join(reasons) or "your new results") + ".")
        paragraphs.append(text)
    if "medication" in matched:
        drivers = [s.get("display_name") for s in (current.get("shap") or [])[:3] if s.get("shap_score", 0) > 0]
        paragraphs.append("I can't recommend starting, stopping or changing the dose of any medicine. Please ask your "
                          f"cardiologist, and bring today's results: risk {_f(pred.get('overall_risk_score'))}% "
                          f"({pred.get('severity_tier')})" + (f", main drivers {', '.join(drivers)}" if drivers else "") + ".")
    if not paragraphs:
        present = [CODE_NAMES[c] if not staff else c for c in _present(current)]
        line = (f"Your latest visit ({current.get('visit_date', 'n/a')}) shows a risk score of "
                f"{_f(pred.get('overall_risk_score'))}% ({pred.get('severity_tier', 'unknown')})"
                + (f", with {', '.join(present)}" if present else ", with no major ECG abnormality") + ".")
        if previous:
            line += (f" At the previous visit on {previous.get('visit_date')} it was "
                     f"{_f(prev_pred.get('overall_risk_score'))}% ({prev_pred.get('severity_tier')}).")
        if targets and ex:
            line += (f" Today's plan: sodium under {targets.get('sodium_mg')} mg, {ex.get('allowed_activity', '').lower()} "
                     f"with a {ex.get('max_hr_ceiling')} bpm limit. Ask me about your creatinine, sugar, BP, food or exercise.")
        paragraphs.append(line)
    if not staff:
        paragraphs.append("Please check with your care team before any change in diagnosis or medication.")
    return "\n\n".join(p for p in paragraphs if p)


def build_context_chat_answer(messages: list, vitals_context: dict, predictions_context: dict) -> str:
    if not vitals_context and not predictions_context:
        return "No patient data is available yet. Add vitals or run a prediction, then ask again."
    vitals = ", ".join(f"{k}: {v}" for k, v in (vitals_context or {}).items()) or "not provided"
    pc = predictions_context or {}
    return (f"Based on the current data (vitals: {vitals}), the overall risk score is "
            f"{pc.get('overall_risk_score', 'not available')} and the primary condition is "
            f"{pc.get('primary_condition', 'not available')}. Please consult a physician for diagnosis.")


def _context_prompt(ctx: dict) -> str:
    patient, current, previous, deltas, plan, staff = _ctx_parts(ctx)

    def slim(v: dict | None) -> dict | None:
        if not v:
            return None
        meta = dict(v.get("ecg_metadata") or {})
        meta.pop("leads_preview", None)
        return {k: v.get(k) for k in ("visit_date", "hospital_name", "vitals", "blood", "predictions", "code15_flags",
                                      "shap", "adherence_notes", "approved_by_staff", "staff_notes",
                                      "diet_plan")} | {"ecg": meta}

    slim_current = slim(current)
    if slim_current:
        slim_current.pop("diet_plan", None)  # sent once below as "diet_plan" (the current plan)
    if slim_current and slim_current.get("predictions"):
        slim_current["predictions"] = {k: v for k, v in slim_current["predictions"].items() if k != "cross_attention_weights"}
    return json.dumps({"patient_profile": patient, "current_visit": slim_current, "previous_visit": slim(previous),
                       "deltas": {k: deltas.get(k) for k in ("risk_delta", "heart_age_delta", "biomarker_deltas",
                                  "new_arrhythmias", "resolved_arrhythmias", "what_went_wrong_summary",
                                  "what_improved_summary")} if deltas else None,
                       "diet_plan": plan}, default=str)


async def _stream_words(text: str) -> AsyncGenerator[str, None]:
    for word in text.split(" "):
        yield word + " "
        if FALLBACK_TOKEN_DELAY:
            await asyncio.sleep(FALLBACK_TOKEN_DELAY)


async def _llm_or_fallback(system: str, messages: list, fallback: str) -> AsyncGenerator[str, None]:
    """Stream from the LLM; before the first token, any failure switches to ``fallback``."""
    from genai.llm_provider import LLMError, configured_provider, stream_text

    if configured_provider() == "none":
        async for t in _stream_words(fallback):
            yield t
        return
    try:
        async for chunk in stream_text(system, messages):
            yield chunk
    except LLMError:
        async for t in _stream_words(fallback):
            yield t


async def generate_narrative_stream(
    vitals: dict, conditions: list, shap: list, trend: list, patient_context: dict,
    *, longitudinal: dict | None = None,
) -> AsyncGenerator[str, None]:
    if longitudinal and longitudinal.get("current_visit"):
        staff = bool(longitudinal.get("is_staff_mode"))
        system = ("You are CardiacSense AI, a longitudinal cardiac companion. Explain what changed between the "
                  "previous and current hospital visit. Quote the exact previous and current numbers with their units "
                  "and dates from the JSON (risk %, severity tier, BP, creatinine, glucose, cholesterol, heart rate, ECG "
                  "heart age, new or resolved CODE-15% arrhythmias). Never write generic advice that is not tied to a "
                  "number in the JSON. " + mode_instructions(staff))
        user = ("Write the visit summary (150-250 words) in 3 short paragraphs: 1) what the new ECG and blood report "
                "show today, 2) what got worse or better since the previous visit and why, 3) what changed in the diet "
                "and exercise plan compared with the previous plan. Context:\n" + _context_prompt(longitudinal))
        async for t in _llm_or_fallback(system, [{"role": "user", "content": user}],
                                        build_longitudinal_narrative(longitudinal)):
            yield t
        return

    prompt = _build_narrative_prompt(vitals, conditions, shap, trend, patient_context)
    async for t in _llm_or_fallback("", [{"role": "user", "content": prompt}],
                                    _deterministic_narrative(vitals, conditions, shap)):
        yield t


RECOMMEND_PROMPT = """You are CardiacSense AI, a clinical assistant.

PATIENT VITALS: {vitals_text}
PRIMARY CONDITION: {primary_condition} ({primary_prob})
ALL CONDITIONS: {conditions_text}
TOP SHAP DRIVERS: {shap_text}
{narrative_section}
Generate exactly 3 specific clinical action recommendations tailored to this patient's unique report.
Return ONLY a JSON array — no markdown, no explanation:
[
  {{"label": "concise action (max 7 words)", "badge": "+24% Diagnostic", "type": "urgent"}},
  {{"label": "concise action", "badge": "-18% Risk", "type": "moderate"}},
  {{"label": "concise action", "badge": "-12% Tone", "type": "routine"}}
]
type must be one of: urgent, moderate, routine.
badge must start with + or - and include a % and short descriptor."""


async def generate_recommendations(vitals: dict, conditions: list, shap: list, narrative: str = '') -> list:
    primary = conditions[0] if conditions else {}
    vitals_text = ", ".join(f"{k}: {v}" for k, v in vitals.items() if k in ("hr", "bp_systolic", "bp_diastolic", "spo2", "rr", "temp"))
    conditions_text = ", ".join(f"{c.get('name','?')}: {c.get('probability',0):.0%}" for c in conditions)
    shap_text = "; ".join(f"{s.get('display_name','?')} ({s.get('shap_score',0):+.2f})" for s in shap[:4])
    narrative_section = f"CLINICAL SUMMARY:\n{narrative}" if narrative else ""

    fallback = _rule_recommendations(vitals, conditions, shap)

    from genai.llm_provider import LLMError, complete_text, configured_provider
    if configured_provider() == "none":
        return fallback
    try:
        prompt = RECOMMEND_PROMPT.format(
            vitals_text=vitals_text,
            primary_condition=primary.get("name", "Unknown"),
            primary_prob=f"{primary.get('probability', 0):.0%}",
            conditions_text=conditions_text,
            shap_text=shap_text or "No SHAP data",
            narrative_section=narrative_section,
        )
        raw, _provider = await asyncio.to_thread(complete_text, "", prompt, timeout=20.0, max_tokens=600)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        return data if isinstance(data, list) and data else fallback
    except (LLMError, ValueError, TypeError):
        return fallback


def _rule_recommendations(vitals: dict, conditions: list, shap: list) -> list:
    """Three actions derived from the submitted values (most abnormal first)."""
    abnormal, _ = _vital_findings(vitals or {})
    recs: list[dict] = []
    for text in abnormal[:2]:
        label = text.split(" is ")[0]
        recs.append({"label": f"Re-check {label.lower()}"[:48], "badge": "+Recheck within 24h", "type": "urgent" if not recs else "moderate"})
    ranked = sorted(conditions or [], key=lambda c: c.get("probability", 0), reverse=True)
    if ranked and len(recs) < 3:
        top = ranked[0]
        recs.append({"label": f"Review {top.get('name', 'top finding')} ({top.get('probability', 0):.0%})"[:48],
                     "badge": f"+{round(top.get('probability', 0) * 100)}% Probability", "type": "moderate" if recs else "urgent"})
    for d in sorted(shap or [], key=lambda s: -abs(s.get("shap_score", 0))):
        if len(recs) >= 3:
            break
        recs.append({"label": f"Address {d.get('display_name', d.get('feature'))}"[:48],
                     "badge": f"{d.get('shap_score', 0) * 100:+.0f}% Driver", "type": "routine"})
    return recs or [{"label": "All submitted vitals are within range", "badge": "+0% Change", "type": "routine"}]


async def generate_chat_stream(
    messages: list, vitals_context: dict, predictions_context: dict,
    *, longitudinal: dict | None = None,
) -> AsyncGenerator[str, None]:
    if longitudinal and longitudinal.get("current_visit"):
        staff = bool(longitudinal.get("is_staff_mode"))
        system = ("You are CardiacSense AI, a 24/7 cardiac companion that knows this patient's full history. "
                  "Answer the user's exact question directly in the first sentence, then support it with the exact "
                  "previous-visit and current-visit numbers (with dates and units), the deltas, and the current vs. previous "
                  "diet & exercise plan from the JSON. Do not repeat the question, do not give generic advice that is not "
                  "tied to this patient's numbers, and keep it under 180 words. "
                  + mode_instructions(staff) + "\n\nPATIENT CONTEXT:\n" + _context_prompt(longitudinal))
        async for t in _llm_or_fallback(system, messages, build_longitudinal_chat_answer(messages, longitudinal)):
            yield t
        return

    from genai.llm_provider import configured_provider

    if configured_provider() == "none":
        async for t in _stream_words(build_context_chat_answer(messages, vitals_context, predictions_context)):
            yield t
        return

    vitals_str = ", ".join(f"{k}: {v}" for k, v in vitals_context.items()) if vitals_context else "not provided"
    clinical_summary = predictions_context.get('clinical_summary', '') if predictions_context else ''
    pred_str = (
        f"Primary condition: {predictions_context.get('primary_condition', 'unknown')}, "
        f"Risk score: {predictions_context.get('overall_risk_score', 'unknown')}"
        if predictions_context else "not provided"
    )
    system_content = (
        f"{CHAT_SYSTEM}\n\n"
        f"Current vitals: {vitals_str}\n"
        f"Current predictions: {pred_str}\n"
        + (f"Clinical summary report: {clinical_summary}\n" if clinical_summary else "")
    )
    async for t in _llm_or_fallback(system_content, messages,
                                    build_context_chat_answer(messages, vitals_context, predictions_context)):
        yield t
