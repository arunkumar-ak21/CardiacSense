"""
Longitudinal Delta wrapper and adaptive Diet & Exercise planner.

Everything in the plan is computed from TODAY's report compared with the PREVIOUS
visit stored in companion.db:

* daily targets (sodium, saturated fat, protein, potassium, fluid, fiber, calories)
  follow clinical thresholds on today's BP, cholesterol, creatinine and glucose;
* meals switch to low-glycemic options when glucose is high, to low-potassium options
  when creatinine is high, and always respect the patient's diet preference;
* the exercise ceiling is a Karvonen heart-rate target from the patient's age and
  resting ECG heart rate, lowered further when risk / AF / LBBB / ST worsen;
* ``why_plan_changed_summary`` cites the exact old -> new numbers and dates.

When ``GEMINI_API_KEY`` (or ``OPENAI_API_KEY``) is set, the LLM rewrites the meal items
and the explanation. Its meals are checked against the exclusion lists, its explanation
must quote the same numbers as the rule-based one, and all numeric targets stay rule-based.
On any LLM failure the rule-based plan is returned.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from db import delta_engine

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
logger = logging.getLogger("vitalsense.diet_planner")

DIET_PREFERENCES = ("vegetarian", "non_vegetarian", "vegan", "diabetic", "dash")

_VEG = [
    "meat", "chicken", "mutton", "beef", "pork", "lamb", "bacon", "ham", "sausage", "turkey",
    "poultry", "fish", "salmon", "tuna", "sardine", "seafood", "prawn", "shrimp", "crab", "egg",
]
EXCLUSION_LISTS: dict[str, list[str]] = {
    "non_vegetarian": [],
    "vegetarian": _VEG,
    "vegan": _VEG + ["dairy", "milk", "cheese", "butter", "ghee", "paneer", "yogurt", "curd", "cream", "honey"],
    "diabetic": ["sugar", "sweets", "dessert", "candy", "jaggery", "syrup", "soda", "soft drink", "sweetened"],
    "dash": ["pickle", "papad", "chips", "instant noodles", "salami", "bacon", "sausage", "processed meat", "salted"],
}
RENAL_EXCLUSIONS = ["banana", "potato", "tomato", "spinach", "orange", "avocado", "coconut water"]
PLANT_PREFIXES = ("soy", "almond", "oat", "coconut", "peanut", "cashew", "plant-based")
PLANT_EXEMPT = ("milk", "cheese", "butter", "yogurt", "curd", "cream")
# Removed from meals when total cholesterol is high (>= 240 mg/dL)
SATFAT_EXCLUSIONS = ["paneer", "cheese", "ghee", "butter", "cream", "mutton", "red meat"]

# Clinical thresholds
BP_SYS_HIGH, BP_DIA_HIGH = 130, 80
CREAT_RENAL, CREAT_SEVERE = 1.4, 2.0
GLUCOSE_PREDIABETIC, GLUCOSE_HIGH = 100, 140
CHOL_BORDERLINE, CHOL_HIGH = 200, 240


# ═══════════════════════════════════════════════════════════════════════════
# Deltas
# ═══════════════════════════════════════════════════════════════════════════
def compute_visit_deltas(current_visit_dict: dict, previous_visit_dict: dict | None) -> dict:
    """Before-vs-Now comparison; delegates to ``db.delta_engine`` (one implementation)."""
    return delta_engine.compute_visit_deltas(previous_visit_dict, current_visit_dict)


# ═══════════════════════════════════════════════════════════════════════════
# Exclusion matching
# ═══════════════════════════════════════════════════════════════════════════
def find_excluded(text: str, keywords: list[str], *, vegan: bool = False) -> list[str]:
    """Keywords found in ``text`` (case-insensitive, whole word/phrase, plural s/es)."""
    hits: list[str] = []
    lowered = (text or "").lower()
    for kw in keywords:
        pattern = r"\b" + r"\s+".join(map(re.escape, kw.lower().split())) + r"(?:s|es)?\b"
        for m in re.finditer(pattern, lowered):
            if vegan and kw in PLANT_EXEMPT:
                before = lowered[: m.start()].rstrip()
                if any(before.endswith(p) for p in PLANT_PREFIXES):
                    continue
            hits.append(kw)
            break
    return hits


def _item_allowed(item: str, preference: str, renal: bool, low_gi: bool = False, high_chol: bool = False) -> bool:
    if find_excluded(item, EXCLUSION_LISTS.get(preference, []), vegan=preference == "vegan"):
        return False
    if low_gi and find_excluded(item, EXCLUSION_LISTS["diabetic"]):
        return False
    if high_chol and find_excluded(item, SATFAT_EXCLUSIONS):
        return False
    return not (renal and find_excluded(item, RENAL_EXCLUSIONS))


# ═══════════════════════════════════════════════════════════════════════════
# Conditions (today's numbers + previous numbers)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Conditions:
    high_bp: bool
    renal: bool
    metabolic: bool
    cardiac: bool
    risk: float | None
    age: float | None
    bp_sys: float | None
    bp_dia: float | None
    creatinine: float | None
    glucose: float | None
    cholesterol: float | None
    present_codes: tuple[str, ...]
    missing: tuple[str, ...]
    resting_hr: float | None = None
    glucose_high: bool = False
    cholesterol_high: bool = False


def _present_codes(visit: dict) -> list[str]:
    flags = visit.get("code15_flags") or {}
    probs = (visit.get("predictions") or {}).get("arrhythmias") or {}
    out = []
    for code in delta_engine.CODE15_LABELS:
        p = delta_engine._num(probs.get(code))
        if bool(flags.get(code)) or (p is not None and p >= delta_engine.ARRHYTHMIA_THRESHOLD):
            out.append(code)
    return out


def _resting_hr(visit: dict) -> float | None:
    dsp = ((visit.get("ecg_metadata") or {}).get("dsp_measurements")) or {}
    hr = delta_engine._num(dsp.get("heart_rate"))
    if hr is None or hr <= 0 or dsp.get("measurable") is False:
        hr = delta_engine._num((visit.get("vitals") or {}).get("hr"))
    return hr if hr and hr > 0 else None


def evaluate_conditions(patient: dict, visit: dict) -> Conditions:
    vitals = visit.get("vitals") or {}
    blood = visit.get("blood") or {}
    num = delta_engine._num
    values = {
        "bp_sys": num(vitals.get("bp_sys")), "bp_dia": num(vitals.get("bp_dia")),
        "creatinine": num(blood.get("creatinine")), "glucose": num(blood.get("glucose")),
        "cholesterol": num(blood.get("cholesterol")),
    }
    missing = tuple(k for k, v in values.items() if v is None)
    risk = num((visit.get("predictions") or {}).get("overall_risk_score"))
    age = num((patient or {}).get("age"))
    codes = tuple(_present_codes(visit))
    glucose = values["glucose"] or 0
    chol = values["cholesterol"] or 0
    return Conditions(
        high_bp=(values["bp_sys"] or 0) >= BP_SYS_HIGH or (values["bp_dia"] or 0) >= BP_DIA_HIGH,
        renal=(values["creatinine"] or 0) >= CREAT_RENAL,
        metabolic=glucose >= GLUCOSE_PREDIABETIC or chol >= CHOL_BORDERLINE,
        cardiac=risk is None or age is None or risk >= 50 or any(c in codes for c in ("AF", "LBBB", "ST")),
        risk=risk, age=age, bp_sys=values["bp_sys"], bp_dia=values["bp_dia"],
        creatinine=values["creatinine"], glucose=values["glucose"], cholesterol=values["cholesterol"],
        present_codes=codes, missing=missing, resting_hr=_resting_hr(visit),
        glucose_high=glucose >= GLUCOSE_HIGH, cholesterol_high=chol >= CHOL_HIGH,
    )


def compute_targets(cond: Conditions, preference: str, gender: str | None, age: float | None):
    """``(daily_targets, exercise_plan, rules)`` computed from today's numbers."""
    rules: list[str] = []
    creat = cond.creatinine or 0.0
    severe = (cond.risk or 0) >= 75 or (cond.bp_sys or 0) >= 160 or creat >= CREAT_SEVERE

    # Sodium & saturated fat — BP, cholesterol, risk
    if cond.high_bp or (cond.risk or 0) >= 50 or cond.cholesterol_high:
        sodium = 1200 if severe else 1300
        rules.append("sodium_restriction")
    else:
        sodium = 1800 if (cond.cholesterol or 0) >= CHOL_BORDERLINE else 2000
        rules.append("sodium_standard")
    chol = cond.cholesterol or 0
    sat_fat = 12 if chol >= CHOL_HIGH else 15 if chol >= CHOL_BORDERLINE else 20
    if chol >= CHOL_BORDERLINE:
        rules.append("saturated_fat_limit")

    # Kidney — creatinine
    if cond.renal:
        fluid = 1.2 if creat >= CREAT_SEVERE else 1.5
        protein = 40 if creat >= CREAT_SEVERE else 50
        potassium = 1800 if creat >= CREAT_SEVERE else 2000
        rules.append("renal_restriction")
        if preference == "dash":
            rules.append("renal_potassium_override")
    else:
        fluid = 2.5
        protein = 75 if preference == "non_vegetarian" else 60
        potassium = 3500
        if preference == "dash":
            potassium = 4000
            rules.append("dash_potassium")

    # Glucose / metabolic
    calories = 2000 if gender == "M" else 1800 if gender == "F" else 1900
    fiber = 30
    if cond.metabolic:
        calories -= 200
        fiber = 35
        rules += ["soluble_fiber", "low_glycemic_index"]
    if cond.glucose_high:
        calories -= 100
        fiber = 38
        rules.append("diabetic_low_gi_meals")

    # Exercise — Karvonen ceiling from age + resting ECG heart rate
    max_hr = 220 - (age if age is not None else 80)
    resting = cond.resting_hr if cond.resting_hr else 75.0
    reserve = max(0.0, max_hr - resting)
    if cond.cardiac:
        very_light = (cond.risk or 100) >= 70 or any(c in cond.present_codes for c in ("AF", "LBBB"))
        frac = 0.30 if very_light else 0.40
        karvonen = math.floor(resting + frac * reserve)
        ceiling = int(min(110, math.floor(0.60 * max_hr), karvonen))
        rest_only = ceiling < resting + 10
        exercise = {
            "allowed_activity": (
                "Seated breathing exercises and gentle stretching only" if rest_only
                else "Slow, flat indoor pacing" if very_light else "Easy flat walking"
            ),
            "duration_mins": 10 if (very_light or rest_only) else 20,
            "sessions_per_day": 2 if very_light else 1,
            "frequency": "Twice daily (morning and evening)" if very_light else "Daily",
            "intensity": "very_light" if (very_light or rest_only) else "light",
            "max_hr_ceiling": max(ceiling, 60),
            "warnings": [
                "No high-intensity interval training (HIIT), running, stairs, uphill walking or lifting over 5 kg.",
                f"Keep your heart rate below {max(ceiling, 60)} bpm; check your pulse before and after each session.",
                "Stop immediately and seek care for chest pain, palpitations, fainting or breathlessness.",
            ],
        }
        if rest_only:
            exercise["warnings"].insert(0, f"Your resting ECG heart rate is {resting:.0f} bpm — walking is paused until your care team reviews your heart rate.")
        if "AF" in cond.present_codes:
            exercise["warnings"].append("Atrial fibrillation: avoid caffeine/energy drinks before activity; exercise only under the care team's guidance.")
        rules.append("cardiac_safe_exercise")
    else:
        karvonen = math.floor(resting + 0.60 * reserve)
        ceiling = int(min(math.floor(0.75 * max_hr), karvonen))
        exercise = {
            "allowed_activity": "Brisk outdoor walking or zone-2 cycling",
            "duration_mins": 30,
            "sessions_per_day": 1,
            "frequency": "5-6 days per week",
            "intensity": "moderate",
            "max_hr_ceiling": ceiling,
            "warnings": [
                "Warm up for 5 minutes and cool down for 5 minutes.",
                "Stop and rest if you feel chest discomfort, dizziness or unusual breathlessness.",
            ],
        }
        rules.append("standard_exercise_ceiling")
    exercise["hr_calculation"] = {
        "age": age, "age_predicted_max_hr": int(max_hr), "resting_ecg_hr": round(resting),
        "method": f"Karvonen: resting + {int(round((0.30 if exercise['intensity'] == 'very_light' else 0.40 if cond.cardiac else 0.60) * 100))}% of heart-rate reserve",
    }

    rules += [f"missing_input_{f}" for f in cond.missing]
    targets = {
        "sodium_mg": sodium, "saturated_fat_g": sat_fat, "protein_g": protein, "fluid_liters": fluid,
        "calories": calories, "potassium_mg": potassium, "fiber_g": fiber,
    }
    return targets, exercise, rules


# ═══════════════════════════════════════════════════════════════════════════
# Meals — standard vs. low-glycemic, per diet preference
# ═══════════════════════════════════════════════════════════════════════════
_Lib = dict[str, list[list[str]]]
STANDARD_MEALS: dict[str, _Lib] = {
    "vegetarian": {
        "breakfast": [["Vegetable poha with peanuts (no added salt)", "1 cup low-fat milk", "1 banana"],
                      ["Oats porridge with low-fat milk (no added salt)", "1 apple, sliced", "1 tsp ground flaxseed"]],
        "lunch": [["2 whole-wheat phulkas", "Moong dal (1 bowl)", "Spinach and tomato sabzi", "Cucumber salad"],
                  ["2 unsalted phulkas", "Lauki (bottle gourd) sabzi with jeera", "Moong dal (1/2 cup)", "Cucumber salad with lemon"]],
        "snack": [["Roasted chana (30 g)", "1 orange"], ["Unsalted puffed rice (1 cup)", "1 pear"]],
        "dinner": [["Brown rice (1 cup)", "Low-fat paneer bhurji with capsicum", "Sauteed beans"],
                   ["Vegetable dalia khichdi with ridge gourd and carrot", "Sauteed cabbage with herbs"]],
    },
    "vegan": {
        "breakfast": [["Oats cooked in soy milk with walnuts", "1 banana"],
                      ["Oats cooked in almond milk", "1 apple, sliced", "1 tsp ground flaxseed"]],
        "lunch": [["Quinoa and chickpea salad with olive oil", "Steamed spinach"],
                  ["2 whole-wheat phulkas", "Lauki (bottle gourd) sabzi", "Moong dal (1/2 cup)"]],
        "snack": [["Unsalted almonds (20 g)", "1 orange"], ["Unsalted puffed rice (1 cup)", "1 pear"]],
        "dinner": [["Tofu and vegetable stir-fry (low-sodium soy sauce)", "Brown rice (2/3 cup)"],
                   ["Vegetable dalia khichdi with ridge gourd", "Sauteed cabbage with herbs"]],
    },
    "non_vegetarian": {
        "breakfast": [["Vegetable omelette (2 egg whites + 1 egg)", "Whole-grain toast", "1 orange"],
                      ["Vegetable omelette (2 egg whites)", "Whole-grain toast", "1 apple"]],
        "lunch": [["Grilled chicken breast (120 g)", "Quinoa salad with tomato and olive oil", "Steamed broccoli"],
                  ["Grilled chicken breast (90 g)", "White rice (2/3 cup)", "Steamed cabbage and carrot"]],
        "snack": [["Unsalted almonds (20 g)", "1 banana"], ["Unsalted rice crackers", "1 pear"]],
        "dinner": [["Baked salmon (120 g)", "Roasted vegetables with olive oil", "Brown rice (2/3 cup)"],
                   ["Baked fish (90 g) with lemon and herbs", "Sauteed green beans", "Plain rice (2/3 cup)"]],
    },
    "dash": {
        "breakfast": [["Oats with low-fat milk and berries", "1 banana", "Unsalted walnuts (10 g)"],
                      ["Oats with low-fat milk", "1 apple", "1 tsp ground flaxseed"]],
        "lunch": [["Grilled chicken or lentil patty", "Brown rice (2/3 cup)", "Spinach and tomato salad"],
                  ["Lentil patty (unsalted)", "Plain rice (2/3 cup)", "Cucumber and cabbage salad"]],
        "snack": [["Low-fat yogurt with berries", "Unsalted almonds (15 g)"], ["Unsalted rice crackers", "1 pear"]],
        "dinner": [["Baked fish or tofu (100 g)", "Steamed broccoli and carrots", "Whole-wheat roti"],
                   ["Vegetable dalia khichdi", "Sauteed green beans"]],
    },
}
LOW_GI_MEALS: dict[str, _Lib] = {
    "vegetarian": {
        "breakfast": [["Vegetable besan chilla (2) with mint chutney", "Unsweetened green tea", "1 guava"],
                      ["Steel-cut oats with cinnamon (no added sugar)", "1 tsp ground flaxseed", "1 small apple"]],
        "lunch": [["2 multigrain (jowar/bajra) rotis", "Rajma (1/2 cup)", "Spinach sabzi", "Cucumber salad"],
                  ["2 multigrain rotis", "Lauki sabzi", "Moong dal (1/2 cup)", "Cucumber salad with lemon"]],
        "snack": [["Roasted chana (30 g)", "Buttermilk (unsweetened)"], ["Unsalted roasted makhana (1 cup)", "1 pear"]],
        "dinner": [["Low-fat paneer tikka (80 g) with grilled vegetables", "Brown rice (1/2 cup)"],
                   ["Moong dal cheela (2) with mint chutney", "Sauteed lauki and capsicum", "Brown rice (1/2 cup)"],
                   ["Vegetable dalia khichdi (whole-grain)", "Sauteed cabbage with herbs"]],
    },
    "vegan": {
        "breakfast": [["Vegetable besan chilla (2) with mint chutney", "Unsweetened green tea", "1 guava"],
                      ["Steel-cut oats in unsweetened almond milk with cinnamon", "1 tsp ground flaxseed"]],
        "lunch": [["2 multigrain rotis", "Rajma (1/2 cup)", "Mixed vegetable sabzi", "Cucumber salad"],
                  ["2 multigrain rotis", "Lauki sabzi", "Moong dal (1/2 cup)"]],
        "snack": [["Roasted chana (30 g)", "Unsalted walnuts (10 g)"], ["Unsalted roasted makhana (1 cup)", "1 pear"]],
        "dinner": [["Tofu and vegetable stir-fry (low-sodium soy sauce)", "Brown rice (1/2 cup)"],
                   ["Vegetable dalia khichdi (whole-grain)", "Sauteed cabbage with herbs"]],
    },
    "non_vegetarian": {
        "breakfast": [["Vegetable omelette (2 egg whites + 1 egg)", "1 multigrain toast", "Unsweetened green tea"],
                      ["Boiled egg whites (3)", "Steel-cut oats with cinnamon (no added sugar)"]],
        "lunch": [["Grilled chicken breast (100 g)", "2 multigrain rotis", "Spinach sabzi", "Cucumber salad"],
                  ["Grilled chicken breast (90 g)", "Brown rice (1/2 cup)", "Steamed cabbage and carrot"]],
        "snack": [["Roasted chana (30 g)", "Buttermilk (unsweetened)"], ["Unsalted roasted makhana (1 cup)", "1 pear"]],
        "dinner": [["Grilled fish (100 g) with lemon and herbs", "Sauteed vegetables", "Brown rice (1/2 cup)"],
                   ["Baked fish (90 g)", "Sauteed green beans", "1 multigrain roti"]],
    },
}
MEAL_TIMES = {"breakfast": "08:00", "lunch": "13:00", "snack": "17:00", "dinner": "19:30"}
SAFE_DEFAULT = {
    "breakfast": ["Oats porridge cooked in water", "1 apple, sliced"],
    "lunch": ["2 unsalted phulkas", "Lauki (bottle gourd) sabzi", "Cucumber salad with lemon"],
    "snack": ["Unsalted puffed rice (1 cup)", "1 pear"],
    "dinner": ["Vegetable dalia khichdi with ridge gourd", "Sauteed cabbage with herbs"],
}
# Backwards-compatible name used by older imports.
MEAL_LIBRARY = STANDARD_MEALS


def _library(preference: str, low_gi: bool) -> _Lib:
    if low_gi:
        base = preference if preference in LOW_GI_MEALS else "non_vegetarian"
        return LOW_GI_MEALS[base]
    if preference == "diabetic":
        return LOW_GI_MEALS["non_vegetarian"]
    return STANDARD_MEALS.get(preference, STANDARD_MEALS["non_vegetarian"])


def fallback_meals(preference: str, renal: bool, cond: Conditions | None = None) -> dict:
    low_gi = bool(cond and (cond.glucose_high or preference == "diabetic"))
    high_chol = bool(cond and cond.cholesterol_high)
    library = _library(preference, low_gi)
    meals: dict[str, dict] = {}
    for slot in MEAL_TIMES:
        options = library[slot]
        items = next((o for o in options if all(_item_allowed(i, preference, renal, low_gi, high_chol) for i in o)), SAFE_DEFAULT[slot])
        notes: list[str] = []
        if cond:
            if slot == "breakfast" and cond.glucose_high:
                notes.append(f"Low-glycemic start because fasting glucose is {cond.glucose:g} mg/dL — no sugar, jaggery or white bread.")
            elif slot == "breakfast" and cond.metabolic:
                notes.append("Soluble fiber (oats, flaxseed) helps lower cholesterol"
                             + (f" ({cond.cholesterol:g} mg/dL today)." if cond.cholesterol else "."))
            if slot == "lunch" and renal:
                notes.append(f"Low-potassium vegetables because creatinine is {_cr(cond.creatinine)} mg/dL.")
            if slot == "lunch" and cond.high_bp:
                notes.append(f"No added salt — BP today is {cond.bp_sys:g}/{cond.bp_dia:g} mmHg." if cond.bp_sys and cond.bp_dia else "No added salt.")
            if slot == "snack" and renal:
                notes.append("Count tea and water toward your daily fluid limit.")
            if slot == "dinner" and (cond.cholesterol or 0) >= CHOL_BORDERLINE:
                notes.append("Cook with 1-2 tsp oil only; no ghee, butter or cream (saturated fat limit).")
        if slot == "dinner":
            notes.append("Finish dinner 2 hours before bed.")
        meals[slot] = {"time": MEAL_TIMES[slot], "items": list(items), "notes": " ".join(notes) or "Keep portions moderate."}
    return meals


def _foods_to_avoid(preference: str, cond: Conditions) -> list[str]:
    avoid = ["Pickles (achar) and papad", "Restaurant, takeaway and street food",
             "Salted namkeen, chips and instant noodles", "Deep-fried snacks (samosa, pakoda)"]
    if cond.high_bp and cond.bp_sys:
        avoid[0] = f"Pickles (achar), papad and added salt (BP {cond.bp_sys:g}/{cond.bp_dia or 0:g} mmHg)"
    if (cond.cholesterol or 0) >= CHOL_BORDERLINE:
        avoid.append(f"Ghee, butter, cream, full-fat paneer and red meat (cholesterol {cond.cholesterol:g} mg/dL)")
    if cond.metabolic:
        label = f" (glucose {cond.glucose:g} mg/dL)" if cond.glucose else ""
        avoid += [f"Refined sugar, sweets, jaggery and sugary drinks or soda{label}",
                  "White bread, maida and large portions of white rice"]
    if cond.renal:
        avoid += [f"High-potassium foods: bananas, coconut water, potatoes, tomatoes, oranges (creatinine {cond.creatinine:g} mg/dL)",
                  "Salt substitutes containing potassium chloride", "Large portions of protein (paneer, meat, cheese)"]
    if any(c in cond.present_codes for c in ("AF", "ST")):
        avoid.append("Caffeine-heavy and energy drinks (can trigger palpitations)")
    avoid.append("Alcohol")
    if preference == "non_vegetarian":
        avoid.append("Processed and cured meats (bacon, sausages, ham)")
    return avoid


# ═══════════════════════════════════════════════════════════════════════════
# "Why your plan changed" — exact old -> new numbers
# ═══════════════════════════════════════════════════════════════════════════
def _fmt(x: float | None, nd: int = 0) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}" if nd else f"{x:g}"


def _cr(x: float | None) -> str:
    """Creatinine with clinical precision: 1.0, 1.7, 2.05."""
    if x is None:
        return "n/a"
    return f"{x:.2f}" if round(x * 100) % 10 else f"{x:.1f}"


def _prev_value(prev: dict | None, section: str, key: str) -> float | None:
    return delta_engine._num(((prev or {}).get(section) or {}).get(key))


def _target_change(new: dict, old: dict | None, key: str, unit: str) -> str:
    o = delta_engine._num((old or {}).get(key))
    n = new.get(key)
    if o is None or n is None:
        return f"{n:g} {unit}" if n is not None else ""
    if abs(o - float(n)) < 1e-9:
        return f"{n:g} {unit} (unchanged)"
    return f"{n:g} {unit} (was {o:g} {unit})"


def why_plan_changed(cond: Conditions, targets: dict, exercise: dict, previous_plan: dict | None,
                     deltas: dict | None, previous_visit: dict | None = None, current_visit: dict | None = None) -> str:
    risk_txt = f"{cond.risk:.1f}%" if cond.risk is not None else "not available"
    prev_targets = (previous_plan or {}).get("daily_targets") if isinstance((previous_plan or {}).get("daily_targets"), dict) else None
    prev_ex = (previous_plan or {}).get("exercise_plan") or {}
    hr_calc = exercise.get("hr_calculation", {})

    if not previous_visit or prev_targets is None:
        parts = [f"This is your baseline plan, built from today's report: overall risk {risk_txt}"]
        if cond.bp_sys and cond.bp_dia:
            parts.append(f"BP {_fmt(cond.bp_sys)}/{_fmt(cond.bp_dia)} mmHg")
        if cond.creatinine is not None:
            parts.append(f"creatinine {_cr(cond.creatinine)} mg/dL")
        for label, val, unit in (("fasting glucose", cond.glucose, "mg/dL"), ("cholesterol", cond.cholesterol, "mg/dL")):
            if val is not None:
                parts.append(f"{label} {_fmt(val)} {unit}")
        text = ", ".join(parts) + "."
        if cond.present_codes:
            text += f" The 12-lead ECG shows {', '.join(delta_engine.CODE15_LABELS[c] for c in cond.present_codes)}."
        text += (f" Sodium is set to {targets['sodium_mg']} mg, protein {targets['protein_g']} g, fluid {targets['fluid_liters']} L, and the "
                 f"exercise heart-rate ceiling is {exercise['max_hr_ceiling']} bpm from your age {_fmt(cond.age)} "
                 f"and resting ECG heart rate {hr_calc.get('resting_ecg_hr', 'n/a')} bpm.")
        return text

    date = previous_visit.get("visit_date", "your last visit")
    sentences: list[str] = []
    num = delta_engine._num

    # Creatinine -> protein / potassium / fluid
    p_cr, c_cr = _prev_value(previous_visit, "blood", "creatinine"), cond.creatinine
    if c_cr is not None and p_cr is not None and abs(c_cr - p_cr) >= 0.05:
        kidney_targets = (f"protein is now {_target_change(targets, prev_targets, 'protein_g', 'g')}, potassium "
                          f"{_target_change(targets, prev_targets, 'potassium_mg', 'mg')} and fluid {_target_change(targets, prev_targets, 'fluid_liters', 'L')}")
        if c_cr > p_cr and cond.renal:
            sentences.append(f"Because your Creatinine rose from {_cr(p_cr)} to {_cr(c_cr)} mg/dL since your last visit on {date} "
                             f"(above the {CREAT_RENAL} mg/dL kidney-strain level), {kidney_targets}.")
        elif c_cr < p_cr and p_cr >= CREAT_RENAL > c_cr:
            sentences.append(f"Because your Creatinine improved from {_cr(p_cr)} to {_cr(c_cr)} mg/dL since {date}, the kidney limits were relaxed: {kidney_targets}.")
        elif cond.renal:
            sentences.append(f"Your Creatinine moved from {_cr(p_cr)} to {_cr(c_cr)} mg/dL since {date} and is still above {CREAT_RENAL} mg/dL, so {kidney_targets}.")

    # Glucose -> low-GI meals
    p_gl, c_gl = _prev_value(previous_visit, "blood", "glucose"), cond.glucose
    if c_gl is not None and p_gl is not None and abs(c_gl - p_gl) >= 3:
        if cond.glucose_high and c_gl > p_gl:
            sentences.append(f"Because your fasting glucose jumped from {_fmt(p_gl)} to {_fmt(c_gl)} mg/dL since {date}, every meal was switched "
                             f"to low-glycemic diabetic options (multigrain rotis, dal, no sugar or jaggery), fiber raised to "
                             f"{_target_change(targets, prev_targets, 'fiber_g', 'g')} and calories set to {_target_change(targets, prev_targets, 'calories', 'kcal')}.")
        elif c_gl < p_gl and p_gl >= GLUCOSE_HIGH > c_gl:
            sentences.append(f"Your fasting glucose improved from {_fmt(p_gl)} to {_fmt(c_gl)} mg/dL since {date}, so regular balanced meals are back.")
        elif c_gl > p_gl:
            sentences.append(f"Fasting glucose rose from {_fmt(p_gl)} to {_fmt(c_gl)} mg/dL since {date}; refined carbohydrates stay limited.")

    # BP + cholesterol -> sodium & saturated fat
    p_s, p_d = _prev_value(previous_visit, "vitals", "bp_sys"), _prev_value(previous_visit, "vitals", "bp_dia")
    p_ch = _prev_value(previous_visit, "blood", "cholesterol")
    bp_part = (f"blood pressure went from {_fmt(p_s)}/{_fmt(p_d)} to {_fmt(cond.bp_sys)}/{_fmt(cond.bp_dia)} mmHg"
               if p_s is not None and cond.bp_sys is not None and abs(cond.bp_sys - p_s) >= 3 else None)
    ch_part = (f"cholesterol went from {_fmt(p_ch)} to {_fmt(cond.cholesterol)} mg/dL"
               if p_ch is not None and cond.cholesterol is not None and abs(cond.cholesterol - p_ch) >= 5 else None)
    if bp_part or ch_part:
        cause = " and ".join(x for x in (bp_part, ch_part) if x)
        rising = ((cond.bp_sys or 0) > (p_s or 0) if bp_part else False) or ((cond.cholesterol or 0) > (p_ch or 0) if ch_part else False)
        verb = "cut" if rising else "adjusted"
        sentences.append(f"Because your {cause} since {date}, sodium was {verb} to {_target_change(targets, prev_targets, 'sodium_mg', 'mg')} "
                         f"and saturated fat to {targets.get('saturated_fat_g')} g/day.")

    # Risk / ECG -> exercise
    p_risk = num((previous_visit.get("predictions") or {}).get("overall_risk_score"))
    new_codes = [a.get("code") for a in (deltas or {}).get("new_arrhythmias") or []]
    old_act, old_hr, old_dur = prev_ex.get("allowed_activity"), num(prev_ex.get("max_hr_ceiling")), num(prev_ex.get("duration_mins"))
    ex_changed = (old_act and old_act != exercise["allowed_activity"]) or (old_hr is not None and old_hr != exercise["max_hr_ceiling"]) \
        or (old_dur is not None and old_dur != exercise["duration_mins"])
    if ex_changed:
        why = []
        if p_risk is not None and cond.risk is not None:
            why.append(f"your risk went from {p_risk:.1f}% to {cond.risk:.1f}%")
        if new_codes:
            why.append(f"the new 12-lead ECG shows {', '.join(delta_engine.CODE15_LABELS.get(c, c) for c in new_codes)}")
        lower = old_hr is not None and exercise["max_hr_ceiling"] < old_hr
        sentences.append(
            f"Because {' and '.join(why) or 'your results changed'}, exercise was {'lowered' if lower else 'updated'} from "
            f"'{old_act or 'previous routine'}' ({_fmt(old_dur)} min, max HR {_fmt(old_hr)} bpm) to '{exercise['allowed_activity']}' "
            f"({exercise['duration_mins']} min, max HR {exercise['max_hr_ceiling']} bpm). The new ceiling is calculated from your age "
            f"{_fmt(cond.age)} (age-predicted max {hr_calc.get('age_predicted_max_hr')} bpm) and today's resting ECG heart rate "
            f"{hr_calc.get('resting_ecg_hr')} bpm ({hr_calc.get('method')}).")

    if not sentences:
        return (f"Your report is stable compared with {date} (risk {_fmt(p_risk, 1)}% → {risk_txt}); targets stay at sodium "
                f"{targets['sodium_mg']} mg, protein {targets['protein_g']} g, fluid {targets['fluid_liters']} L and max HR "
                f"{exercise['max_hr_ceiling']} bpm.")
    return " ".join(sentences)


# ═══════════════════════════════════════════════════════════════════════════
# Plan assembly
# ═══════════════════════════════════════════════════════════════════════════
def is_valid_plan(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    meals = obj.get("meals")
    return isinstance(meals, dict) and all(
        isinstance(meals.get(s), dict) and isinstance(meals[s].get("items"), list) and meals[s]["items"]
        and all(isinstance(i, str) and i.strip() for i in meals[s]["items"])
        for s in MEAL_TIMES
    )


def _snapshot(visit: dict | None) -> dict | None:
    if not visit:
        return None
    return {"visit_date": visit.get("visit_date"), "vitals": visit.get("vitals"), "blood": visit.get("blood"),
            "risk": (visit.get("predictions") or {}).get("overall_risk_score"),
            "arrhythmias_present": _present_codes(visit)}


def _fallback_plan(patient: dict, current: dict, previous: dict | None, deltas: dict | None) -> dict:
    pref = str((patient or {}).get("diet_preference") or "").strip().lower()
    rules_extra: list[str] = []
    if pref not in DIET_PREFERENCES:
        pref = "non_vegetarian"
        rules_extra.append("diet_preference_defaulted")
    cond = evaluate_conditions(patient, current)
    targets, exercise, rules = compute_targets(cond, pref, (patient or {}).get("gender"), cond.age)
    prev_plan = (previous or {}).get("diet_plan") if previous else None
    title_bits = []
    if cond.renal:
        title_bits.append("Renal-Cardiac")
    if cond.glucose_high or pref == "diabetic":
        title_bits.append("Low-GI Diabetic")
    if "sodium_restriction" in rules:
        title_bits.append("Low-Sodium")
    title = " ".join(title_bits + [pref.replace("_", "-").title(), "Heart-Healthy Plan"])
    return {
        "plan_title": title,
        "diet_preference": pref,
        "daily_targets": targets,
        "meals": fallback_meals(pref, cond.renal, cond),
        "foods_to_avoid": _foods_to_avoid(pref, cond),
        "exercise_plan": exercise,
        "why_plan_changed_summary": why_plan_changed(cond, targets, exercise, prev_plan, deltas, previous, current),
        "clinical_rules_applied": rules_extra + rules,
        "computed_from": {"current_report": _snapshot(current), "previous_report": _snapshot(previous)},
        "generated_by": "fallback",
    }


_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _quotes_same_numbers(llm_text: str, rule_text: str, min_share: float = 0.8) -> bool:
    """True when the LLM explanation repeats (most of) the exact numbers of the rule-based one."""
    wanted = {n.rstrip("0").rstrip(".") if "." in n else n for n in _NUM_RE.findall(rule_text)}
    if not wanted:
        return True
    got = {n.rstrip("0").rstrip(".") if "." in n else n for n in _NUM_RE.findall(llm_text)}
    return len(wanted & got) / len(wanted) >= min_share


def _llm_enhance(plan: dict, patient: dict, current: dict, previous: dict | None, deltas: dict | None) -> dict | None:
    from genai.llm_provider import complete_text

    pref = plan["diet_preference"]
    renal = "renal_restriction" in plan["clinical_rules_applied"]
    low_gi = "diabetic_low_gi_meals" in plan["clinical_rules_applied"] or pref == "diabetic"
    high_chol = (delta_engine._num((current.get("blood") or {}).get("cholesterol")) or 0) >= CHOL_HIGH
    forbidden = (EXCLUSION_LISTS[pref] + (RENAL_EXCLUSIONS if renal else []) + (EXCLUSION_LISTS["diabetic"] if low_gi else [])
                 + (SATFAT_EXCLUSIONS if high_chol else []))
    system = ("You are a hospital cardiac dietitian. Write culturally appropriate Indian/international meals that follow the "
              "numeric targets exactly and respect the diet preference in EVERY item. Reply with JSON only.")
    user = json.dumps({
        "task": ("Return {\"meals\": {breakfast|lunch|snack|dinner: {items: [...], notes: str}}, \"why_plan_changed_summary\": str}. "
                 "The summary (3-5 sentences, plain words) must cite the exact previous and current numbers and the previous visit "
                 "date given below, exactly like the rule_based_summary does. Never change any number."),
        "diet_preference": pref,
        "low_glycemic_required": low_gi,
        "low_potassium_required": renal,
        "forbidden_ingredients": forbidden,
        "daily_targets": plan["daily_targets"],
        "exercise_plan": plan["exercise_plan"],
        "rule_based_summary": plan["why_plan_changed_summary"],
        "current_report": plan["computed_from"]["current_report"],
        "previous_report": plan["computed_from"]["previous_report"],
        "previous_plan_targets": ((previous or {}).get("diet_plan") or {}).get("daily_targets"),
        "what_went_wrong": (deltas or {}).get("what_went_wrong_summary"),
        "patient_age": patient.get("age"),
    })
    text, provider = complete_text(system, user, json_mode=True, timeout=25.0, max_tokens=2500)
    data = json.loads(text)
    if not is_valid_plan({"meals": data.get("meals")}):
        return None
    enhanced = copy.deepcopy(plan)
    replaced = 0
    for slot in MEAL_TIMES:
        items = [str(i) for i in data["meals"][slot]["items"]]
        if all(_item_allowed(i, pref, renal, low_gi, high_chol) for i in items):
            enhanced["meals"][slot] = {"time": MEAL_TIMES[slot], "items": items,
                                       "notes": str(data["meals"][slot].get("notes") or enhanced["meals"][slot]["notes"])}
            replaced += 1
    summary = data.get("why_plan_changed_summary")
    if isinstance(summary, str) and summary.strip() and _quotes_same_numbers(summary, plan["why_plan_changed_summary"]):
        enhanced["rule_based_summary"] = plan["why_plan_changed_summary"]
        enhanced["why_plan_changed_summary"] = summary.strip()
    enhanced["generated_by"] = f"llm_{provider}"
    enhanced["llm_meal_slots_used"] = replaced
    return enhanced


def generate_diet_exercise_plan(patient_dict: dict, current_visit_dict: dict,
                                previous_visit_dict: dict | None, deltas_dict: dict | None) -> dict:
    """Complete diet & exercise plan; LLM-enhanced when a key is configured, else rule-based."""
    plan = _fallback_plan(patient_dict, current_visit_dict, previous_visit_dict, deltas_dict)
    try:
        from genai.llm_provider import configured_provider

        if configured_provider() == "none":
            return plan
        enhanced = _llm_enhance(plan, patient_dict, current_visit_dict, previous_visit_dict, deltas_dict)
        return enhanced if enhanced is not None else plan
    except Exception as exc:  # LLM down, rate-limited, bad JSON...
        logger.warning("LLM diet plan failed (%s); using rule-based plan", type(exc).__name__)
        plan["generated_by"] = "fallback"
        return plan
