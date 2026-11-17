"""Properties 44 and 45: diet plan meal exclusions and rule triggers (Req 13.2-13.9, 13.16-13.19)."""
from __future__ import annotations

import json
import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from genai import diet_planner as dp
from genai import llm_provider
from strategies import visit_dicts

PREFERENCES = st.one_of(
    st.sampled_from(dp.DIET_PREFERENCES),
    st.sampled_from(["VEGAN ", "Dash", "keto", "", None, "pescatarian"]),
)


@st.composite
def plan_inputs(draw):
    patient = {
        "id": "pat_hypothesis",
        "name": "Test Patient",
        "age": draw(st.one_of(st.none(), st.integers(min_value=18, max_value=110))),
        "gender": draw(st.sampled_from(["M", "F", None])),
        "diet_preference": draw(PREFERENCES),
    }
    return patient, draw(visit_dicts())


def _num(visit, section, key):
    v = (visit.get(section) or {}).get(key)
    return None if v is None else float(v)


def _pref(patient) -> tuple[str, bool]:
    pref = str(patient.get("diet_preference") or "").strip().lower()
    return (pref, False) if pref in dp.DIET_PREFERENCES else ("non_vegetarian", True)


def _present(visit) -> set[str]:
    flags = visit.get("code15_flags") or {}
    probs = (visit.get("predictions") or {}).get("arrhythmias") or {}
    return {c for c in ("1dAVb", "RBBB", "LBBB", "SB", "AF", "ST") if flags.get(c) or (probs.get(c) or 0) >= 0.5}


def _assert_meals_respect_exclusions(plan: dict, renal: bool) -> None:
    pref = plan["diet_preference"]
    for slot in dp.MEAL_TIMES:
        items = plan["meals"][slot]["items"]
        assert items, slot
        for item in items:
            assert not dp.find_excluded(item, dp.EXCLUSION_LISTS[pref], vegan=pref == "vegan"), (pref, item)
            if renal:
                assert not dp.find_excluded(item, dp.RENAL_EXCLUSIONS), item


@pytest.fixture
def no_llm(monkeypatch):
    monkeypatch.setattr(llm_provider, "configured_provider", lambda *a, **k: "none")


# â”€â”€ Property 44: fallback plans â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
@given(plan_inputs())
def test_fallback_meals_respect_exclusions(no_llm, inputs):
    patient, visit = inputs
    plan = dp.generate_diet_exercise_plan(patient, visit, None, None)
    assert plan["generated_by"] == "fallback"
    _assert_meals_respect_exclusions(plan, (_num(visit, "blood", "creatinine") or 0) >= dp.CREAT_RENAL)


# â”€â”€ Property 44: LLM-post-processed plans â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
LLM_ITEMS = [
    "Grilled chicken salad", "Boiled eggs", "Paneer tikka", "Banana smoothie", "Potato curry",
    "Tomato soup", "Oats with soy milk", "Cheese toast", "Sugar-free jelly", "Moong dal",
    "Cucumber salad", "Brown rice", "Salmon fillet", "Almond milk porridge", "Pickles",
    "Spinach dal", "Tofu stir-fry", "Coconut water", "Jaggery laddoo", "Steamed cabbage",
]
llm_meals = st.fixed_dictionaries({
    slot: st.fixed_dictionaries({
        "items": st.lists(st.sampled_from(LLM_ITEMS), min_size=1, max_size=4),
        "notes": st.just("LLM note"),
    })
    for slot in dp.MEAL_TIMES
})


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
@given(plan_inputs(), llm_meals)
def test_llm_meals_respect_exclusions(monkeypatch, inputs, meals):
    patient, visit = inputs
    reply = json.dumps({"meals": meals, "why_plan_changed_summary": "LLM summary"})
    monkeypatch.setattr(llm_provider, "configured_provider", lambda *a, **k: "gemini")
    monkeypatch.setattr(llm_provider, "complete_text", lambda *a, **k: (reply, "gemini"))

    plan = dp.generate_diet_exercise_plan(patient, visit, None, None)
    assert plan["generated_by"] == "llm_gemini"
    _assert_meals_respect_exclusions(plan, (_num(visit, "blood", "creatinine") or 0) >= dp.CREAT_RENAL)


# â”€â”€ Property 45: rule triggers and rule listing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow])
@given(plan_inputs())
def test_rule_triggers_and_listing(no_llm, inputs):
    patient, visit = inputs
    plan = dp.generate_diet_exercise_plan(patient, visit, None, None)
    rules = set(plan["clinical_rules_applied"])
    targets = plan["daily_targets"]
    exercise = plan["exercise_plan"]

    bp_sys, bp_dia = _num(visit, "vitals", "bp_sys"), _num(visit, "vitals", "bp_dia")
    creat = _num(visit, "blood", "creatinine")
    glucose = _num(visit, "blood", "glucose")
    chol = _num(visit, "blood", "cholesterol")
    risk = (visit.get("predictions") or {}).get("overall_risk_score")
    age = patient["age"]
    present = _present(visit)

    # Diet preference
    pref, defaulted = _pref(patient)
    assert plan["diet_preference"] == pref
    assert ("diet_preference_defaulted" in rules) == defaulted

    # Sodium. Restriction also triggers on cholesterol >= 240 (dynamic-plan extension
    # agreed after the original Req 13.4/13.5 wording).
    restrict = ((bp_sys or 0) >= 130 or (bp_dia or 0) >= 80 or (risk or 0) >= 50 or (chol or 0) >= 240)
    assert ("sodium_restriction" in rules) == restrict
    assert ("sodium_standard" in rules) == (not restrict)
    if restrict:
        assert 1000 <= targets["sodium_mg"] <= 1400
    else:
        assert 1500 <= targets["sodium_mg"] <= 2300

    # Saturated fat and glucose extensions
    assert ("saturated_fat_limit" in rules) == ((chol or 0) >= 200)
    assert ("diabetic_low_gi_meals" in rules) == ((glucose or 0) >= 140)

    # Renal
    renal = (creat or 0) >= 1.4
    assert ("renal_restriction" in rules) == renal
    if renal:
        assert targets["fluid_liters"] <= 1.5
        assert targets["protein_g"] <= 50
        assert targets["potassium_mg"] <= 2000

    # Metabolic
    metabolic = (glucose or 0) >= 100 or (chol or 0) >= 200
    assert ("soluble_fiber" in rules) == metabolic
    assert ("low_glycemic_index" in rules) == metabolic
    if metabolic:
        assert targets["fiber_g"] >= 30
        assert any("sugar" in entry.lower() for entry in plan["foods_to_avoid"])

    # Exercise
    cardiac = risk is None or age is None or risk >= 50 or bool(present & {"AF", "LBBB", "ST"})
    assert ("cardiac_safe_exercise" in rules) == cardiac
    assert ("standard_exercise_ceiling" in rules) == (not cardiac)
    ceiling = exercise["max_hr_ceiling"]
    if cardiac:
        assert exercise["intensity"] in ("light", "very_light")
        limit = 110 if age is None else min(110, math.floor(0.6 * (220 - age)))
        assert ceiling <= limit
        assert any("HIIT" in w for w in exercise["warnings"])
    else:
        assert ceiling <= math.floor(0.75 * (220 - age))

    # DASH potassium
    if pref == "dash" and not renal:
        assert targets["potassium_mg"] >= 3500
        assert "dash_potassium" in rules
    if pref == "dash" and renal:
        assert "renal_potassium_override" in rules
        assert "dash_potassium" not in rules

    # Missing inputs
    values = {"bp_sys": bp_sys, "bp_dia": bp_dia, "creatinine": creat, "glucose": glucose, "cholesterol": chol}
    for field, value in values.items():
        assert (f"missing_input_{field}" in rules) == (value is None)
