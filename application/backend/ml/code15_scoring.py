"""
Pure-Python scoring helpers for the CODE-15% multimodal engine.

This module holds the deterministic parts of a prediction that need no tensors:

* the ECG heart-age DSP fallback and the heart-age resolution (Req 7);
* the CODE-15% arrhythmia calibration: DSP evidence, the noisy-OR with the
  user-entered flags and the bounded deep-feature term (Req 8);
* the Present_Arrhythmia set and the stored ``code15_flags`` (Req 8.11);
* the ``ClinicalVector`` used by the fusion module and the clinical risk formula;
* the attribution drivers stored in ``shap`` and the clinical/neural combination
  (Req 10.5–10.9, 10.11, 10.13, 11).

It imports no NumPy, SciPy or torch, so it is safe to import anywhere (including
before torch in ``main.py``) and its property tests run without the ML stack.

None of these outputs is clinically validated. The arrhythmia probabilities are
calibrated research heuristics, not trained classifiers.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from db.delta_engine import ARRHYTHMIA_THRESHOLD, CODE15_LABELS, _num

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════
CODE15_CODES: tuple[str, ...] = ("1dAVb", "RBBB", "LBBB", "SB", "AF", "ST")
if set(CODE15_CODES) != set(CODE15_LABELS):  # keep in step with the Delta Engine labels
    raise RuntimeError("CODE15_CODES does not match delta_engine.CODE15_LABELS")

# Normal reference values substituted for absent Clinical_Vector elements (Req 10.9).
REFERENCE_VALUES: dict[str, float | int] = {
    "bp_sys": 120.0,
    "bp_dia": 80.0,
    "glucose": 90.0,
    "cholesterol": 180.0,
    "creatinine": 1.0,
    "st_amp": 0.0,
    "heart_age_gap": 0.0,
    "active_arrhythmia_count": 0,
}

# Attribution features in their tie-break order (Req 11).
ATTRIBUTION_FEATURES: tuple[str, ...] = (
    "creatinine",
    "glucose",
    "cholesterol",
    "bp_sys",
    "bp_dia",
    "ecg_heart_age_gap",
    "AF",
    "LBBB",
    "st_depression",
)

HEART_AGE_MIN = 18.0
HEART_AGE_MAX = 100.0
HEART_AGE_ADJ_MIN = -10.0
HEART_AGE_ADJ_MAX = 15.0
SOURCE_RESNET = "resnet1d"
SOURCE_DSP = "dsp_fallback"

FLAG_PROB = 0.92        # q_flag for a checked CODE-15% flag
BASE_PROB = 0.03        # q_dsp with no DSP evidence
DEEP_LIMIT = 0.10       # largest deep-feature shift of a probability (Req 8.8)

_TRUE_STRINGS = frozenset({"true", "1", "yes", "on"})


# ═══════════════════════════════════════════════════════════════════════════
# Clinical vector
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ClinicalVector:
    """The 10-element Clinical_Vector, in ``multimodal_fusion.CLINICAL_FEATURES`` order."""

    age: float
    is_male: int
    bp_sys: float
    bp_dia: float
    glucose: float
    cholesterol: float
    creatinine: float
    heart_age_gap: float
    st_amp: float
    active_arrhythmia_count: int

    def as_list(self) -> list[float]:
        return [
            float(self.age),
            float(self.is_male),
            float(self.bp_sys),
            float(self.bp_dia),
            float(self.glucose),
            float(self.cholesterol),
            float(self.creatinine),
            float(self.heart_age_gap),
            float(self.st_amp),
            float(self.active_arrhythmia_count),
        ]


# ═══════════════════════════════════════════════════════════════════════════
# Small helpers
# ═══════════════════════════════════════════════════════════════════════════
def _finite(value: Any) -> float | None:
    """``value`` as a finite float, or None when missing, boolean, non-numeric or non-finite."""
    num = _num(value)
    if num is None or not math.isfinite(num):
        return None
    return num


def _clip(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def _ramp(value: float, a: float, b: float) -> float:
    """``clip((value - a) / (b - a), 0, 1)``."""
    return _clip((value - a) / (b - a), 0.0, 1.0)


def _capped(value: float, cap: float) -> float:
    return _clip(value, 0.0, cap)


def _is_measurable(dsp: Mapping) -> bool:
    """False only when the DSP dict reports ``measurable`` as false (Req 6.6, 8.13)."""
    flag = dsp.get("measurable", True)
    return True if flag is None else bool(flag)


def _st_category(dsp: Mapping) -> str:
    """First word of ``st_status`` (manifest values carry details such as ``Depression (-0.15 mV)``)."""
    status = dsp.get("st_status")
    if not isinstance(status, str) or not status.strip():
        return "Normal"
    return status.strip().split()[0].capitalize()


def _truthy(value: Any) -> bool:
    """A CODE15_Flag value as a bool: ``true/1/yes/on`` strings, non-zero numbers, ``True``."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0 and not (isinstance(value, float) and math.isnan(value))
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_STRINGS
    try:  # e.g. numpy.bool_
        return bool(value)
    except (TypeError, ValueError):
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Heart age (Req 7)
# ═══════════════════════════════════════════════════════════════════════════
def heart_age_dsp_adjustment(dsp: Mapping) -> float:
    """
    Years added to the chronological age by the DSP fallback (Req 7.2).

    0 when ``measurable`` is false. Otherwise the sum of the capped design terms
    (tachycardia, bradycardia, wide QRS, long or short PR, ST depression or
    elevation, irregular RR), clipped to [-10, 15]. Missing or non-finite fields
    add nothing, so a normal recording gives exactly 0.
    """
    if not _is_measurable(dsp):
        return 0.0
    hr = _finite(dsp.get("heart_rate"))
    qrs = _finite(dsp.get("qrs_ms"))
    pr = _finite(dsp.get("pr_ms"))
    st_amp = _finite(dsp.get("st_amp"))
    rr = _finite(dsp.get("rr_irregularity"))
    status = _st_category(dsp)

    adj = 0.0
    if hr is not None and hr > 100:
        adj += _capped(0.15 * (hr - 100.0), 5.0)
    if hr is not None and hr < 60:
        adj += _capped(0.10 * (60.0 - hr), 3.0)
    if qrs is not None and qrs >= 120:
        adj += _capped(0.20 * (qrs - 120.0), 6.0)
    if pr is not None and pr > 200:
        adj += _capped(0.05 * (pr - 200.0), 3.0)
    if pr is not None and pr < 120:
        adj += _capped(0.05 * (120.0 - pr), 1.0)
    if st_amp is not None and status == "Depression":
        adj += _capped(40.0 * (-st_amp - 0.05), 4.0)
    if st_amp is not None and status == "Elevation":
        adj += _capped(20.0 * (st_amp - 0.1), 2.0)
    if rr is not None and rr > 0.10:
        adj += _capped(20.0 * (rr - 0.10), 3.0)
    return _clip(adj, HEART_AGE_ADJ_MIN, HEART_AGE_ADJ_MAX)


def resolve_heart_age(
    model_out: float | None, age: float, dsp: Mapping, weights_loaded: bool
) -> tuple[float, float, str]:
    """
    ``(predicted_heart_age, heart_age_gap, heart_age_source)`` (Req 7.1–7.3, 7.5, 7.7).

    The ResNet1d output is used, clamped to [18, 100], only when the weights are
    loaded and the output is finite. Otherwise the DSP fallback
    ``age + heart_age_dsp_adjustment(dsp)`` is used. A non-finite model output with
    loaded weights is logged.
    """
    age_f = float(age)
    out = None if model_out is None else _num(model_out)

    if weights_loaded and out is not None and math.isfinite(out):
        predicted = round(_clip(out, HEART_AGE_MIN, HEART_AGE_MAX), 1)
        source = SOURCE_RESNET
    else:
        if weights_loaded:
            logger.warning(
                "ResNet1d heart-age output is not finite (%r); using the DSP fallback", model_out
            )
        adj = heart_age_dsp_adjustment(dsp)
        predicted = round(_clip(age_f + adj, HEART_AGE_MIN, HEART_AGE_MAX), 1)
        source = SOURCE_DSP
    gap = round(predicted - age_f, 1)
    return predicted, gap, source


# ═══════════════════════════════════════════════════════════════════════════
# Arrhythmia calibration (Req 8)
# ═══════════════════════════════════════════════════════════════════════════
def dsp_evidence(dsp: Mapping) -> dict[str, float]:
    """
    Per-code DSP probability ``q_dsp`` in [0.03, 0.98] (0.98 only for AF with a null PR).

    Every code is 0.03 when ``measurable`` is false (Req 8.13). A null or missing
    ``pr_ms`` gives no 1dAVb evidence (Req 8.12) and adds 0.15 to AF. A missing
    or non-finite measurement contributes no evidence.
    """
    if not _is_measurable(dsp):
        return {code: BASE_PROB for code in CODE15_CODES}

    hr = _finite(dsp.get("heart_rate"))
    qrs = _finite(dsp.get("qrs_ms"))
    pr = _finite(dsp.get("pr_ms"))
    rr = _finite(dsp.get("rr_irregularity"))

    q_1davb = BASE_PROB + 0.85 * _ramp(pr, 200.0, 240.0) if pr is not None else BASE_PROB
    q_bbb = BASE_PROB + 0.30 * _ramp(qrs, 110.0, 150.0) if qrs is not None else BASE_PROB
    q_sb = BASE_PROB + 0.85 * _ramp(60.0 - hr, 0.0, 10.0) if hr is not None and hr > 0 else BASE_PROB
    a = _ramp(rr, 0.15, 0.23) if rr is not None else 0.0
    q_af = BASE_PROB + (0.15 if pr is None else 0.0) + 0.80 * a
    q_st = BASE_PROB + 0.85 * _ramp(hr, 100.0, 115.0) * (1.0 - 0.6 * a) if hr is not None else BASE_PROB

    return {
        "1dAVb": q_1davb,
        "RBBB": q_bbb,
        "LBBB": q_bbb,
        "SB": q_sb,
        "AF": q_af,
        "ST": q_st,
    }


def _deep_term(deep_terms: Mapping[str, float] | None, code: str) -> float:
    """The deep-feature term ``t`` for ``code`` in [-1, 1]; 0 when absent or non-finite (Req 8.14)."""
    if not deep_terms:
        return 0.0
    t = _finite(deep_terms.get(code))
    return 0.0 if t is None else _clip(t, -1.0, 1.0)


def calibrate_arrhythmias(
    flags: Mapping[str, bool],
    dsp: Mapping,
    deep_terms: Mapping[str, float] | None,
    *,
    round_output: bool = True,
) -> dict[str, float]:
    """
    One probability in [0, 1] per CODE15_Code (Req 8.1–8.10, 8.12–8.14).

    ``p0 = 1 − (1 − q_flag)·(1 − q_dsp)`` is the noisy-OR of a checked flag (0.92)
    and the DSP evidence. The deep term shifts it by ``t·min(0.10, p0, 1 − p0)``,
    which keeps ``p`` in [0, 1], within 0.10 of ``p0`` and increasing in ``p0``.
    Pass ``deep_terms=None`` when the weights are not loaded or torch is missing.
    Values are rounded to 3 decimals unless ``round_output`` is False.
    """
    flags = flags or {}
    evidence = dsp_evidence(dsp or {})
    probs: dict[str, float] = {}
    for code in CODE15_CODES:
        q_flag = FLAG_PROB if _truthy(flags.get(code)) else 0.0
        p0 = 1.0 - (1.0 - q_flag) * (1.0 - evidence[code])
        t = _deep_term(deep_terms, code)
        p = _clip(p0 + t * min(DEEP_LIMIT, p0, 1.0 - p0), 0.0, 1.0)
        probs[code] = round(p, 3) if round_output else p
    return probs


def present_codes(flags: Mapping[str, bool], probs: Mapping[str, float]) -> frozenset[str]:
    """Present_Arrhythmias: codes with a true flag or a probability ≥ ARRHYTHMIA_THRESHOLD."""
    flags = flags or {}
    probs = probs or {}
    present = set()
    for code in CODE15_CODES:
        prob = _finite(probs.get(code))
        if _truthy(flags.get(code)) or (prob is not None and prob >= ARRHYTHMIA_THRESHOLD):
            present.add(code)
    return frozenset(present)


def build_code15_flags(flags: Mapping[str, Any], probs: Mapping[str, float]) -> dict:
    """
    The stored ``code15_flags``: the six submitted flags (missing → False) plus
    ``normal_ecg``, true only when no code is a Present_Arrhythmia (Req 8.11).
    """
    flags = flags or {}
    stored: dict[str, bool] = {code: _truthy(flags.get(code)) for code in CODE15_CODES}
    stored["normal_ecg"] = not present_codes(stored, probs)
    return stored


# ═══════════════════════════════════════════════════════════════════════════
# Clinical risk (Req 10.5, 10.8, 11.2, 11.5)
# ═══════════════════════════════════════════════════════════════════════════
NEURAL_LIMIT = 5.0           # |Neural_Risk_Adjustment| bound in points (Req 10.5)
# Calibration: every feature/arrhythmia contribution is scaled by RISK_SCALE and the
# healthy baseline starts at BASE_CONST. Tuned so the seeded demo inputs land near the
# seeded scores (Ramesh Visit 1 ~34%, Visit 2 ~75%).
RISK_SCALE = 0.6
BASE_CONST = 18.0
ARRHYTHMIA_POINTS = 5.0      # points per Present_Arrhythmia
FLAGGED_CODE_BONUS = 5.0     # extra points for AF and for LBBB
ATTRIBUTED_CODES: tuple[str, ...] = ("AF", "LBBB")

# Clinical_Vector element names in ``multimodal_fusion.CLINICAL_FEATURES`` order.
_CV_FIELDS: tuple[str, ...] = (
    "age", "is_male", "bp_sys", "bp_dia", "glucose", "cholesterol",
    "creatinine", "heart_age_gap", "st_amp", "active_arrhythmia_count",
)
# Elements taken from the request ``values`` mapping (Req 10.13).
_REQUEST_FIELDS: tuple[str, ...] = ("bp_sys", "bp_dia", "glucose", "cholesterol", "creatinine")

# feature -> (Clinical_Vector field, display label, unit) for the numeric drivers (Req 11.7, 11.8).
_NUMERIC_DRIVERS: dict[str, tuple[str, str, str]] = {
    "creatinine": ("creatinine", "Serum Creatinine", "mg/dL"),
    "glucose": ("glucose", "Fasting Glucose", "mg/dL"),
    "cholesterol": ("cholesterol", "Total Cholesterol", "mg/dL"),
    "bp_sys": ("bp_sys", "Systolic BP", "mmHg"),
    "bp_dia": ("bp_dia", "Diastolic BP", "mmHg"),
    "ecg_heart_age_gap": ("heart_age_gap", "ECG Heart Age Gap", "years"),
    "st_depression": ("st_amp", "ST Depression", "mV"),
}
_CODE_LABELS: dict[str, str] = {"AF": "Atrial Fibrillation", "LBBB": "Left Bundle Branch Block"}
MAX_DRIVERS = 7


def lin(x: float, ref: float, up: float, down: float, lo: float, hi: float) -> float:
    """
    Piecewise-linear risk term: ``clip(up·(x − ref), 0, hi)`` for ``x ≥ ref``,
    otherwise ``clip(down·(x − ref), lo, 0)``. Non-decreasing in ``x``, 0 at ``x = ref``.
    """
    d = float(x) - float(ref)
    if d >= 0:
        return _clip(up * d, 0.0, hi)
    return _clip(down * d, lo, 0.0)


def base(age: float, is_male: int) -> float:
    """``18 + min(21, 0.30·max(0, age − 30)) + 3·is_male``: the healthy reference profile's risk."""
    return BASE_CONST + min(21.0, 0.30 * max(0.0, float(age) - 30.0)) + 3.0 * (1.0 if is_male else 0.0)


def _code_points(present: frozenset[str]) -> dict[str, float]:
    """Attributed points of AF and LBBB: 10 each when present (5 count + 5 bonus)."""
    present = present or frozenset()
    return {
        code: (ARRHYTHMIA_POINTS + FLAGGED_CODE_BONUS) if code in present else 0.0
        for code in ATTRIBUTED_CODES
    }


def feature_contributions(cv: ClinicalVector, present: frozenset[str]) -> dict[str, float]:
    """
    Unrounded attribution, in risk-score points, of each of the nine features in
    ``ATTRIBUTION_FEATURES`` order, relative to the healthy reference profile (Req 11.1, 11.2).
    """
    codes = _code_points(present)
    terms = {
        "creatinine": lin(cv.creatinine, 1.0, up=15.0, down=5.0, lo=-2.0, hi=22.5),
        "glucose": lin(cv.glucose, 90.0, up=0.12, down=0.05, lo=-1.0, hi=18.0),
        "cholesterol": lin(cv.cholesterol, 180.0, up=0.08, down=0.04, lo=-2.0, hi=12.0),
        "bp_sys": lin(cv.bp_sys, 120.0, up=0.30, down=0.10, lo=-3.0, hi=24.0),
        "bp_dia": lin(cv.bp_dia, 80.0, up=0.25, down=0.10, lo=-2.0, hi=10.0),
        "ecg_heart_age_gap": lin(cv.heart_age_gap, 0.0, up=0.80, down=0.30, lo=-3.0, hi=12.0),
        "AF": codes["AF"],
        "LBBB": codes["LBBB"],
        "st_depression": 50.0 * _clip(-float(cv.st_amp), 0.0, 0.4),
    }
    return {feature: RISK_SCALE * terms[feature] for feature in ATTRIBUTION_FEATURES}


def clinical_risk(cv: ClinicalVector, present: frozenset[str]) -> float:
    """
    The unclamped Clinical_Risk_Component (Req 10.5, 10.8):
    ``base + f_bp_sys + f_bp_dia + f_glucose + f_chol + f_creat + f_gap + f_st + f_arr``
    with ``f_arr = 5·active_arrhythmia_count + 5·[AF present] + 5·[LBBB present]``.
    """
    present = present or frozenset()
    contrib = feature_contributions(cv, present)
    f_arr = RISK_SCALE * (ARRHYTHMIA_POINTS * float(cv.active_arrhythmia_count) + sum(
        FLAGGED_CODE_BONUS for code in ATTRIBUTED_CODES if code in present
    ))
    continuous = sum(v for k, v in contrib.items() if k not in ATTRIBUTED_CODES)
    return base(cv.age, cv.is_male) + continuous + f_arr


# ═══════════════════════════════════════════════════════════════════════════
# Attribution drivers (Req 11.3–11.8)
# ═══════════════════════════════════════════════════════════════════════════
def _fmt(value: float) -> str:
    """Shortest form of a 2-decimal value: ``1.7``, ``156``, ``-0.15``, ``0``."""
    text = f"{round(value, 2) + 0.0:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def attribution_drivers(cv: ClinicalVector, present: frozenset[str]) -> list[dict]:
    """
    The stored ``shap`` list: ``{feature, display_name, value, shap_score, direction}``
    for each feature whose ``round(points/100, 3)`` is not 0.0, sorted by
    descending ``|shap_score|`` then ``ATTRIBUTION_FEATURES`` order, at most 7.
    """
    present = present or frozenset()
    contrib = feature_contributions(cv, present)
    drivers: list[tuple[int, dict]] = []
    for index, feature in enumerate(ATTRIBUTION_FEATURES):
        score = round(contrib[feature] / 100.0, 3) + 0.0
        if score == 0.0:
            continue
        if feature in _CODE_LABELS:
            is_present = feature in present
            value = 1.0 if is_present else 0.0
            display = f"{_CODE_LABELS[feature]} ({'Present' if is_present else 'Absent'})"
        else:
            field, label, unit = _NUMERIC_DRIVERS[feature]
            value = round(float(getattr(cv, field)), 2) + 0.0
            display = f"{label} ({_fmt(value)} {unit})"
        drivers.append((index, {
            "feature": feature,
            "display_name": display,
            "value": value,
            "shap_score": score,
            "direction": "positive" if score >= 0 else "negative",
        }))
    drivers.sort(key=lambda item: (-abs(item[1]["shap_score"]), item[0]))
    return [driver for _, driver in drivers[:MAX_DRIVERS]]


# ═══════════════════════════════════════════════════════════════════════════
# Combination (Req 10.5, 10.11)
# ═══════════════════════════════════════════════════════════════════════════
def combine_risk(clinical: float, neural: float | None) -> tuple[float, float, float]:
    """
    ``(overall_risk_score, clinical_risk_component, neural_risk_adjustment)``.

    The neural adjustment is clipped to [-5, 5] (0 when missing or non-finite);
    the overall score is ``round(clip(clinical + neural, 0, 100), 1)`` and both
    components are rounded to 1 decimal place.
    """
    clinical_f = float(clinical)
    neural_f = _finite(neural)
    neural_f = 0.0 if neural_f is None else _clip(neural_f, -NEURAL_LIMIT, NEURAL_LIMIT)
    overall = round(_clip(clinical_f + neural_f, 0.0, 100.0), 1)
    return overall, round(clinical_f, 1) + 0.0, round(neural_f, 1) + 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Clinical vector assembly (Req 10.9, 10.13)
# ═══════════════════════════════════════════════════════════════════════════
def build_clinical_vector(
    patient: Mapping,
    values: Mapping[str, float | None],
    heart_age_gap: float | None,
    st_amp: float | None,
    present: frozenset[str],
) -> tuple[ClinicalVector, list[str]]:
    """
    ``(ClinicalVector, imputed_fields)``.

    ``age`` comes from the patient record and ``is_male`` is 1 only for gender
    ``"M"``. The five blood/BP elements come from ``values``; ``heart_age_gap``
    and ``st_amp`` are passed in. Any of those seven that is absent, null or
    non-finite is replaced by its ``REFERENCE_VALUES`` entry and listed in
    ``imputed_fields`` (Clinical_Vector order). ``active_arrhythmia_count`` is
    ``len(present)``.
    """
    patient = patient or {}
    values = values or {}
    age = _finite(patient.get("age"))
    if age is None:
        raise ValueError("patient record has no numeric age")

    supplied: dict[str, float | None] = {key: _finite(values.get(key)) for key in _REQUEST_FIELDS}
    supplied["heart_age_gap"] = _finite(heart_age_gap)
    supplied["st_amp"] = _finite(st_amp)

    imputed: list[str] = []
    resolved: dict[str, float] = {}
    for key in _CV_FIELDS:
        if key not in supplied:
            continue
        if supplied[key] is None:
            resolved[key] = float(REFERENCE_VALUES[key])
            imputed.append(key)
        else:
            resolved[key] = float(supplied[key])

    cv = ClinicalVector(
        age=age,
        is_male=1 if patient.get("gender") == "M" else 0,
        bp_sys=resolved["bp_sys"],
        bp_dia=resolved["bp_dia"],
        glucose=resolved["glucose"],
        cholesterol=resolved["cholesterol"],
        creatinine=resolved["creatinine"],
        heart_age_gap=resolved["heart_age_gap"],
        st_amp=resolved["st_amp"],
        active_arrhythmia_count=len(present or ()),
    )
    return cv, imputed
