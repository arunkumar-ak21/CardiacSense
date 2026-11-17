"""
Hypothesis strategies shared by the CODE-15% multimodal engine tests.

Large arrays are built with NumPy from a drawn seed instead of element by
element, so a 20000 x 12 signal costs one small draw and shrinks well.
``db.seed_and_samples`` (NumPy only, no torch) is imported lazily, inside the
strategies that synthesize ECGs.

Import with ``from strategies import raw_signals`` (``tests/`` is on
``sys.path`` via ``conftest.py``).
"""
from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
from hypothesis import strategies as st

N_LEADS = 12
N_CANONICAL = 4096
CODE15_CODES = ("1dAVb", "RBBB", "LBBB", "SB", "AF", "ST")

# Req 18.4 inclusive ranges for prediction inputs.
NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "hr": (20, 250),
    "bp_sys": (60, 260),
    "bp_dia": (30, 160),
    "spo2": (50, 100),
    "rr": (4, 60),
    "temp": (30, 43),
    "glucose": (20, 600),
    "cholesterol": (50, 500),
    "creatinine": (0.1, 15),
    "hrv": (0, 300),
    "sample_rate": (100, 2000),
}
VITAL_KEYS = ("hr", "bp_sys", "bp_dia", "spo2", "rr", "temp")
BLOOD_KEYS = ("glucose", "cholesterol", "creatinine", "hrv")

# Smallest float32 whose float64 value is >= 0.02 (float32(0.02) is just below 0.02).
_F32_MIN_AMP = float(np.nextafter(np.float32(0.02), np.float32(1.0)))
_F32_MAX_AMP = 50.0  # exactly representable in float32

_seeds = st.integers(min_value=0, max_value=2**32 - 1)


def _finite(lo: float, hi: float) -> st.SearchStrategy[float]:
    return st.floats(min_value=lo, max_value=hi, allow_nan=False, allow_infinity=False)


# ═══════════════════════════════════════════════════════════════════════════
# Signals
# ═══════════════════════════════════════════════════════════════════════════
def _synth_ecg(seed: int, heart_rate: float, af: bool, lbbb: bool, st_amp: float) -> np.ndarray:
    """A (4096, 12) float32 ECG in mV from ``synthesize_12lead_ecg``."""
    from db.seed_and_samples import ECGProfile, synthesize_12lead_ecg

    profile = ECGProfile(
        name="hypothesis",
        heart_rate=heart_rate,
        qrs_ms=140.0 if lbbb else 92.0,
        pr_ms=None if af else 160.0,
        rhythm="af" if af else "sinus",
        conduction="lbbb" if lbbb else "normal",
        st_amp_mv=st_amp,
        seed=seed,
    )
    return np.asarray(synthesize_12lead_ecg(profile)["signal"], dtype=np.float32)


@st.composite
def _base_waveform(draw: st.DrawFn, n: int, allow_zero: bool) -> np.ndarray:
    """An (n, 12) float64 waveform in mV of one drawn kind."""
    kinds = ["ecg", "noise", "sine", "constant"] + (["zero"] if allow_zero else [])
    kind = draw(st.sampled_from(kinds))
    seed = draw(_seeds)
    rng = np.random.default_rng(seed)

    if kind == "zero":
        return np.zeros((n, N_LEADS))
    if kind == "constant":
        level = draw(_finite(0.05, 5.0)) * draw(st.sampled_from([-1.0, 1.0]))
        return np.full((n, N_LEADS), level)
    if kind == "noise":
        return rng.normal(0.0, draw(_finite(0.01, 2.0)), size=(n, N_LEADS))
    if kind == "sine":
        t = np.arange(n)[:, None] / 400.0
        freq = rng.uniform(0.5, 40.0, size=N_LEADS)
        phase = rng.uniform(0.0, 2 * np.pi, size=N_LEADS)
        amp = rng.uniform(0.05, 2.0, size=N_LEADS)
        return amp * np.sin(2 * np.pi * freq * t + phase)

    ecg = _synth_ecg(
        seed % (2**31),
        heart_rate=draw(_finite(45.0, 150.0)),
        af=draw(st.booleans()),
        lbbb=draw(st.booleans()),
        st_amp=draw(st.sampled_from([0.0, -0.15, 0.15])),
    ).astype(np.float64)
    reps = -(-n // N_CANONICAL)  # ceil
    return np.tile(ecg, (reps, 1))[:n]


@st.composite
def raw_signals(
    draw: st.DrawFn,
    min_samples: int = 400,
    max_samples: int = 20000,
    orientation: str | None = None,
) -> np.ndarray:
    """
    Finite 12-lead Raw_Signals with ``min_samples``..``max_samples`` samples per
    lead, as ``(n, 12)`` (``orientation="samples_first"``) or ``(12, n)``
    (``"leads_first"``); either when ``orientation`` is None. Amplitudes cover the
    volt, millivolt and microvolt bands (Req 4.9, 4.10), dtype float32 or float64.
    """
    n = draw(st.integers(min_value=min_samples, max_value=max_samples))
    x = draw(_base_waveform(n, allow_zero=True))
    x = x * draw(st.sampled_from([1e-3, 1.0, 1e3]))
    x = x.astype(draw(st.sampled_from([np.float32, np.float64])))
    if orientation is None:
        orientation = draw(st.sampled_from(["samples_first", "leads_first"]))
    if orientation == "leads_first":
        x = np.ascontiguousarray(x.T)
    elif orientation != "samples_first":
        raise ValueError(f"unknown orientation {orientation!r}")
    return x


@st.composite
def canonical_signals(draw: st.DrawFn) -> np.ndarray:
    """
    Canonical_Signals: ``(4096, 12)`` float32 with max |x| in [0.02, 50]
    (checked in float64), so preprocessing applies no unit scaling (Req 4.17).
    """
    x = draw(_base_waveform(N_CANONICAL, allow_zero=False))
    target = draw(_finite(0.02, 50.0))
    peak = float(np.max(np.abs(x)))
    x = (x * (target / peak)).astype(np.float32)
    np.clip(x, -_F32_MAX_AMP, _F32_MAX_AMP, out=x)

    flat = np.abs(x.astype(np.float64)).ravel()
    i = int(np.argmax(flat))
    if flat[i] < 0.02:  # float32 rounding pushed the peak just under 0.02
        idx = np.unravel_index(i, x.shape)
        x[idx] = np.float32(np.copysign(_F32_MIN_AMP, x[idx] if x[idx] != 0 else 1.0))
    return x


# ═══════════════════════════════════════════════════════════════════════════
# DSP measurements
# ═══════════════════════════════════════════════════════════════════════════
def _st_status(st_amp: float) -> str:
    if st_amp > 0.1:
        return "Elevation"
    if st_amp < -0.05:
        return "Depression"
    return "Normal"


NOT_MEASURABLE: dict[str, Any] = {
    "heart_rate": 0.0,
    "qrs_ms": 0.0,
    "pr_ms": 0.0,
    "st_amp": 0.0,
    "st_status": "Normal",
    "rr_irregularity": 0.0,
    "measurable": False,
}


@st.composite
def dsp_dicts(draw: st.DrawFn, measurable: bool | None = None) -> dict[str, Any]:
    """
    Seven-key DSP_Measurements dicts (Req 6.1, 6.5, 6.6, 6.7). Measurable dicts
    hold clamped, rounded values with ``st_status`` derived from ``st_amp`` and
    ``pr_ms`` possibly None; unmeasurable dicts are the Req 6.6 defaults.
    """
    if measurable is None:
        measurable = draw(st.booleans())
    if not measurable:
        return dict(NOT_MEASURABLE)
    st_amp = round(draw(_finite(-2.0, 2.0)), 3)
    pr = draw(st.one_of(st.none(), _finite(0.0, 400.0)))
    return {
        "heart_rate": round(draw(_finite(0.0, 300.0)), 1),
        "qrs_ms": round(draw(_finite(0.0, 300.0)), 1),
        "pr_ms": None if pr is None else round(pr, 1),
        "st_amp": st_amp,
        "st_status": _st_status(st_amp),
        "rr_irregularity": round(draw(_finite(0.0, 1.5)), 3),
        "measurable": True,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Clinical values, flags and visits
# ═══════════════════════════════════════════════════════════════════════════
@st.composite
def clinical_values(draw: st.DrawFn) -> dict[str, Any]:
    """
    Dicts with the ``ClinicalVector`` fields (CLINICAL_FEATURES order) inside the
    Req 18.4 ranges, with ``bp_dia < bp_sys``, age 0-120, heart-age gap -102 to
    100, ``st_amp`` in [-2, 2] and 0-6 active arrhythmias.
    """
    bp_sys = draw(_finite(*NUMERIC_RANGES["bp_sys"]))
    bp_dia = draw(_finite(NUMERIC_RANGES["bp_dia"][0], min(NUMERIC_RANGES["bp_dia"][1], bp_sys)))
    if bp_dia >= bp_sys:
        bp_dia = float(np.nextafter(bp_sys, -np.inf))
    return {
        "age": draw(_finite(0.0, 120.0)),
        "is_male": draw(st.integers(min_value=0, max_value=1)),
        "bp_sys": bp_sys,
        "bp_dia": bp_dia,
        "glucose": draw(_finite(*NUMERIC_RANGES["glucose"])),
        "cholesterol": draw(_finite(*NUMERIC_RANGES["cholesterol"])),
        "creatinine": draw(_finite(*NUMERIC_RANGES["creatinine"])),
        "heart_age_gap": draw(_finite(-102.0, 100.0)),
        "st_amp": draw(_finite(-2.0, 2.0)),
        "active_arrhythmia_count": draw(st.integers(min_value=0, max_value=len(CODE15_CODES))),
    }


def code15_flag_maps(allow_missing: bool = False) -> st.SearchStrategy[dict[str, bool]]:
    """CODE15_Flags: code -> bool for the six codes (some codes may be absent when ``allow_missing``)."""
    if allow_missing:
        return st.fixed_dictionaries({}, optional={c: st.booleans() for c in CODE15_CODES})
    return st.fixed_dictionaries({c: st.booleans() for c in CODE15_CODES})


def _measure(key: str) -> st.SearchStrategy[float]:
    lo, hi = NUMERIC_RANGES[key]
    return st.one_of(st.integers(min_value=int(np.ceil(lo)), max_value=int(hi)).map(float), _finite(lo, hi))


def _section(keys: tuple[str, ...]) -> st.SearchStrategy[dict[str, Any]]:
    """A vitals/blood dict: each key absent, None, or a value in its Req 18.4 range."""
    return st.fixed_dictionaries(
        {}, optional={k: st.one_of(st.none(), _measure(k)) for k in keys}
    )


_iso_dates = st.dates(min_value=date(2020, 1, 1), max_value=date(2030, 12, 31)).map(date.isoformat)


@st.composite
def visit_dicts(draw: st.DrawFn) -> dict[str, Any]:
    """
    Visit dicts shaped like ``Visit.to_dict(include_signals=False)``, as consumed by
    ``delta_engine.compute_visit_deltas``. Risk score, heart age, arrhythmia
    probabilities and DSP measurements may each be missing; ``visit_date`` is
    usually ISO and occasionally not.
    """
    flags = draw(code15_flag_maps(allow_missing=True))
    probs = draw(st.one_of(
        st.none(),
        st.fixed_dictionaries({}, optional={c: _finite(0.0, 1.0).map(lambda p: round(p, 3)) for c in CODE15_CODES}),
    ))
    risk = draw(st.one_of(st.none(), _finite(0.0, 100.0).map(lambda r: round(r, 1))))
    heart_age = draw(st.one_of(st.none(), _finite(18.0, 100.0).map(lambda a: round(a, 1))))
    age = draw(st.integers(min_value=0, max_value=120))

    predictions: dict[str, Any] = {
        "overall_risk_score": risk,
        "predicted_heart_age": heart_age,
        "chronological_age": age,
    }
    if probs is not None:
        predictions["arrhythmias"] = probs
    if risk is not None:
        from db.delta_engine import severity_tier

        predictions["severity_tier"] = severity_tier(risk)

    dsp = draw(st.one_of(st.none(), dsp_dicts()))
    return {
        "id": "vis_" + draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=4, max_size=12)),
        "patient_id": "pat_hypothesis",
        "visit_date": draw(st.one_of(_iso_dates, _iso_dates, st.just("not-a-date"))),
        "hospital_name": "General Hospital",
        "adherence_notes": None,
        "vitals": draw(_section(VITAL_KEYS)),
        "blood": draw(_section(BLOOD_KEYS)),
        "code15_flags": flags,
        "ecg_metadata": {} if dsp is None else {"dsp_measurements": dsp},
        "predictions": predictions,
        "shap": [],
        "heatmap": None,
        "delta_summary": None,
        "diet_plan": {},
        "approved_by_staff": False,
        "staff_notes": None,
        "created_at": None,
    }
