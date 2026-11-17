"""
Longitudinal "Before vs. Now" Delta Engine.

``compute_visit_deltas(previous, current)`` compares two visits and returns the
structure stored in ``visits.delta_summary_json``:

* ``risk_delta`` / ``heart_age_delta``: current minus previous (percentage points / years)
* ``biomarker_deltas``: per-metric previous, current, delta, % change and status
  (``worsened`` / ``improved`` / ``stable``)
* ``ecg_morphology_deltas``: DSP interval and ST-segment changes
* ``new_arrhythmias`` / ``resolved_arrhythmias``: CODE-15% findings that appeared or cleared
* ``what_went_wrong_summary`` / ``what_improved_summary``: plain-language explanations

Inputs may be ``Visit`` ORM objects or dicts shaped like ``Visit.to_dict()``. The module
is pure Python (no numpy/scipy) so it is safe to import anywhere.
"""
from __future__ import annotations

from datetime import date
from typing import Any

CODE15_LABELS: dict[str, str] = {
    "1dAVb": "1st Degree AV Block",
    "RBBB": "Right Bundle Branch Block",
    "LBBB": "Left Bundle Branch Block",
    "SB": "Sinus Bradycardia",
    "AF": "Atrial Fibrillation",
    "ST": "Sinus Tachycardia",
}

ARRHYTHMIA_THRESHOLD = 0.5   # probability at/above which a finding counts as present
STABLE_PCT = 3.0             # |% change| below this is "stable"
RISK_TREND_POINTS = 5.0      # |risk delta| below this keeps the overall trend "stable"

# key, source section, label, unit, better ("lower" | "higher" | "range"), normal (low, high), decimals
METRICS: list[tuple[str, str, str, str, str, tuple[float, float], int]] = [
    ("bp_sys", "vitals", "Systolic BP", "mmHg", "lower", (90, 120), 0),
    ("bp_dia", "vitals", "Diastolic BP", "mmHg", "lower", (60, 80), 0),
    ("hr", "vitals", "Heart Rate", "bpm", "range", (60, 100), 0),
    ("spo2", "vitals", "SpO2", "%", "higher", (95, 100), 0),
    ("rr", "vitals", "Respiratory Rate", "/min", "range", (12, 20), 0),
    ("temp", "vitals", "Temperature", "°C", "range", (36.1, 37.5), 1),
    ("glucose", "blood", "Fasting Glucose", "mg/dL", "lower", (70, 100), 0),
    ("cholesterol", "blood", "Total Cholesterol", "mg/dL", "lower", (125, 200), 0),
    ("creatinine", "blood", "Serum Creatinine", "mg/dL", "lower", (0.6, 1.2), 2),
    ("hrv", "blood", "Heart Rate Variability", "ms", "higher", (40, 100), 0),
]


def severity_tier(risk_score: float | None) -> str | None:
    """Map a 0-100 risk percentage to the blueprint's severity tiers."""
    if risk_score is None:
        return None
    if risk_score < 25:
        return "Low"
    if risk_score < 50:
        return "Moderate"
    if risk_score < 75:
        return "High"
    return "Critical"


def _clinical_note(key: str, value: float | None) -> str | None:
    if value is None:
        return None
    if key == "creatinine" and value > 1.3:
        return "possible kidney strain"
    if key == "glucose" and value >= 126:
        return "diabetic range"
    if key == "glucose" and value >= 100:
        return "prediabetic range"
    if key == "cholesterol" and value >= 240:
        return "high"
    if key == "cholesterol" and value >= 200:
        return "borderline high"
    if key == "bp_sys" and value >= 140:
        return "stage 2 hypertension range"
    if key == "bp_sys" and value >= 130:
        return "stage 1 hypertension range"
    if key == "spo2" and value < 95:
        return "low oxygen saturation"
    if key == "hr" and value > 100:
        return "tachycardia"
    if key == "hr" and value < 60:
        return "bradycardia"
    if key == "hrv" and value < 30:
        return "low variability"
    return None


def _as_dict(visit: Any) -> dict[str, Any] | None:
    if visit is None:
        return None
    if isinstance(visit, dict):
        return visit
    if hasattr(visit, "to_dict"):
        return visit.to_dict(include_signals=False)
    raise TypeError(f"Unsupported visit type: {type(visit)!r}")


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: float | None, decimals: int) -> float | None:
    if value is None:
        return None
    rounded = round(value, decimals)
    return int(rounded) if decimals == 0 else rounded


def _fmt(value: float | None, decimals: int) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}"


def _signed(value: float, decimals: int) -> str:
    return f"{value:+.{decimals}f}"


def _distance_from_range(value: float, low: float, high: float) -> float:
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def _metric_status(better: str, prev: float, curr: float, normal: tuple[float, float]) -> str:
    delta = curr - prev
    pct = abs(delta) / abs(prev) * 100.0 if prev != 0 else (0.0 if delta == 0 else 100.0)
    if pct < STABLE_PCT:
        return "stable"
    if better == "lower":
        return "worsened" if delta > 0 else "improved"
    if better == "higher":
        return "worsened" if delta < 0 else "improved"
    d_prev = _distance_from_range(prev, *normal)
    d_curr = _distance_from_range(curr, *normal)
    if d_curr > d_prev:
        return "worsened"
    if d_curr < d_prev:
        return "improved"
    return "stable"


def _biomarker_deltas(prev: dict[str, Any], curr: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for key, section, label, unit, better, normal, decimals in METRICS:
        p = _num((prev.get(section) or {}).get(key))
        c = _num((curr.get(section) or {}).get(key))
        if p is None or c is None:
            continue
        delta = c - p
        pct = (delta / abs(p) * 100.0) if p != 0 else None
        status = _metric_status(better, p, c, normal)
        note = _clinical_note(key, c)
        display = (
            f"{label}: {_fmt(p, decimals)} → {_fmt(c, decimals)} {unit} "
            f"({_signed(delta, decimals)} {unit})"
        )
        results.append(
            {
                "key": key,
                "category": section,
                "label": label,
                "unit": unit,
                "previous": _round(p, decimals),
                "current": _round(c, decimals),
                "delta": _round(delta, decimals),
                "delta_pct": round(pct, 1) if pct is not None else None,
                "status": status,
                "normal_range": list(normal),
                "clinical_note": note,
                "display": display,
            }
        )
    return results


def _arrhythmia_probs(visit: dict[str, Any]) -> dict[str, float]:
    preds = visit.get("predictions") or {}
    probs = {k: _num(v) for k, v in (preds.get("arrhythmias") or {}).items()}
    flags = visit.get("code15_flags") or {}
    out: dict[str, float] = {}
    for code in CODE15_LABELS:
        prob = probs.get(code)
        if prob is None and code in flags:
            prob = 1.0 if flags.get(code) else 0.0
        out[code] = float(prob) if prob is not None else 0.0
    return out


def _present(visit: dict[str, Any], code: str, probs: dict[str, float]) -> bool:
    flags = visit.get("code15_flags") or {}
    return bool(flags.get(code)) or probs.get(code, 0.0) >= ARRHYTHMIA_THRESHOLD


def _arrhythmia_changes(prev: dict[str, Any], curr: dict[str, Any]) -> tuple[list, list, list]:
    p_probs = _arrhythmia_probs(prev)
    c_probs = _arrhythmia_probs(curr)
    new, resolved, persistent = [], [], []
    for code, name in CODE15_LABELS.items():
        was = _present(prev, code, p_probs)
        now = _present(curr, code, c_probs)
        entry = {
            "code": code,
            "name": name,
            "previous_prob": round(p_probs[code], 3),
            "current_prob": round(c_probs[code], 3),
        }
        if now and not was:
            new.append(entry)
        elif was and not now:
            resolved.append(entry)
        elif was and now:
            persistent.append(entry)
    return new, resolved, persistent


def _dsp(visit: dict[str, Any]) -> dict[str, Any]:
    return ((visit.get("ecg_metadata") or {}).get("dsp_measurements")) or {}


def _ecg_morphology_deltas(prev: dict[str, Any], curr: dict[str, Any]) -> list[dict[str, Any]]:
    p, c = _dsp(prev), _dsp(curr)
    out: list[dict[str, Any]] = []
    for key, label, unit, normal, decimals in (
        ("heart_rate", "ECG Heart Rate", "bpm", (60, 100), 0),
        ("qrs_ms", "QRS Duration", "ms", (80, 120), 0),
        ("pr_ms", "PR Interval", "ms", (120, 200), 0),
        ("st_amp", "ST Deviation", "mV", (-0.05, 0.1), 2),
    ):
        pv, cv = _num(p.get(key)), _num(c.get(key))
        if pv is None and cv is None:
            continue
        entry: dict[str, Any] = {
            "key": key,
            "label": label,
            "unit": unit,
            "previous": _round(pv, decimals),
            "current": _round(cv, decimals),
            "delta": _round(cv - pv, decimals) if pv is not None and cv is not None else None,
        }
        if pv is not None and cv is not None:
            d_prev = _distance_from_range(pv, *normal)
            d_curr = _distance_from_range(cv, *normal)
            if abs(cv - pv) < (0.02 if key == "st_amp" else 0.03 * max(abs(pv), 1.0)):
                entry["status"] = "stable"
            elif d_curr > d_prev:
                entry["status"] = "worsened"
            elif d_curr < d_prev:
                entry["status"] = "improved"
            else:
                entry["status"] = "stable"
        else:
            entry["status"] = "not_measurable"
        out.append(entry)

    if p.get("st_status") or c.get("st_status"):
        out.append(
            {
                "key": "st_status",
                "label": "ST Segment",
                "unit": None,
                "previous": p.get("st_status"),
                "current": c.get("st_status"),
                "delta": None,
                "status": "stable" if p.get("st_status") == c.get("st_status") else "changed",
            }
        )
    return out


def _days_between(d1: Any, d2: Any) -> int | None:
    try:
        return (date.fromisoformat(str(d2)[:10]) - date.fromisoformat(str(d1)[:10])).days
    except ValueError:
        return None


def _join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _baseline(curr: dict[str, Any]) -> dict[str, Any]:
    preds = curr.get("predictions") or {}
    risk = _num(preds.get("overall_risk_score"))
    heart_age = _num(preds.get("predicted_heart_age"))
    chrono = _num(preds.get("chronological_age"))
    tier = preds.get("severity_tier") or severity_tier(risk)
    gap = round(heart_age - chrono, 1) if heart_age is not None and chrono is not None else None
    return {
        "is_baseline": True,
        "previous_visit_id": None,
        "previous_visit_date": None,
        "current_visit_date": curr.get("visit_date"),
        "days_between_visits": None,
        "previous_risk_score": None,
        "current_risk_score": risk,
        "previous_severity_tier": None,
        "current_severity_tier": tier,
        "risk_delta": None,
        "previous_heart_age": None,
        "current_heart_age": heart_age,
        "heart_age_delta": None,
        "heart_age_gap_current": gap,
        "overall_trend": "baseline",
        "biomarker_deltas": [],
        "worsened": [],
        "improved": [],
        "stable": [],
        "ecg_morphology_deltas": [],
        "new_arrhythmias": [],
        "resolved_arrhythmias": [],
        "persistent_arrhythmias": [],
        "what_went_wrong_summary": (
            "Baseline visit: there is no earlier visit to compare against. "
            f"Current cardiac risk is {_fmt(risk, 1)}% ({tier})."
        ),
        "what_improved_summary": "Baseline visit: improvements will be tracked from the next visit onward.",
    }


def compute_visit_deltas(previous: Any, current: Any) -> dict[str, Any]:
    """Compare ``current`` against ``previous`` (or build a baseline when ``previous`` is None)."""
    curr = _as_dict(current)
    if curr is None:
        raise ValueError("current visit is required")
    prev = _as_dict(previous)
    if prev is None:
        return _baseline(curr)

    p_pred, c_pred = prev.get("predictions") or {}, curr.get("predictions") or {}
    p_risk, c_risk = _num(p_pred.get("overall_risk_score")), _num(c_pred.get("overall_risk_score"))
    p_age, c_age = _num(p_pred.get("predicted_heart_age")), _num(c_pred.get("predicted_heart_age"))
    chrono = _num(c_pred.get("chronological_age"))
    p_tier = p_pred.get("severity_tier") or severity_tier(p_risk)
    c_tier = c_pred.get("severity_tier") or severity_tier(c_risk)

    risk_delta = round(c_risk - p_risk, 1) if p_risk is not None and c_risk is not None else None
    age_delta = round(c_age - p_age, 1) if p_age is not None and c_age is not None else None
    age_gap = round(c_age - chrono, 1) if c_age is not None and chrono is not None else None

    biomarkers = _biomarker_deltas(prev, curr)
    worsened = [b for b in biomarkers if b["status"] == "worsened"]
    improved = [b for b in biomarkers if b["status"] == "improved"]
    stable = [b for b in biomarkers if b["status"] == "stable"]
    ecg_deltas = _ecg_morphology_deltas(prev, curr)
    new_arr, resolved_arr, persistent_arr = _arrhythmia_changes(prev, curr)
    days = _days_between(prev.get("visit_date"), curr.get("visit_date"))

    if risk_delta is None:
        trend = "unknown"
    elif risk_delta >= RISK_TREND_POINTS:
        trend = "worsened"
    elif risk_delta <= -RISK_TREND_POINTS:
        trend = "improved"
    else:
        trend = "stable"

    since = f"since the visit on {prev.get('visit_date')}" + (f" ({days} days ago)" if days is not None else "")

    def _bio_phrase(b: dict[str, Any]) -> str:
        decimals = 2 if b["key"] == "creatinine" else (1 if b["key"] == "temp" else 0)
        text = (
            f"{b['label']} {_fmt(b['previous'], decimals)} → {_fmt(b['current'], decimals)} {b['unit']} "
            f"({_signed(b['delta'], decimals)})"
        )
        return text + (f", {b['clinical_note']}" if b["clinical_note"] else "")

    ecg_worse = [e for e in ecg_deltas if e["status"] == "worsened"]
    ecg_better = [e for e in ecg_deltas if e["status"] == "improved"]

    def _ecg_phrase(e: dict[str, Any]) -> str:
        decimals = 2 if e["key"] == "st_amp" else 0
        return f"{e['label']} {_fmt(e['previous'], decimals)} → {_fmt(e['current'], decimals)} {e['unit']}"

    # ── What went wrong ────────────────────────────────────────────────────
    wrong: list[str] = []
    if risk_delta is not None and risk_delta > 0:
        wrong.append(
            f"Overall cardiac risk rose from {_fmt(p_risk, 1)}% ({p_tier}) to {_fmt(c_risk, 1)}% ({c_tier}), "
            f"{_signed(risk_delta, 1)} percentage points {since}."
        )
    if age_delta is not None and age_delta > 0:
        gap_txt = f"; it is now {age_gap:+.1f} years versus chronological age" if age_gap is not None else ""
        wrong.append(
            f"Predicted ECG heart age increased from {_fmt(p_age, 1)} to {_fmt(c_age, 1)} years "
            f"({_signed(age_delta, 1)} yrs){gap_txt}."
        )
    if new_arr:
        wrong.append(
            "New on this 12-lead ECG: "
            + _join([f"{a['name']} ({a['code']}, {a['current_prob'] * 100:.0f}%)" for a in new_arr])
            + "."
        )
    if persistent_arr:
        wrong.append("Still present: " + _join([f"{a['name']} ({a['code']})" for a in persistent_arr]) + ".")
    if ecg_worse:
        wrong.append("ECG morphology worsened: " + _join([_ecg_phrase(e) for e in ecg_worse]) + ".")
    if worsened:
        wrong.append("Worsened biomarkers: " + "; ".join(_bio_phrase(b) for b in worsened) + ".")
    if not wrong:
        wrong.append(f"No measurable deterioration {since}.")
    if curr.get("adherence_notes") and trend == "worsened":
        wrong.append(f"Reported adherence to the previous plan: {curr['adherence_notes']}")

    # ── What improved ──────────────────────────────────────────────────────
    better: list[str] = []
    if risk_delta is not None and risk_delta < 0:
        better.append(
            f"Overall cardiac risk fell from {_fmt(p_risk, 1)}% ({p_tier}) to {_fmt(c_risk, 1)}% ({c_tier}), "
            f"{_signed(risk_delta, 1)} percentage points {since}."
        )
    if age_delta is not None and age_delta < 0:
        gap_txt = f"; it is now {age_gap:+.1f} years versus chronological age" if age_gap is not None else ""
        better.append(
            f"Predicted ECG heart age dropped from {_fmt(p_age, 1)} to {_fmt(c_age, 1)} years "
            f"({_signed(age_delta, 1)} yrs){gap_txt}."
        )
    if resolved_arr:
        better.append("Resolved since last visit: " + _join([f"{a['name']} ({a['code']})" for a in resolved_arr]) + ".")
    if ecg_better:
        better.append("ECG morphology improved: " + _join([_ecg_phrase(e) for e in ecg_better]) + ".")
    if improved:
        better.append("Improved biomarkers: " + "; ".join(_bio_phrase(b) for b in improved) + ".")
    if not better:
        better.append(f"No biomarker improved {since}.")
    if curr.get("adherence_notes") and trend != "worsened":
        better.append(f"Reported adherence to the previous plan: {curr['adherence_notes']}")

    return {
        "is_baseline": False,
        "previous_visit_id": prev.get("id"),
        "previous_visit_date": prev.get("visit_date"),
        "current_visit_date": curr.get("visit_date"),
        "days_between_visits": days,
        "previous_risk_score": p_risk,
        "current_risk_score": c_risk,
        "previous_severity_tier": p_tier,
        "current_severity_tier": c_tier,
        "risk_delta": risk_delta,
        "previous_heart_age": p_age,
        "current_heart_age": c_age,
        "heart_age_delta": age_delta,
        "heart_age_gap_current": age_gap,
        "overall_trend": trend,
        "biomarker_deltas": biomarkers,
        "worsened": [b["key"] for b in worsened],
        "improved": [b["key"] for b in improved],
        "stable": [b["key"] for b in stable],
        "ecg_morphology_deltas": ecg_deltas,
        "new_arrhythmias": new_arr,
        "resolved_arrhythmias": resolved_arr,
        "persistent_arrhythmias": persistent_arr,
        "what_went_wrong_summary": " ".join(wrong),
        "what_improved_summary": " ".join(better),
    }
