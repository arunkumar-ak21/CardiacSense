"""Property 34: Risk combination and tiers (Req 10.5, 10.6, 10.7, 10.11)."""
from __future__ import annotations

import math

from hypothesis import given
from hypothesis import strategies as st

from db.delta_engine import severity_tier
from ml.code15_scoring import ClinicalVector, clinical_risk, combine_risk
from strategies import clinical_values, code15_flag_maps

neural_values = st.one_of(
    st.none(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.floats(min_value=-10.0, max_value=10.0),
)


def _expected_tier(score: float) -> str:
    if score < 25:
        return "Low"
    if score < 50:
        return "Moderate"
    if score < 75:
        return "High"
    return "Critical"


@given(clinical_values(), code15_flag_maps(), neural_values)
def test_risk_combination_and_tiers(values, flags, neural):
    present = frozenset(c for c, on in flags.items() if on)
    cv = ClinicalVector(**{**values, "active_arrhythmia_count": len(present)})
    clinical = clinical_risk(cv, present)
    assert math.isfinite(clinical)

    overall, clin_part, neural_part = combine_risk(clinical, neural)

    n = 0.0 if neural is None or not math.isfinite(neural) else min(5.0, max(-5.0, neural))
    assert overall == round(min(100.0, max(0.0, clinical + n)), 1)
    assert 0.0 <= overall <= 100.0
    assert -5.0 <= neural_part <= 5.0
    assert abs((clin_part + neural_part) - (clinical + n)) <= 0.1 + 1e-9

    tier = severity_tier(overall)
    assert tier == _expected_tier(overall)
