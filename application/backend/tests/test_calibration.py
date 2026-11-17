"""Properties 23, 24 and 25: arrhythmia probability calibration (Req 8.1-8.9)."""
from __future__ import annotations

import math

from hypothesis import given
from hypothesis import strategies as st

from ml.code15_scoring import calibrate_arrhythmias
from strategies import CODE15_CODES, code15_flag_maps, dsp_dicts

deep_term_maps = st.one_of(
    st.none(),
    st.fixed_dictionaries({}, optional={
        c: st.floats(min_value=-1.0, max_value=1.0, exclude_min=True, exclude_max=True) for c in CODE15_CODES
    }),
)


# ── Property 23 ────────────────────────────────────────────────────────────
@given(code15_flag_maps(allow_missing=True), dsp_dicts(), deep_term_maps)
def test_probabilities_are_well_formed(flags, dsp, deep):
    probs = calibrate_arrhythmias(flags, dsp, deep)
    assert set(probs) == set(CODE15_CODES)
    for value in probs.values():
        assert isinstance(value, float) and math.isfinite(value)
        assert 0.0 <= value <= 1.0
        assert round(value, 3) == value
    assert calibrate_arrhythmias(flags, dsp, deep) == probs


# ── Property 24 ────────────────────────────────────────────────────────────
@given(code15_flag_maps(), dsp_dicts(), deep_term_maps, st.sampled_from(CODE15_CODES))
def test_flag_sets_floor_and_never_lowers(flags, dsp, deep, code):
    probs = calibrate_arrhythmias(flags, dsp, deep)
    for c in CODE15_CODES:
        if flags[c]:
            assert probs[c] >= 0.80

    off = calibrate_arrhythmias({**flags, code: False}, dsp, deep)
    on = calibrate_arrhythmias({**flags, code: True}, dsp, deep)
    assert on[code] >= off[code]


# ── Property 25 ────────────────────────────────────────────────────────────
def _raw(flags, dsp, deep, **changes):
    return calibrate_arrhythmias(flags, {**dsp, **changes}, deep, round_output=False)


# Deep terms within 0.01 of +/-1 push p to exactly 1.0 (or 0.0) in float64 when a flag is
# set, so a strict ">" cannot be observed there; the math is still strictly increasing.
strict_deep_term_maps = st.one_of(
    st.none(),
    st.fixed_dictionaries({}, optional={c: st.floats(min_value=-0.99, max_value=0.99) for c in CODE15_CODES}),
)


@given(code15_flag_maps(), dsp_dicts(measurable=True), strict_deep_term_maps, st.data())
def test_dsp_evidence_raises_matching_code(flags, dsp, deep, data):
    # Margins keep the evidence step above float rounding of p0 near 1.
    pr = data.draw(st.floats(min_value=200.5, max_value=400.0))
    assert _raw(flags, dsp, deep, pr_ms=pr)["1dAVb"] > _raw(flags, dsp, deep, pr_ms=160.0)["1dAVb"]

    qrs = data.draw(st.floats(min_value=120.0, max_value=300.0))
    wide, narrow = _raw(flags, dsp, deep, qrs_ms=qrs), _raw(flags, dsp, deep, qrs_ms=90.0)
    assert wide["LBBB"] > narrow["LBBB"]
    assert wide["RBBB"] > narrow["RBBB"]

    slow = data.draw(st.floats(min_value=0.1, max_value=59.9))
    assert _raw(flags, dsp, deep, heart_rate=slow)["SB"] > _raw(flags, dsp, deep, heart_rate=75.0)["SB"]

    fast = data.draw(st.floats(min_value=100.5, max_value=300.0))
    assert _raw(flags, dsp, deep, heart_rate=fast)["ST"] > _raw(flags, dsp, deep, heart_rate=75.0)["ST"]

    rr = data.draw(st.floats(min_value=0.151, max_value=1.5))
    assert _raw(flags, dsp, deep, rr_irregularity=rr)["AF"] > _raw(flags, dsp, deep, rr_irregularity=0.05)["AF"]
