"""Properties 6 and 7: Signal_Preprocessor output (Req 4.6, 4.16, 4.17)."""
from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from ml.code15_engine import SignalPreprocessor
from strategies import canonical_signals, raw_signals

sample_rates = st.one_of(
    st.none(),
    st.integers(min_value=100, max_value=2000),
    st.floats(min_value=100.0, max_value=2000.0),
)


# ── Property 6 ─────────────────────────────────────────────────────────────
@given(raw_signals(), sample_rates)
def test_preprocessing_produces_valid_model_tensor(raw, rate):
    pre = SignalPreprocessor()
    tensor = pre.to_model_array(pre.to_canonical(raw, rate))
    assert tensor.shape == (1, 12, 4096)
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()


# ── Property 7 ─────────────────────────────────────────────────────────────
@given(canonical_signals())
def test_canonical_round_trip_is_exact(signal):
    pre = SignalPreprocessor()
    tensor = pre.to_model_array(pre.to_canonical(signal, 400))
    back = tensor[0].T
    assert back.dtype == np.float32
    assert np.array_equal(back.view(np.uint32), signal.view(np.uint32))  # bit for bit
