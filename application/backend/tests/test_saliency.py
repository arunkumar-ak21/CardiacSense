"""Property 30: Saliency matrix contract (Req 9.1, 9.6, 9.7, 9.8)."""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ml.saliency import compute_saliency
from strategies import canonical_signals, dsp_dicts

REDUCED_BLOCKS = [(8, 1024), (8, 256), (16, 64), (16, 16)]

r_peak_lists = st.lists(st.integers(min_value=-100, max_value=4200), max_size=20)


def _with_non_finite(x: np.ndarray, data: st.DataObject) -> np.ndarray:
    """Optionally put NaN / +inf / -inf values into a copy of ``x``."""
    count = data.draw(st.integers(min_value=0, max_value=50))
    if not count:
        return x
    x = x.copy()
    rng = np.random.default_rng(data.draw(st.integers(min_value=0, max_value=2**32 - 1)))
    rows = rng.integers(0, 12, size=count)
    cols = rng.integers(0, 4096, size=count)
    x[rows, cols] = rng.choice([np.nan, np.inf, -np.inf], size=count)
    return x


def _assert_contract(matrix) -> None:
    assert len(matrix) == 12
    assert all(len(row) == 128 for row in matrix)
    arr = np.asarray(matrix, dtype=np.float64)
    assert np.isfinite(arr).all()
    assert (arr >= 0.0).all() and (arr <= 1.0).all()
    assert np.array_equal(np.round(arr, 4), arr)  # at most 4 decimals
    peak = float(arr.max())
    assert peak == 1.0 or not arr.any()


@pytest.fixture(scope="module")
def reduced_model():
    torch = pytest.importorskip("torch")
    from ml.resnet1d import ResNet1d

    torch.manual_seed(0)
    return ResNet1d((12, 4096), REDUCED_BLOCKS, 1, 17, 0.8).eval()


@given(canonical_signals(), dsp_dicts(), r_peak_lists, st.data())
def test_fallback_path_contract(signal, dsp, r_peaks, data):
    x = _with_non_finite(np.ascontiguousarray(signal.T), data)
    _assert_contract(compute_saliency(x, dsp, r_peaks, model=None))


@settings(max_examples=30)
@given(canonical_signals(), dsp_dicts(), r_peak_lists, st.data())
def test_gradient_path_contract(reduced_model, signal, dsp, r_peaks, data):
    x = _with_non_finite(np.ascontiguousarray(signal.T), data)
    _assert_contract(compute_saliency(x, dsp, r_peaks, model=reduced_model))


def test_all_zero_input_gives_all_zero_matrix():
    matrix = compute_saliency(np.zeros((12, 4096), dtype=np.float32), {"measurable": False}, [], model=None)
    assert matrix == [[0.0] * 128 for _ in range(12)]


def test_all_non_finite_input_gives_all_zero_matrix():
    x = np.full((12, 4096), np.nan, dtype=np.float32)
    assert compute_saliency(x, {"measurable": False}, [], model=None) == [[0.0] * 128 for _ in range(12)]
