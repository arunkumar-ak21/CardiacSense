"""12-lead saliency heatmap for the CODE-15% engine (Requirement 9).

The Saliency_Matrix has 12 rows (Model_Tensor channel order DI, DII, DIII,
AVR, AVL, AVF, V1..V6) and 128 columns. Column ``j`` covers samples
``32j .. 32j + 31``, one 80 ms bin at 400 Hz.

This module imports only NumPy at module level, so the amplitude fallback
works when torch is not installed.
"""
from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

N_BINS = 128
BIN = 32
N_LEADS = 12
N_SAMPLES = N_BINS * BIN  # 4096

# DSP abnormality windows, in samples relative to each R-peak (400 Hz).
QRS_WINDOW = (-24, 24)    # -60 ms .. +60 ms
ST_WINDOW = (24, 80)      # +60 ms .. +200 ms
T_WINDOW = (80, 160)      # +200 ms .. +400 ms

# Normal ranges (Req 9.4).
QRS_NORMAL_MAX_MS = 120.0
ST_NORMAL_LOW_MV = -0.05
ST_NORMAL_HIGH_MV = 0.1
RR_NORMAL_MAX = 0.15
MAX_WEIGHT = 2.0


def intensity_from_grad(grad: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Port of ``intensity_from_grad`` from REPO/ecg-heatmaps/generate_heatmap.py.

    Returns ``scale * g**2 / max(mean(g**2), 0.001)``. Call it once per lead.
    Non-finite inputs propagate as non-finite outputs; ``finalize`` zeroes them.
    """
    g = np.asarray(grad, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        g2 = g * g
        denom = np.maximum(g2.mean(), 0.001) if g2.size else 0.001
        return scale * g2 / denom


def _finite_or_none(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def measurement_weights(dsp: dict) -> tuple[float, float, float]:
    """Return ``(w_qrs, w_st, w_rr)``.

    Each weight is exactly 1.0 inside its normal range, grows linearly with
    the distance from the nearest normal-range bound, and is capped at 2.0.
    Missing or non-finite measurements get weight 1.0.
    """
    dsp = dsp or {}

    w_qrs = 1.0
    qrs = _finite_or_none(dsp.get("qrs_ms"))
    if qrs is not None and qrs >= QRS_NORMAL_MAX_MS:
        w_qrs = min(MAX_WEIGHT, 1.0 + (qrs - QRS_NORMAL_MAX_MS) / 60.0)

    w_st = 1.0
    st = _finite_or_none(dsp.get("st_amp"))
    if st is not None and not (ST_NORMAL_LOW_MV <= st <= ST_NORMAL_HIGH_MV):
        d = ST_NORMAL_LOW_MV - st if st < ST_NORMAL_LOW_MV else st - ST_NORMAL_HIGH_MV
        w_st = min(MAX_WEIGHT, 1.0 + d / 0.2)

    w_rr = 1.0
    rr = _finite_or_none(dsp.get("rr_irregularity"))
    if rr is not None and rr > RR_NORMAL_MAX:
        w_rr = min(MAX_WEIGHT, 1.0 + (rr - RR_NORMAL_MAX) / 0.15)

    return float(w_qrs), float(w_st), float(w_rr)


def _mark_window(mask: np.ndarray, start: int, end: int, last: int) -> None:
    """Set ``mask[b]`` for every bin ``b`` overlapping samples ``[start, end]``
    after clipping the window to ``[0, last]``."""
    a = max(start, 0)
    z = min(end, last)
    if a > z:
        return
    # Bin b overlaps [a, z] iff 32b <= z and 32b + 31 >= a.
    b_lo = a // BIN
    b_hi = min(z // BIN, mask.shape[0] - 1)
    if b_lo <= b_hi:
        mask[b_lo:b_hi + 1] = True


def dsp_bin_weights(dsp: dict, r_peaks: Sequence[int], n: int = N_SAMPLES) -> np.ndarray:
    """Per-bin DSP abnormality weights, shape ``(128,)`` (Req 9.3, 9.5).

    All ones unless ``dsp["measurable"]`` is true and there are at least 2
    R-peaks. QRS-window bins get ``max(w_qrs, w_rr)``; ST- or T-window bins
    get ``w_st``; a bin in both kinds of window gets the larger value.
    """
    weights = np.ones(N_BINS, dtype=np.float64)
    peaks = [] if r_peaks is None else list(r_peaks)
    if not (dsp and dsp.get("measurable")) or len(peaks) < 2:
        return weights

    w_qrs, w_st, w_rr = measurement_weights(dsp)
    last = int(n) - 1
    qrs_mask = np.zeros(N_BINS, dtype=bool)
    st_mask = np.zeros(N_BINS, dtype=bool)
    for r in peaks:
        r = int(r)
        _mark_window(qrs_mask, r + QRS_WINDOW[0], r + QRS_WINDOW[1], last)
        _mark_window(st_mask, r + ST_WINDOW[0], r + ST_WINDOW[1], last)
        _mark_window(st_mask, r + T_WINDOW[0], r + T_WINDOW[1], last)

    weights[qrs_mask] = np.maximum(weights[qrs_mask], max(w_qrs, w_rr))
    weights[st_mask] = np.maximum(weights[st_mask], w_st)
    return weights


def _bin_mean(values: np.ndarray) -> np.ndarray:
    """Average a ``(12, 4096)`` array into ``(12, 128)`` bins of 32 samples."""
    v = np.asarray(values, dtype=np.float64)
    if v.shape != (N_LEADS, N_SAMPLES):
        raise ValueError(
            f"expected array of shape ({N_LEADS}, {N_SAMPLES}); received {v.shape}"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        return v.reshape(N_LEADS, N_BINS, BIN).mean(axis=2)


def amplitude_saliency(x: np.ndarray) -> np.ndarray:
    """Fallback saliency: mean squared amplitude per 32-sample bin, ``(12, 128)``."""
    v = np.asarray(x, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        return _bin_mean(v * v)


def finalize(raw_binned: np.ndarray, weights: np.ndarray) -> list[list[float]]:
    """Apply bin weights and normalize to the Saliency_Matrix (Req 9.6-9.8).

    Non-finite values become 0. If the maximum is not positive the result is
    12 x 128 zeros; otherwise every value is divided by the global maximum,
    clipped to [0, 1] and rounded to 4 decimals.
    """
    raw = np.asarray(raw_binned, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        vals = raw * w[None, :]
    vals = np.where(np.isfinite(vals), vals, 0.0)

    peak = float(vals.max()) if vals.size else 0.0
    if peak <= 0.0:
        return [[0.0] * N_BINS for _ in range(N_LEADS)]

    out = np.round(np.clip(vals / peak, 0.0, 1.0), 4) + 0.0  # +0.0 turns -0.0 into 0.0
    return out.tolist()


def gradient_saliency(model, x_tensor) -> np.ndarray:
    """Raw input gradient of ``model(x)[0, 0]``, shape ``(12, 4096)`` (Req 9.2, 9.9).

    ``x_tensor`` is a ``(1, 12, 4096)`` torch tensor, or an array-like of shape
    ``(12, 4096)`` or ``(1, 12, 4096)`` that is converted to one. The gradient
    is taken on a detached copy with ``torch.autograd.grad``, so no parameter
    ``.grad`` is written, no hooks are registered, the model's train/eval mode
    is untouched and the caller's input is never modified.
    """
    import torch  # lazy: the fallback path must work without torch

    if isinstance(x_tensor, torch.Tensor):
        x = x_tensor
    else:
        x = torch.as_tensor(np.array(x_tensor, dtype=np.float32, copy=True))
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if tuple(x.shape) != (1, N_LEADS, N_SAMPLES):
        raise ValueError(
            f"expected tensor of shape (1, {N_LEADS}, {N_SAMPLES}); received {tuple(x.shape)}"
        )

    param = next(iter(model.parameters()), None)
    if param is not None:
        x = x.to(device=param.device, dtype=param.dtype)

    with torch.enable_grad():
        xi = x.detach().clone().requires_grad_(True)
        out = model(xi)[0, 0]
        (g,) = torch.autograd.grad(out, xi)

    return g[0].detach().cpu().numpy().astype(np.float64)


def compute_saliency(x: np.ndarray, dsp: dict, r_peaks, model=None) -> list[list[float]]:
    """Build the 12 x 128 Saliency_Matrix (Req 9.1-9.11).

    ``x`` is the ``(12, 4096)`` Model_Tensor for one ECG. With a model, the
    raw value is the per-lead ``intensity_from_grad`` of the input gradient,
    averaged over 32-sample bins. Without a model, or if the gradient path
    raises, the raw value is ``amplitude_saliency(x)``. Both are multiplied by
    ``dsp_bin_weights`` and normalized by ``finalize``.
    """
    weights = dsp_bin_weights(dsp, r_peaks)

    raw_binned = None
    if model is not None:
        try:
            grad = gradient_saliency(model, x)
            intensity = np.stack([intensity_from_grad(grad[i]) for i in range(N_LEADS)])
            raw_binned = _bin_mean(intensity)
        except Exception as exc:  # noqa: BLE001 - any failure falls back (Req 9.11)
            logger.error(
                "Gradient saliency failed (%s); using amplitude fallback",
                type(exc).__name__,
            )
            raw_binned = None

    if raw_binned is None:
        raw_binned = amplitude_saliency(x)

    return finalize(raw_binned, weights)
