"""
Demo data for the longitudinal companion system.

1. ``generate_sample_code15_files()`` writes four synthetic CODE-15%-style 12-lead
   ECGs, shape ``(4096, 12)`` at 400 Hz, to ``application/sample_code15_ecgs/``
   as ``.npy`` and ``.csv`` (and ``.hdf5`` with a ``tracings`` dataset of shape
   ``(1, 4096, 12)`` when ``h5py`` is installed).
2. ``seed_initial_database(db_session)`` inserts two patients with two visits each
   when the ``patients`` table is empty.

Signal scale follows the CODE-15% convention used by the pre-trained ResNet1d
(values numerically in millivolts, i.e. volts x 1000). Lead order is
DI, DII, DIII, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6.

Limb leads are built from a frontal-plane cardiac dipole and then derived with the
Einthoven/Goldberger relations (DIII = DII - DI, aVR = -(DI + DII)/2, ...), so the
six limb leads are mutually consistent. Precordial leads use per-lead amplitude
tables with normal R-wave progression (or LBBB morphology).

Run directly from ``application/backend``::

    python -m db.seed_and_samples
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .database import BACKEND_DIR
from .delta_engine import CODE15_LABELS, compute_visit_deltas, severity_tier
from .models import LEAD_NAMES, Patient, Visit, json_dumps

FS = 400
N_SAMPLES = 4096
DURATION_S = N_SAMPLES / FS
SAMPLES_DIR = BACKEND_DIR.parent / "sample_code15_ecgs"
PREVIEW_DECIMATION = 4          # 400 Hz -> 100 Hz preview (1024 points per lead)
HEATMAP_BINS = 128              # 12 x 128 saliency matrix (80 ms per bin)

LIMB_ANGLES = {"DI": 0.0, "DII": 60.0}
PRECORDIAL = ["V1", "V2", "V3", "V4", "V5", "V6"]


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic 12-lead ECG generator
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class ECGProfile:
    """Parameters for one synthetic 12-lead recording."""

    name: str
    heart_rate: float
    qrs_ms: float
    pr_ms: float | None            # None -> no organised P waves (atrial fibrillation)
    rhythm: str = "sinus"          # "sinus" | "af"
    conduction: str = "normal"     # "normal" | "lbbb"
    st_amp_mv: float = 0.0         # max ST deviation in the affected leads (negative = depression)
    seed: int = 0
    noise_mv: float = 0.010
    description: str = ""
    expected_flags: dict[str, bool] = field(default_factory=dict)


# Frontal-plane dipoles: wave -> (magnitude mV, axis degrees)
_NORMAL_FRONTAL = {"P": (0.15, 55.0), "Q": (0.10, 210.0), "R": (1.30, 55.0), "S": (0.25, 250.0), "T": (0.35, 45.0)}
_LBBB_FRONTAL = {"P": (0.15, 55.0), "Q": (0.00, 0.0), "R": (1.20, -10.0), "S": (0.30, 200.0), "T": (0.30, 180.0)}

# Precordial amplitude tables: lead -> {wave: amplitude mV}
_NORMAL_PRECORDIAL = {
    "V1": {"P": 0.06, "Q": 0.00, "R": 0.25, "S": -1.00, "T": 0.10},
    "V2": {"P": 0.08, "Q": 0.00, "R": 0.50, "S": -1.30, "T": 0.45},
    "V3": {"P": 0.08, "Q": 0.00, "R": 0.90, "S": -0.90, "T": 0.50},
    "V4": {"P": 0.10, "Q": -0.05, "R": 1.40, "S": -0.50, "T": 0.45},
    "V5": {"P": 0.10, "Q": -0.08, "R": 1.50, "S": -0.25, "T": 0.35},
    "V6": {"P": 0.10, "Q": -0.08, "R": 1.20, "S": -0.10, "T": 0.30},
}
_LBBB_PRECORDIAL = {
    "V1": {"P": 0.06, "Q": 0.00, "R": 0.05, "S": -1.60, "T": 0.45},
    "V2": {"P": 0.08, "Q": 0.00, "R": 0.08, "S": -1.90, "T": 0.55},
    "V3": {"P": 0.08, "Q": 0.00, "R": 0.20, "S": -1.30, "T": 0.40},
    "V4": {"P": 0.10, "Q": 0.00, "R": 0.70, "S": -0.70, "T": 0.05},
    "V5": {"P": 0.10, "Q": 0.00, "R": 1.40, "S": 0.00, "T": -0.30},
    "V6": {"P": 0.10, "Q": 0.00, "R": 1.30, "S": 0.00, "T": -0.35},
}

# ST-deviation distribution per lead (multiplied by profile.st_amp_mv)
_ST_WEIGHTS_ISCHEMIC = {
    "DI": 0.6, "DII": 1.0, "DIII": 0.5, "AVR": -0.8, "AVL": 0.3, "AVF": 0.8,
    "V1": -0.3, "V2": 0.2, "V3": 0.6, "V4": 1.0, "V5": 1.0, "V6": 0.9,
}
_ST_WEIGHTS_LBBB = {
    "DI": 0.9, "DII": 0.3, "DIII": -0.3, "AVR": -0.4, "AVL": 0.9, "AVF": 0.0,
    "V1": -0.9, "V2": -0.9, "V3": -0.6, "V4": 0.2, "V5": 0.9, "V6": 1.0,
}
# Fibrillatory-wave visibility per lead (largest in V1 and the inferior leads)
_FWAVE_GAIN = {
    "DI": 0.3, "DII": 0.6, "DIII": 0.6, "AVR": 0.4, "AVL": 0.3, "AVF": 0.6,
    "V1": 1.0, "V2": 0.6, "V3": 0.4, "V4": 0.3, "V5": 0.3, "V6": 0.3,
}


def _gauss(t: np.ndarray, center: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - center) / sigma) ** 2)


def _asym_gauss(t: np.ndarray, center: float, sigma_left: float, sigma_right: float) -> np.ndarray:
    sigma = np.where(t < center, sigma_left, sigma_right)
    return np.exp(-0.5 * ((t - center) / sigma) ** 2)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _amplitude_table(profile: ECGProfile) -> dict[str, dict[str, float]]:
    """Per-lead signed amplitude of each wave (P, Q, R, S, T)."""
    frontal = _LBBB_FRONTAL if profile.conduction == "lbbb" else _NORMAL_FRONTAL
    precordial = _LBBB_PRECORDIAL if profile.conduction == "lbbb" else _NORMAL_PRECORDIAL

    lead_i: dict[str, float] = {}
    lead_ii: dict[str, float] = {}
    for wave, (mag, axis) in frontal.items():
        lead_i[wave] = mag * math.cos(math.radians(axis - LIMB_ANGLES["DI"]))
        lead_ii[wave] = mag * math.cos(math.radians(axis - LIMB_ANGLES["DII"]))

    table: dict[str, dict[str, float]] = {}
    for wave in frontal:
        i, ii = lead_i[wave], lead_ii[wave]
        table.setdefault("DI", {})[wave] = i
        table.setdefault("DII", {})[wave] = ii
        table.setdefault("DIII", {})[wave] = ii - i
        table.setdefault("AVR", {})[wave] = -(i + ii) / 2.0
        table.setdefault("AVL", {})[wave] = i - ii / 2.0
        table.setdefault("AVF", {})[wave] = ii - i / 2.0
    for lead in PRECORDIAL:
        table[lead] = dict(precordial[lead])

    if profile.st_amp_mv != 0.0:
        # Leads with ST depression also show flattened T waves.
        weights = _ST_WEIGHTS_LBBB if profile.conduction == "lbbb" else _ST_WEIGHTS_ISCHEMIC
        for lead in LEAD_NAMES:
            st = profile.st_amp_mv * weights[lead]
            if st < 0:
                table[lead]["T"] *= max(0.3, 1.0 - 2.5 * abs(st))
    return table


def _beat_times(profile: ECGProfile, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """R-peak times (s) and the RR interval preceding each beat."""
    rr_mean = 60.0 / profile.heart_rate
    t0 = -rr_mean * rng.uniform(0.3, 0.9)  # start before 0 so the record opens mid-cycle
    n_beats = int(math.ceil((DURATION_S + 1.0 - t0) / (0.5 * rr_mean))) + 2

    if profile.rhythm == "af":
        # Irregularly irregular: log-normal RR (~18 % CV), rescaled so the mean RR
        # inside the 10.24 s window matches the requested heart rate exactly.
        rr = rr_mean * rng.lognormal(mean=0.0, sigma=0.18, size=n_beats)
        rr = np.clip(rr, 0.55 * rr_mean, 1.7 * rr_mean)
        for _ in range(5):
            times = t0 + np.concatenate([[0.0], np.cumsum(rr)])
            inside = times[(times >= 0) & (times < DURATION_S)]
            if inside.size < 2:
                break
            rr = rr * (rr_mean / float(np.diff(inside).mean()))
    else:
        # Respiratory sinus arrhythmia (~0.25 Hz) plus small beat-to-beat jitter.
        rr = np.empty(n_beats)
        t = t0
        for k in range(n_beats):
            rr[k] = rr_mean * (1.0 + 0.03 * math.sin(2 * math.pi * 0.25 * t) + float(rng.normal(0.0, 0.008)))
            t += rr[k]

    times = t0 + np.concatenate([[0.0], np.cumsum(rr)])
    prev_rr = np.concatenate([[rr_mean], rr])
    keep = times < DURATION_S + 1.0
    return times[keep], prev_rr[keep]


def synthesize_12lead_ecg(profile: ECGProfile) -> dict[str, Any]:
    """
    Generate a ``(4096, 12)`` float32 ECG for ``profile``.

    Returns the signal plus ground-truth timing used for DSP measurements and saliency:
    ``signal``, ``r_peak_samples``, ``r_times``, ``measured_hr``, ``rr_std_ms``.
    """
    rng = np.random.default_rng(profile.seed)
    t = np.arange(N_SAMPLES) / FS
    sig = np.zeros((N_SAMPLES, len(LEAD_NAMES)), dtype=np.float64)
    table = _amplitude_table(profile)
    st_weights = _ST_WEIGHTS_LBBB if profile.conduction == "lbbb" else _ST_WEIGHTS_ISCHEMIC

    qrs = profile.qrs_ms / 1000.0
    r_times, rr_prev = _beat_times(profile, rng)

    for r, rr in zip(r_times, rr_prev):
        lo = max(0, int((r - 0.45) * FS))
        hi = min(N_SAMPLES, int((r + 0.75) * FS) + 1)
        if hi <= lo:
            continue
        tw = t[lo:hi]
        amp_jitter = 1.0 + float(rng.normal(0.0, 0.025))

        qrs_onset = r - 0.5 * qrs
        j_point = r + 0.5 * qrs
        qt = 0.40 * math.sqrt(max(rr, 0.3)) + max(0.0, qrs - 0.09)
        t_end = qrs_onset + qt
        t_peak = t_end - 0.07 * math.sqrt(max(rr, 0.3))
        t_sig_l = max(0.030, 0.050 * math.sqrt(rr))
        t_sig_r = max(0.022, 0.032 * math.sqrt(rr))

        p_wave = None
        if profile.pr_ms is not None:
            p_center = qrs_onset - profile.pr_ms / 1000.0 + 0.05
            p_wave = _gauss(tw, p_center, 0.018)

        q_wave = _gauss(tw, r - 0.32 * qrs, 0.07 * qrs)
        if profile.conduction == "lbbb":
            # Broad, notched R (two summits) and a wide S/QS deflection.
            r_wave = 0.8 * _gauss(tw, r - 0.18 * qrs, 0.12 * qrs) + _gauss(tw, r + 0.14 * qrs, 0.13 * qrs)
            s_wave = _gauss(tw, r + 0.05 * qrs, 0.22 * qrs)
        else:
            r_wave = _gauss(tw, r, 0.11 * qrs)
            s_wave = _gauss(tw, r + 0.30 * qrs, 0.09 * qrs)
        t_wave = _asym_gauss(tw, t_peak, t_sig_l, t_sig_r)
        st_window = _sigmoid((tw - j_point) / 0.006) * (1.0 - _sigmoid((tw - t_peak) / 0.025))

        for li, lead in enumerate(LEAD_NAMES):
            amps = table[lead]
            beat = amps["Q"] * q_wave + amps["R"] * r_wave + amps["S"] * s_wave + amps["T"] * t_wave
            if p_wave is not None:
                beat = beat + amps["P"] * p_wave
            if profile.st_amp_mv != 0.0:
                beat = beat + profile.st_amp_mv * st_weights[lead] * st_window
            sig[lo:hi, li] += amp_jitter * beat

    # Atrial fibrillation: continuous 4-8 Hz fibrillatory baseline.
    if profile.rhythm == "af":
        fwave = np.zeros_like(t)
        for _ in range(4):
            f = rng.uniform(4.5, 7.5)
            phase = rng.uniform(0, 2 * math.pi)
            mod = 0.6 + 0.4 * np.sin(2 * math.pi * rng.uniform(0.1, 0.4) * t + rng.uniform(0, 2 * math.pi))
            fwave += rng.uniform(0.015, 0.03) * mod * np.sin(2 * math.pi * f * t + phase)
        for li, lead in enumerate(LEAD_NAMES):
            sig[:, li] += _FWAVE_GAIN[lead] * fwave

    # Baseline wander, mains hum and white noise.
    for li in range(len(LEAD_NAMES)):
        sig[:, li] += 0.04 * np.sin(2 * math.pi * 0.22 * t + rng.uniform(0, 2 * math.pi))
        sig[:, li] += 0.02 * np.sin(2 * math.pi * 0.07 * t + rng.uniform(0, 2 * math.pi))
        sig[:, li] += 0.004 * np.sin(2 * math.pi * 60.0 * t + rng.uniform(0, 2 * math.pi))
        sig[:, li] += rng.normal(0.0, profile.noise_mv, N_SAMPLES)

    in_window = (r_times >= 0) & (r_times < DURATION_S)
    r_in = r_times[in_window]
    rr_in = np.diff(r_in)
    measured_hr = float(60.0 / rr_in.mean()) if rr_in.size else float(profile.heart_rate)
    rr_std_ms = float(rr_in.std() * 1000.0) if rr_in.size else 0.0

    return {
        "signal": sig.astype(np.float32),
        "r_times": r_in,
        "r_peak_samples": np.round(r_in * FS).astype(int),
        "measured_hr": measured_hr,
        "rr_std_ms": rr_std_ms,
    }


def dsp_measurements(profile: ECGProfile, ecg: dict[str, Any]) -> dict[str, Any]:
    """DSP-style interval summary derived from the generator's ground truth."""
    st = round(profile.st_amp_mv, 2)
    if st <= -0.05:
        st_status = f"Depression ({st:.2f} mV)"
    elif st >= 0.1:
        st_status = f"Elevation (+{st:.2f} mV)"
    else:
        st_status = "Normal (isoelectric)"
    rr_std = ecg["rr_std_ms"]
    return {
        "heart_rate": int(round(ecg["measured_hr"])),
        "qrs_ms": int(round(profile.qrs_ms)),
        "pr_ms": int(round(profile.pr_ms)) if profile.pr_ms is not None else None,
        "pr_status": "Not measurable (no organised P waves)" if profile.pr_ms is None else (
            "Prolonged (>200 ms)" if profile.pr_ms > 200 else "Normal"
        ),
        "st_status": st_status,
        "st_amp": st,
        "rr_std_ms": round(rr_std, 1),
        "rhythm_regularity": "Irregularly irregular" if profile.rhythm == "af" else "Regular",
        "r_peak_count": int(len(ecg["r_peak_samples"])),
    }


def _downsample_preview(signal: np.ndarray) -> dict[str, list[float]]:
    from scipy.signal import decimate  # imported lazily: keep scipy out of module import time

    reduced = decimate(signal.astype(np.float64), PREVIEW_DECIMATION, axis=0, zero_phase=True)
    return {lead: np.round(reduced[:, i], 4).tolist() for i, lead in enumerate(LEAD_NAMES)}


def synthesize_saliency(profile: ECGProfile, ecg: dict[str, Any], n_bins: int = HEATMAP_BINS) -> list[list[float]]:
    """
    Build a normalised ``(12, n_bins)`` saliency matrix that highlights the leads and
    time segments carrying the profile's abnormalities (seed data for the UI heatmap).
    """
    rng = np.random.default_rng(profile.seed + 7919)
    t = np.arange(N_SAMPLES) / FS
    sal = np.full((len(LEAD_NAMES), N_SAMPLES), 0.03)
    idx = {lead: i for i, lead in enumerate(LEAD_NAMES)}
    qrs = profile.qrs_ms / 1000.0

    def bump(lead: str, center: float, sigma: float, weight: float) -> None:
        sal[idx[lead]] += weight * _gauss(t, center, sigma)

    r_times = ecg["r_times"]
    rr = np.diff(r_times) if len(r_times) > 1 else np.array([60.0 / profile.heart_rate])
    rr_mean = float(rr.mean())

    for k, r in enumerate(r_times):
        # Baseline QRS attention on every lead.
        for lead in LEAD_NAMES:
            bump(lead, r, 0.03, 0.12)

        if profile.conduction == "lbbb":
            for lead, w in {"DI": 0.85, "AVL": 0.8, "V1": 1.0, "V2": 0.8, "V5": 0.95, "V6": 1.0}.items():
                bump(lead, r, 0.5 * qrs, w)

        if profile.st_amp_mv != 0.0:
            weights = _ST_WEIGHTS_LBBB if profile.conduction == "lbbb" else _ST_WEIGHTS_ISCHEMIC
            mag = min(1.0, abs(profile.st_amp_mv) / 0.2)
            for lead, w in weights.items():
                bump(lead, r + 0.5 * qrs + 0.08, 0.05, 0.9 * mag * abs(w))

        if profile.pr_ms is not None and profile.pr_ms > 200:
            p_mid = r - 0.5 * qrs - profile.pr_ms / 2000.0
            for lead, w in {"DII": 0.9, "V1": 0.8, "AVF": 0.6, "DIII": 0.5}.items():
                bump(lead, p_mid, 0.06, w)

        if profile.rhythm == "af":
            # Attention on the fibrillatory TP segment and on beats with abnormal RR.
            nxt = r_times[k + 1] if k + 1 < len(r_times) else r + rr_mean
            tp_mid = r + 0.6 * (nxt - r)
            for lead, w in {"V1": 0.8, "DII": 0.55, "AVF": 0.5, "DIII": 0.45}.items():
                bump(lead, tp_mid, 0.08, w)
            if k > 0:
                dev = min(1.0, abs((r - r_times[k - 1]) - rr_mean) / (0.25 * rr_mean))
                bump("DII", r, 0.04, 0.6 * dev)

        if profile.heart_rate > 100 and profile.rhythm == "sinus":
            for lead, w in {"DII": 0.5, "V2": 0.4, "V3": 0.45, "V4": 0.4}.items():
                bump(lead, r, 0.05, w)

        if profile.heart_rate < 60:
            for lead, w in {"DII": 0.45, "V5": 0.3}.items():
                bump(lead, r, 0.05, w)

    sal += rng.normal(0.0, 0.01, sal.shape)
    pooled = sal[:, : (N_SAMPLES // n_bins) * n_bins].reshape(len(LEAD_NAMES), n_bins, -1).mean(axis=2)
    pooled = np.clip(pooled, 0.0, None)
    # Scale into [0, 1] against a fixed reference so a normal ECG stays "cool" while
    # strongly abnormal records saturate toward 1.0.
    pooled = np.clip(pooled / max(float(pooled.max()), 1.0), 0.0, 1.0)
    return np.round(pooled, 3).tolist()


# ── Sample-file profiles (match the four required filenames) ─────────────────
SAMPLE_PROFILES: list[tuple[str, ECGProfile]] = [
    (
        "1_normal_sinus_4096x12",
        ECGProfile(
            name="Normal sinus rhythm",
            heart_rate=72, qrs_ms=90, pr_ms=160, seed=1001,
            description="Clean 72 bpm normal sinus rhythm, narrow QRS, isoelectric ST.",
            expected_flags={"1dAVb": False, "RBBB": False, "LBBB": False, "SB": False, "AF": False, "ST": False, "normal_ecg": True},
        ),
    ),
    (
        "2_atrial_fib_lbbb_high_risk_4096x12",
        ECGProfile(
            name="Atrial fibrillation with LBBB",
            heart_rate=92, qrs_ms=148, pr_ms=None, rhythm="af", conduction="lbbb", st_amp_mv=-0.15, seed=2002,
            description="Irregularly irregular RR, absent P waves with f-waves, QRS 148 ms (LBBB), ST depression in lateral leads.",
            expected_flags={"1dAVb": False, "RBBB": False, "LBBB": True, "SB": False, "AF": True, "ST": False, "normal_ecg": False},
        ),
    ),
    (
        "3_sinus_bradycardia_1davb_4096x12",
        ECGProfile(
            name="Sinus bradycardia with 1st-degree AV block",
            heart_rate=48, qrs_ms=94, pr_ms=245, seed=3003,
            description="Regular 48 bpm sinus rhythm with PR interval 245 ms (>220 ms).",
            expected_flags={"1dAVb": True, "RBBB": False, "LBBB": False, "SB": True, "AF": False, "ST": False, "normal_ecg": False},
        ),
    ),
    (
        "4_critical_st_depression_tachycardia_4096x12",
        ECGProfile(
            name="Sinus tachycardia with deep ST depression",
            heart_rate=118, qrs_ms=88, pr_ms=132, st_amp_mv=-0.25, seed=4004,
            description="Regular 118 bpm sinus tachycardia with 0.25 mV ST depression in II, aVF, V3-V6 and reciprocal aVR elevation.",
            expected_flags={"1dAVb": False, "RBBB": False, "LBBB": False, "SB": False, "AF": False, "ST": True, "normal_ecg": False},
        ),
    ),
]


def generate_sample_code15_files(output_dir: str | Path | None = None) -> list[str]:
    """
    Write the four sample ECGs (``.npy`` + ``.csv``, plus ``.hdf5`` if h5py is
    available) and a ``manifest.json``. Output is deterministic; re-running
    overwrites the files with identical content. Returns the written file paths.
    """
    out_dir = Path(output_dir) if output_dir is not None else SAMPLES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import h5py  # type: ignore
    except ImportError:
        h5py = None

    written: list[str] = []
    manifest: list[dict[str, Any]] = []
    header = ",".join(LEAD_NAMES)

    for stem, profile in SAMPLE_PROFILES:
        ecg = synthesize_12lead_ecg(profile)
        signal = ecg["signal"]
        if signal.shape != (N_SAMPLES, 12):
            raise ValueError(f"{stem}: unexpected shape {signal.shape}")

        npy_path = out_dir / f"{stem}.npy"
        np.save(npy_path, signal)
        written.append(str(npy_path))

        csv_path = out_dir / f"{stem}.csv"
        np.savetxt(csv_path, signal, delimiter=",", header=header, comments="", fmt="%.5f")
        written.append(str(csv_path))

        files = [npy_path.name, csv_path.name]
        if h5py is not None:
            h5_path = out_dir / f"{stem}.hdf5"
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("tracings", data=signal[np.newaxis, :, :], dtype="float32")
                f.create_dataset("exam_id", data=np.array([profile.seed], dtype=np.int64))
                f.attrs["sample_rate"] = FS
                f.attrs["lead_names"] = ",".join(LEAD_NAMES)
                f.attrs["description"] = profile.description
            written.append(str(h5_path))
            files.append(h5_path.name)

        manifest.append(
            {
                "stem": stem,
                "files": files,
                "label": profile.name,
                "description": profile.description,
                "shape": [N_SAMPLES, 12],
                "sample_rate": FS,
                "lead_names": LEAD_NAMES,
                "units": "mV (CODE-15% scale: volts x 1000)",
                "dsp_measurements": dsp_measurements(profile, ecg),
                "expected_code15_flags": profile.expected_flags,
                "synthetic": True,
            }
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    written.append(str(manifest_path))
    return written


# ═══════════════════════════════════════════════════════════════════════════
# Demo patient seeding
# ═══════════════════════════════════════════════════════════════════════════
def _flags_from_probs(probs: dict[str, float]) -> dict[str, bool]:
    flags = {code: probs.get(code, 0.0) >= 0.5 for code in CODE15_LABELS}
    flags["normal_ecg"] = not any(flags.values())
    return flags


def _ecg_bundle(profile: ECGProfile, filename: str) -> tuple[dict[str, Any], list[list[float]]]:
    ecg = synthesize_12lead_ecg(profile)
    metadata = {
        "filename": filename,
        "format": "CODE-15% HDF5 (tracings 1x4096x12)",
        "sample_rate": FS,
        "n_samples": N_SAMPLES,
        "duration_s": round(DURATION_S, 2),
        "lead_names": LEAD_NAMES,
        "units": "mV",
        "rhythm_label": profile.name,
        "dsp_measurements": dsp_measurements(profile, ecg),
        "preview_sample_rate": FS // PREVIEW_DECIMATION,
        "leads_preview": _downsample_preview(ecg["signal"]),
        "synthetic": True,
    }
    return metadata, synthesize_saliency(profile, ecg)


def _shap(items: list[tuple[str, str, float, float]]) -> list[dict[str, Any]]:
    rows = [
        {
            "feature": feature,
            "display_name": display,
            "value": value,
            "shap_score": score,
            "direction": "positive" if score >= 0 else "negative",
        }
        for feature, display, value, score in items
    ]
    return sorted(rows, key=lambda r: abs(r["shap_score"]), reverse=True)


def _predictions(risk: float, heart_age: float, chrono_age: int, arrhythmias: dict[str, float]) -> dict[str, Any]:
    return {
        "overall_risk_score": risk,
        "severity_tier": severity_tier(risk),
        "predicted_heart_age": heart_age,
        "chronological_age": chrono_age,
        "heart_age_gap": round(heart_age - chrono_age, 1),
        "arrhythmias": arrhythmias,
        "model_version": "seed-demo-v1",
        "source": "seeded_demo",
    }


def _demo_patients() -> list[dict[str, Any]]:
    """Full demo dataset: two patients with two chronological visits each."""

    # ── Patient 1: Ramesh Kumar ────────────────────────────────────────────
    ramesh_v1_arr = {"1dAVb": 0.05, "RBBB": 0.04, "LBBB": 0.03, "SB": 0.06, "AF": 0.04, "ST": 0.03}
    ramesh_v2_arr = {"1dAVb": 0.08, "RBBB": 0.06, "LBBB": 0.88, "SB": 0.03, "AF": 0.84, "ST": 0.14}

    ramesh_v1_plan = {
        "plan_title": "Standard Heart-Healthy Vegetarian Plan",
        "diet_preference": "vegetarian",
        "daily_targets": {"sodium_mg": 2000, "protein_g": 60, "fluid_liters": 2.5, "calories": 2000, "potassium_mg": 3500, "fiber_g": 30},
        "meals": {
            "breakfast": {"time": "08:00", "items": ["Vegetable poha with peanuts (light salt)", "1 cup low-fat milk or unsweetened chai", "1 guava"], "notes": "Keep added salt to a pinch."},
            "lunch": {"time": "13:00", "items": ["2 whole-wheat rotis", "Moong dal (1 bowl)", "Mixed vegetable sabzi", "Cucumber-carrot salad", "Low-fat curd (1 small bowl)"], "notes": "Fill half the plate with vegetables."},
            "snack": {"time": "17:00", "items": ["Roasted chana (30 g)", "1 apple"], "notes": "Avoid fried namkeen."},
            "dinner": {"time": "20:00", "items": ["Brown rice (1 cup) or 2 rotis", "Palak paneer (low-fat paneer, light salt)", "Sauteed beans"], "notes": "Finish dinner at least 2 hours before bed."},
        },
        "foods_to_avoid": ["Deep-fried snacks (samosa, pakoda)", "Sugary sweets and soft drinks", "Packaged namkeen and chips", "Excess ghee and butter"],
        "exercise_plan": {
            "allowed_activity": "Brisk outdoor walking",
            "duration_mins": 30,
            "sessions_per_day": 1,
            "frequency": "Daily",
            "intensity": "Moderate (able to talk, not sing)",
            "max_hr_ceiling": 130,
            "warnings": ["Stop and rest if you feel chest discomfort, dizziness or unusual breathlessness.", "Warm up for 5 minutes before walking briskly."],
        },
        "why_plan_changed_summary": "Baseline plan for the first recorded visit: moderate risk (34.2%) with normal sinus rhythm and normal kidney function, so a standard 2,000 mg sodium vegetarian diet with daily brisk walking is appropriate.",
        "generated_by": "CardiacSense adaptive planner (seeded demo)",
    }

    ramesh_v2_plan = {
        "plan_title": "Renal-Cardiac Low-Sodium Vegetarian Plan (Fluid-Restricted)",
        "diet_preference": "vegetarian",
        "daily_targets": {"sodium_mg": 1400, "protein_g": 50, "fluid_liters": 1.5, "calories": 1700, "potassium_mg": 2000, "fiber_g": 30},
        "meals": {
            "breakfast": {"time": "08:00", "items": ["Oats porridge cooked in water with a little low-fat milk (no added salt)", "1 small apple, sliced", "1 tsp ground flaxseed"], "notes": "Oats provide soluble fiber to help lower cholesterol (242 mg/dL)."},
            "lunch": {"time": "13:00", "items": ["2 unsalted phulkas", "Lauki (bottle gourd) sabzi cooked with jeera, no salt added at table", "Small bowl of moong dal (1/2 cup)", "Cucumber salad with lemon"], "notes": "Low-potassium vegetables (bottle gourd, ridge gourd, cabbage) protect the kidneys."},
            "snack": {"time": "17:00", "items": ["Unsalted puffed rice (murmura), 1 cup", "1 pear or a few grapes"], "notes": "Count tea and water toward the 1.5 L daily fluid limit."},
            "dinner": {"time": "19:30", "items": ["Vegetable dalia khichdi with ridge gourd and carrot (light salt)", "Sauteed cabbage with herbs"], "notes": "Keep dinner light; season with lemon, garlic and herbs instead of salt."},
        },
        "foods_to_avoid": [
            "Pickles (achar) and papad",
            "Restaurant, takeaway and street food",
            "Salted namkeen, chips and instant noodles",
            "Salt substitutes containing potassium chloride",
            "High-potassium foods: bananas, coconut water, tomatoes in large amounts, potatoes",
            "Paneer and cheese in large portions (excess protein and sodium)",
            "Caffeine-heavy drinks and energy drinks (can trigger AF episodes)",
            "Alcohol",
        ],
        "exercise_plan": {
            "allowed_activity": "Slow, flat indoor pacing",
            "duration_mins": 10,
            "sessions_per_day": 2,
            "frequency": "Twice daily (morning and evening)",
            "intensity": "Very light",
            "max_hr_ceiling": 102,
            "warnings": [
                "Stop the previous 30-minute brisk outdoor walking routine until the cardiology team reviews you.",
                "Keep heart rate below 102 bpm; check your pulse or watch before and after each session.",
                "No stairs, uphill walking, lifting over 5 kg, or exercise in hot weather.",
                "Stop immediately and seek care for palpitations, chest pain, fainting or breathlessness.",
            ],
        },
        "why_plan_changed_summary": (
            "Your plan was tightened because risk rose from 34.2% to 74.6% (High). The new ECG shows atrial "
            "fibrillation and left bundle branch block, so brisk outdoor walking is no longer safe and has been "
            "replaced by two 10-minute slow indoor sessions with a 102 bpm ceiling. Systolic BP rose from 128 to "
            "156 mmHg, so sodium drops from 2,000 mg to under 1,400 mg per day. Serum creatinine rose from 1.0 to "
            "1.7 mg/dL (kidney strain), so fluids are limited to 1.5 L, protein to about 50 g, and high-potassium "
            "foods are restricted. Cholesterol of 242 mg/dL is addressed with oats and other soluble fiber."
        ),
        "generated_by": "CardiacSense adaptive planner (seeded demo)",
    }

    # ── Patient 2: Eleanor Vance ───────────────────────────────────────────
    eleanor_v1_arr = {"1dAVb": 0.04, "RBBB": 0.05, "LBBB": 0.03, "SB": 0.01, "AF": 0.07, "ST": 0.79}
    eleanor_v2_arr = {"1dAVb": 0.03, "RBBB": 0.04, "LBBB": 0.02, "SB": 0.04, "AF": 0.03, "ST": 0.06}

    eleanor_v1_plan = {
        "plan_title": "Mediterranean Low-Sodium DASH Plan (Chicken & Fish)",
        "diet_preference": "non_vegetarian",
        "daily_targets": {"sodium_mg": 1500, "protein_g": 75, "fluid_liters": 2.5, "calories": 1800, "potassium_mg": 4000, "fiber_g": 30},
        "meals": {
            "breakfast": {"time": "07:30", "items": ["Greek yogurt with berries and walnuts", "Whole-grain toast with avocado", "Green tea"], "notes": "No processed breakfast meats."},
            "lunch": {"time": "12:30", "items": ["Grilled chicken breast (120 g)", "Quinoa and chickpea salad with olive oil and lemon", "Steamed broccoli"], "notes": "Use herbs and citrus instead of salt."},
            "snack": {"time": "16:00", "items": ["Unsalted almonds (20 g)", "1 orange"], "notes": "Potassium-rich fruit supports blood pressure control."},
            "dinner": {"time": "19:00", "items": ["Baked salmon or mackerel (120 g)", "Roasted vegetables with olive oil", "Brown rice (2/3 cup)"], "notes": "Oily fish twice a week for omega-3 fats."},
        },
        "foods_to_avoid": ["Processed and cured meats (bacon, sausages, ham)", "Fried chicken and fast food", "Sugary drinks and pastries", "Canned soups and high-sodium sauces", "Energy drinks and more than 1 cup of coffee per day"],
        "exercise_plan": {
            "allowed_activity": "Guided zone-2 stationary cycling",
            "duration_mins": 20,
            "sessions_per_day": 1,
            "frequency": "4-5 days per week",
            "intensity": "Zone 2 (conversational pace)",
            "max_hr_ceiling": 120,
            "warnings": ["Resting heart rate is elevated (sinus tachycardia); start each session only if resting HR is below 100 bpm.", "Stop if you feel palpitations or lightheadedness."],
        },
        "why_plan_changed_summary": "Baseline plan for the first recorded visit: high risk (61.5%) driven by sinus tachycardia, systolic BP 148 mmHg, cholesterol 235 mg/dL and glucose 142 mg/dL. A Mediterranean low-sodium DASH diet with guided zone-2 cycling targets all four.",
        "generated_by": "CardiacSense adaptive planner (seeded demo)",
    }

    eleanor_v2_plan = {
        "plan_title": "Mediterranean DASH Maintenance Plan (Chicken & Fish)",
        "diet_preference": "non_vegetarian",
        "daily_targets": {"sodium_mg": 1800, "protein_g": 75, "fluid_liters": 2.5, "calories": 1850, "potassium_mg": 4000, "fiber_g": 30},
        "meals": {
            "breakfast": {"time": "07:30", "items": ["Vegetable omelette (2 eggs) with spinach", "Whole-grain toast", "1 cup berries"], "notes": "Keep breakfast high in protein and fiber."},
            "lunch": {"time": "12:30", "items": ["Grilled fish tacos on corn tortillas with cabbage slaw", "Black bean and corn salad"], "notes": "Season with lime, garlic and chili instead of salt."},
            "snack": {"time": "16:00", "items": ["Hummus with carrot and cucumber sticks", "Unsalted pistachios (20 g)"], "notes": "Healthy fats keep energy steady between meals."},
            "dinner": {"time": "19:00", "items": ["Herb-roasted chicken thigh (skinless, 120 g)", "Lentil and vegetable soup (low sodium)", "Mixed green salad with olive oil"], "notes": "Oily fish at least twice a week."},
        },
        "foods_to_avoid": ["Processed and cured meats", "Fried and fast food", "Sugary drinks and desserts", "High-sodium sauces and canned soups"],
        "exercise_plan": {
            "allowed_activity": "Zone-2 cycling (outdoor or stationary) plus 2 light strength sessions per week",
            "duration_mins": 35,
            "sessions_per_day": 1,
            "frequency": "5 days per week",
            "intensity": "Zone 2 (conversational pace)",
            "max_hr_ceiling": 130,
            "warnings": ["Increase duration gradually by no more than 5 minutes per week.", "Stop if you feel chest discomfort, palpitations or dizziness."],
        },
        "why_plan_changed_summary": (
            "Your plan was relaxed because risk fell from 61.5% to 28.4% and your ECG rhythm is back to normal. "
            "Systolic BP dropped from 148 to 122 mmHg, so the sodium limit moves from 1,500 mg to 1,800 mg per day "
            "as a maintenance target. With sinus tachycardia resolved, zone-2 cycling increases from 20 to 35 "
            "minutes and the heart-rate ceiling rises from 120 to 130 bpm. Keep the Mediterranean DASH pattern that "
            "brought cholesterol down to 178 mg/dL and glucose to 104 mg/dL."
        ),
        "generated_by": "CardiacSense adaptive planner (seeded demo)",
    }

    return [
        {
            "patient": {
                "id": "pat_ramesh_001", "name": "Ramesh Kumar", "mrn": "MRN-2026-104", "age": 58, "gender": "M",
                "diet_preference": "vegetarian", "known_conditions": ["Hypertension", "Type 2 Diabetes"],
                "created_at": datetime(2026, 8, 10, 9, 0, 0),
            },
            "visits": [
                {
                    "id": "vis_ramesh_01", "visit_date": "2026-08-10", "hospital_name": "City General Hospital - Cardiology OPD",
                    "adherence_notes": "Baseline visit: no previous plan on record.",
                    "vitals": {"hr": 74, "bp_sys": 128, "bp_dia": 82, "spo2": 98, "rr": 15, "temp": 36.7},
                    "blood": {"glucose": 115, "cholesterol": 188, "creatinine": 1.0, "hrv": 46},
                    "ecg_profile": ECGProfile(name="Normal sinus rhythm", heart_rate=74, qrs_ms=92, pr_ms=164, seed=11001),
                    "ecg_filename": "ramesh_kumar_2026-08-10_code15_12lead.hdf5",
                    "predictions": _predictions(34.2, 60.1, 58, ramesh_v1_arr),
                    "shap": _shap([
                        ("age", "Age (58 yrs)", 58, 0.08),
                        ("glucose", "Fasting Glucose (115 mg/dL)", 115, 0.06),
                        ("bp_sys", "Systolic BP (128 mmHg)", 128, 0.04),
                        ("normal_ecg", "Normal Sinus Rhythm on 12-lead ECG", 1, -0.12),
                        ("creatinine", "Serum Creatinine (1.0 mg/dL)", 1.0, -0.05),
                        ("hrv", "Heart Rate Variability (46 ms)", 46, -0.04),
                        ("cholesterol", "Total Cholesterol (188 mg/dL)", 188, -0.03),
                        ("spo2", "SpO2 (98%)", 98, -0.02),
                    ]),
                    "diet_plan": ramesh_v1_plan,
                    "approved_by_staff": True,
                    "staff_notes": "Reviewed by hospital dietitian. Baseline plan approved.",
                    "created_at": datetime(2026, 8, 10, 11, 30, 0),
                },
                {
                    "id": "vis_ramesh_02", "visit_date": "2026-11-10", "hospital_name": "City General Hospital - Cardiology OPD",
                    "adherence_notes": "Skipped evening walks for 5 weeks and ate high-salt restaurant food frequently.",
                    "vitals": {"hr": 88, "bp_sys": 156, "bp_dia": 94, "spo2": 95, "rr": 18, "temp": 36.8},
                    "blood": {"glucose": 168, "cholesterol": 242, "creatinine": 1.7, "hrv": 28},
                    "ecg_profile": ECGProfile(
                        name="Atrial fibrillation with LBBB", heart_rate=88, qrs_ms=138, pr_ms=None,
                        rhythm="af", conduction="lbbb", st_amp_mv=-0.15, seed=11002,
                    ),
                    "ecg_filename": "ramesh_kumar_2026-11-10_code15_12lead.hdf5",
                    "predictions": _predictions(74.6, 69.4, 58, ramesh_v2_arr),
                    "shap": _shap([
                        ("lbbb_prob", "Left Bundle Branch Block (88%)", 0.88, 0.31),
                        ("creatinine", "Serum Creatinine (1.7 mg/dL)", 1.7, 0.29),
                        ("af_prob", "Atrial Fibrillation (84%)", 0.84, 0.27),
                        ("bp_sys", "Systolic BP (156 mmHg)", 156, 0.22),
                        ("cholesterol", "Total Cholesterol (242 mg/dL)", 242, 0.15),
                        ("glucose", "Fasting Glucose (168 mg/dL)", 168, 0.12),
                        ("hrv", "Heart Rate Variability (28 ms)", 28, 0.08),
                        ("age", "Age (58 yrs)", 58, 0.06),
                        ("diet_preference", "Vegetarian Diet", 1, -0.04),
                    ]),
                    "diet_plan": ramesh_v2_plan,
                    "approved_by_staff": False,
                    "staff_notes": "Pending hospital dietitian review.",
                    "created_at": datetime(2026, 11, 10, 10, 45, 0),
                },
            ],
        },
        {
            "patient": {
                "id": "pat_eleanor_002", "name": "Eleanor Vance", "mrn": "849-291-B", "age": 46, "gender": "F",
                "diet_preference": "non_vegetarian", "known_conditions": ["Hypertension", "Hypercholesterolemia"],
                "created_at": datetime(2026, 7, 20, 9, 0, 0),
            },
            "visits": [
                {
                    "id": "vis_eleanor_01", "visit_date": "2026-07-20", "hospital_name": "St. Mary's Heart Center",
                    "adherence_notes": "Baseline visit: no previous plan on record.",
                    "vitals": {"hr": 112, "bp_sys": 148, "bp_dia": 92, "spo2": 97, "rr": 19, "temp": 37.0},
                    "blood": {"glucose": 142, "cholesterol": 235, "creatinine": 1.1, "hrv": 31},
                    "ecg_profile": ECGProfile(name="Sinus tachycardia", heart_rate=112, qrs_ms=86, pr_ms=142, st_amp_mv=-0.05, seed=22001),
                    "ecg_filename": "eleanor_vance_2026-07-20_code15_12lead.hdf5",
                    "predictions": _predictions(61.5, 55.2, 46, eleanor_v1_arr),
                    "shap": _shap([
                        ("st_prob", "Sinus Tachycardia (79%)", 0.79, 0.21),
                        ("bp_sys", "Systolic BP (148 mmHg)", 148, 0.17),
                        ("hr", "Heart Rate (112 bpm)", 112, 0.14),
                        ("cholesterol", "Total Cholesterol (235 mg/dL)", 235, 0.13),
                        ("glucose", "Fasting Glucose (142 mg/dL)", 142, 0.11),
                        ("hrv", "Heart Rate Variability (31 ms)", 31, 0.06),
                        ("creatinine", "Serum Creatinine (1.1 mg/dL)", 1.1, 0.02),
                        ("age", "Age (46 yrs)", 46, -0.07),
                    ]),
                    "diet_plan": eleanor_v1_plan,
                    "approved_by_staff": True,
                    "staff_notes": "Approved by cardiac dietitian. Re-check lipids and BP in 3-4 months.",
                    "created_at": datetime(2026, 7, 20, 12, 15, 0),
                },
                {
                    "id": "vis_eleanor_02", "visit_date": "2026-11-05", "hospital_name": "St. Mary's Heart Center",
                    "adherence_notes": "Followed the Mediterranean low-sodium DASH plan strictly (home-cooked grilled chicken and fish, no processed meats) and completed guided zone-2 cycling 4-5 times per week.",
                    "vitals": {"hr": 70, "bp_sys": 122, "bp_dia": 78, "spo2": 99, "rr": 14, "temp": 36.6},
                    "blood": {"glucose": 104, "cholesterol": 178, "creatinine": 0.95, "hrv": 52},
                    "ecg_profile": ECGProfile(name="Normal sinus rhythm", heart_rate=70, qrs_ms=88, pr_ms=152, seed=22002),
                    "ecg_filename": "eleanor_vance_2026-11-05_code15_12lead.hdf5",
                    "predictions": _predictions(28.4, 47.5, 46, eleanor_v2_arr),
                    "shap": _shap([
                        ("normal_ecg", "Normal Sinus Rhythm Restored", 1, -0.14),
                        ("bp_sys", "Systolic BP (122 mmHg)", 122, -0.08),
                        ("cholesterol", "Total Cholesterol (178 mg/dL)", 178, -0.07),
                        ("hrv", "Heart Rate Variability (52 ms)", 52, -0.06),
                        ("age", "Age (46 yrs)", 46, -0.04),
                        ("glucose", "Fasting Glucose (104 mg/dL)", 104, 0.05),
                        ("bp_dia", "Diastolic BP (78 mmHg)", 78, 0.03),
                        ("creatinine", "Serum Creatinine (0.95 mg/dL)", 0.95, -0.03),
                    ]),
                    "diet_plan": eleanor_v2_plan,
                    "approved_by_staff": True,
                    "staff_notes": "Excellent adherence. Maintenance plan approved by cardiac dietitian.",
                    "created_at": datetime(2026, 11, 5, 11, 0, 0),
                },
            ],
        },
    ]


def seed_initial_database(db_session: Session) -> dict[str, Any]:
    """
    Insert the demo patients and visits if the ``patients`` table is empty.

    Delta summaries are computed by the Delta Engine from the chronological visit
    pairs rather than hard-coded. Commits on success, rolls back on failure.
    Returns ``{"seeded": bool, "patients": int, "visits": int}``.
    """
    existing = db_session.scalar(select(func.count()).select_from(Patient)) or 0
    if existing > 0:
        return {
            "seeded": False,
            "patients": int(existing),
            "visits": int(db_session.scalar(select(func.count()).select_from(Visit)) or 0),
        }

    dataset = _demo_patients()
    try:
        n_visits = 0
        for entry in dataset:
            p = entry["patient"]
            db_session.add(
                Patient(
                    id=p["id"], name=p["name"], mrn=p["mrn"], age=p["age"], gender=p["gender"],
                    diet_preference=p["diet_preference"], known_conditions=json_dumps(p["known_conditions"]),
                    created_at=p["created_at"],
                )
            )

            previous: dict[str, Any] | None = None
            for v in sorted(entry["visits"], key=lambda x: x["visit_date"]):
                expected_tier = severity_tier(v["predictions"]["overall_risk_score"])
                if v["predictions"]["severity_tier"] != expected_tier:
                    raise ValueError(f"{v['id']}: severity tier does not match risk score")

                ecg_metadata, heatmap = _ecg_bundle(v["ecg_profile"], v["ecg_filename"])
                flags = _flags_from_probs(v["predictions"]["arrhythmias"])
                record = {
                    "id": v["id"],
                    "visit_date": v["visit_date"],
                    "adherence_notes": v["adherence_notes"],
                    "vitals": v["vitals"],
                    "blood": v["blood"],
                    "code15_flags": flags,
                    "ecg_metadata": ecg_metadata,
                    "predictions": v["predictions"],
                }
                delta = compute_visit_deltas(previous, record)

                db_session.add(
                    Visit(
                        id=v["id"],
                        patient_id=p["id"],
                        visit_date=v["visit_date"],
                        hospital_name=v["hospital_name"],
                        adherence_notes=v["adherence_notes"],
                        vitals_json=json_dumps(v["vitals"]),
                        blood_json=json_dumps(v["blood"]),
                        code15_flags_json=json_dumps(flags),
                        ecg_metadata_json=json_dumps(ecg_metadata),
                        predictions_json=json_dumps(v["predictions"]),
                        shap_json=json_dumps(v["shap"]),
                        heatmap_json=json_dumps(heatmap),
                        delta_summary_json=json_dumps(delta),
                        diet_plan_json=json_dumps(v["diet_plan"]),
                        approved_by_staff=v["approved_by_staff"],
                        staff_notes=v["staff_notes"],
                        created_at=v["created_at"],
                    )
                )
                previous = record
                n_visits += 1
        db_session.commit()
    except Exception:
        db_session.rollback()
        raise

    return {"seeded": True, "patients": len(dataset), "visits": n_visits}


if __name__ == "__main__":
    from .database import init_db

    print(json.dumps(init_db(seed=True), indent=2))
    for path in generate_sample_code15_files():
        print("wrote", path)
