"""DSP clinical measurements on a 12-lead Canonical_Signal (NumPy and SciPy only).

Ported from ``ECG-Foundation-Engine/web_app/backend/app/core/clinical_measurements.py``
and rebuilt around the algorithm in the code15-multimodal-engine design. The input is a
``(n, 12)`` array in mV sampled at 400 Hz with leads in CODE-15% order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

FS = 400

LEAD_NAMES = ("DI", "DII", "DIII", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
DII_INDEX = 1

# R-peak detection (lead DII).
R_BAND_HZ = (10.0, 25.0)
R_BAND_ORDER = 2
R_INTEGRATION_MS = 150
R_HEIGHT_FRAC = 0.30
R_HEIGHT_PCT = 99
R_MIN_DISTANCE_MS = 200
T_WAVE_WINDOW_MS = 360
T_WAVE_HEIGHT_FRAC = 0.5
R_REFINE_BAND_HZ = (0.5, 40.0)
R_REFINE_ORDER = 2
R_REFINE_MS = 50

# Multi-lead velocity used for QRS delineation and per-lead ST onsets.
VEL_LOWPASS_HZ = 40.0
VEL_LOWPASS_ORDER = 4
VEL_SMOOTH_MS = 8

# QRS onset/offset (global, on the summed velocity).
QRS_FRAC = 0.02
FLOOR_PCT = 75
QUIET_MS = 12
QRS_BACK_MS = 160
QRS_FORWARD_MS = 200
QRS_VMAX_MS = 60

# P waves (lead DII).
P_LOWPASS_HZ = 15.0
P_LOWPASS_ORDER = 4
P_MAX_LOOKBACK_MS = 300
P_RR_FRAC = 0.45
P_GAP_MS = 20
P_MIN_AMP = 0.05
P_OPPOSITE_FRAC = 0.5
P_PROMINENCE_FRAC = 0.5
P_FRAC = 0.20
P_LAG_TOL_MS = 30
P_MIN_BEAT_FRAC = 0.5

# ST deviation at J + 60 ms.
ST_J_OFFSET_MS = 60
ST_BASELINE_MS = 20
LEAD_FRAC = 0.05
LEAD_BACK_MS = 40
EXCLUDE_V1_V3_WHEN_WIDE_QRS = True
WIDE_QRS_MS = 120.0
V1_V3 = ("V1", "V2", "V3")

# Status thresholds and output clamps.
ST_ELEVATION_MV = 0.1
ST_DEPRESSION_MV = -0.05
HR_RANGE = (0.0, 300.0)
QRS_RANGE = (0.0, 300.0)
PR_RANGE = (0.0, 400.0)
ST_RANGE = (-2.0, 2.0)
FLAT_PTP = 1e-6
MIN_R_PEAKS = 2


@dataclass(frozen=True)
class Beat:
    r_peak: int
    qrs_onset: int
    qrs_offset: int
    p_onset: int | None


@dataclass(frozen=True)
class MeasurementResult:
    measurements: dict
    r_peaks: tuple[int, ...]
    beats: tuple[Beat, ...]
    st_lead: str | None


def _ms_to_samples(ms: float, fs: int) -> int:
    return max(1, int(round(ms * fs / 1000.0)))


def _butter_filter(
    x: np.ndarray, fs: int, order: int, low: float | None = None, high: float | None = None
) -> np.ndarray:
    """Zero-phase Butterworth filter along axis 0 (band-pass, low-pass or high-pass)."""
    if low is not None and high is not None:
        b, a = butter(order, [low, high], btype="bandpass", fs=fs)
    elif high is not None:
        b, a = butter(order, high, btype="lowpass", fs=fs)
    elif low is not None:
        b, a = butter(order, low, btype="highpass", fs=fs)
    else:
        raise ValueError("at least one cutoff frequency is required")
    return filtfilt(b, a, np.asarray(x, dtype=np.float64), axis=0)


def _moving_average(x: np.ndarray, width: int) -> np.ndarray:
    """Centered moving average along axis 0 with the output length equal to the input."""
    kernel = np.ones(width, dtype=np.float64) / width
    if x.ndim == 1:
        return np.convolve(x, kernel, mode="same")
    return np.stack([np.convolve(x[:, j], kernel, mode="same") for j in range(x.shape[1])], axis=1)


class ClinicalMeasurer:
    def __init__(self, fs: int = FS) -> None:
        self.fs = int(fs)

    @staticmethod
    def not_measurable() -> dict:
        return {
            "heart_rate": 0.0,
            "qrs_ms": 0.0,
            "pr_ms": 0.0,
            "st_amp": 0.0,
            "st_status": "Normal",
            "rr_irregularity": 0.0,
            "measurable": False,
        }

    def _lead_dii(self, signal: np.ndarray) -> np.ndarray:
        sig = np.asarray(signal, dtype=np.float64)
        if sig.ndim != 2 or sig.shape[1] != len(LEAD_NAMES):
            raise ValueError(f"expected signal of shape (n, 12); received {sig.shape}")
        return sig[:, DII_INDEX]

    def detect_r_peaks(self, signal: np.ndarray) -> np.ndarray:
        """Return sorted R-peak sample indices detected on lead DII."""
        fs = self.fs
        dii = self._lead_dii(signal)
        empty = np.zeros(0, dtype=np.int64)
        distance = _ms_to_samples(R_MIN_DISTANCE_MS, fs)
        if dii.shape[0] < distance or not np.all(np.isfinite(dii)):
            return empty

        band = _butter_filter(dii, fs, R_BAND_ORDER, low=R_BAND_HZ[0], high=R_BAND_HZ[1])
        energy = np.gradient(band) ** 2
        mwa = _moving_average(energy, _ms_to_samples(R_INTEGRATION_MS, fs))
        level = float(np.percentile(mwa, R_HEIGHT_PCT))
        if not np.isfinite(level) or level <= 0.0:
            return empty

        candidates, props = find_peaks(mwa, height=R_HEIGHT_FRAC * level, distance=distance)
        if candidates.size == 0:
            return empty

        kept = self._reject_t_waves(candidates, props["peak_heights"])
        refined = self._refine_peaks(dii, kept)
        return refined

    def _reject_t_waves(self, candidates: np.ndarray, heights: np.ndarray) -> np.ndarray:
        """Drop low candidates that follow a kept peak within the T-wave window."""
        window = _ms_to_samples(T_WAVE_WINDOW_MS, self.fs)
        median_height = float(np.median(heights))
        kept_idx: list[int] = []
        kept_h: list[float] = []
        for idx, h in zip(candidates.tolist(), heights.tolist()):
            if kept_idx and idx - kept_idx[-1] < window:
                if h < T_WAVE_HEIGHT_FRAC * median_height or h < T_WAVE_HEIGHT_FRAC * kept_h[-1]:
                    continue
            kept_idx.append(idx)
            kept_h.append(h)
        return np.asarray(kept_idx, dtype=np.int64)

    def _refine_peaks(self, dii: np.ndarray, peaks: np.ndarray) -> np.ndarray:
        """Move each peak to argmax |DII 0.5-40 Hz| within +/-50 ms, keeping >= 200 ms spacing."""
        fs = self.fs
        n = dii.shape[0]
        wide = np.abs(
            _butter_filter(dii, fs, R_REFINE_ORDER, low=R_REFINE_BAND_HZ[0], high=R_REFINE_BAND_HZ[1])
        )
        half = _ms_to_samples(R_REFINE_MS, fs)
        distance = _ms_to_samples(R_MIN_DISTANCE_MS, fs)
        refined: list[int] = []
        for p in peaks.tolist():
            lo, hi = max(0, p - half), min(n, p + half + 1)
            r = lo + int(np.argmax(wide[lo:hi]))
            if refined and r - refined[-1] < distance:
                if wide[r] > wide[refined[-1]]:
                    refined[-1] = r
                continue
            refined.append(r)
        return np.asarray(refined, dtype=np.int64)

    def rr_intervals_ms(self, r_peaks: np.ndarray) -> np.ndarray:
        r = np.asarray(r_peaks, dtype=np.float64)
        if r.size < 2:
            return np.zeros(0, dtype=np.float64)
        return np.diff(r) / self.fs * 1000.0

    def rate_measures(self, r_peaks: np.ndarray) -> tuple[float, float]:
        """Return ``(heart_rate bpm, rr_irregularity)``; ``(0.0, 0.0)`` with fewer than 2 peaks."""
        rr = self.rr_intervals_ms(r_peaks)
        if rr.size == 0:
            return 0.0, 0.0
        median_rr = float(np.median(rr))
        mean_rr = float(np.mean(rr))
        heart_rate = 60000.0 / median_rr if median_rr > 0 else 0.0
        rr_irregularity = float(np.std(rr, ddof=0)) / mean_rr if mean_rr > 0 else 0.0
        return float(heart_rate), float(rr_irregularity)

    # ------------------------------------------------------------------ velocity

    def _lead_velocity(self, signal: np.ndarray) -> np.ndarray:
        """Per-lead smoothed absolute slope in mV/s, shape ``(n, 12)``."""
        fs = self.fs
        lp = _butter_filter(signal, fs, VEL_LOWPASS_ORDER, high=VEL_LOWPASS_HZ)
        slope = np.abs(np.gradient(lp, axis=0)) * fs
        return _moving_average(slope, _ms_to_samples(VEL_SMOOTH_MS, fs))

    # ------------------------------------------------------------ QRS delineation

    def delineate_qrs(self, signal: np.ndarray, r_peaks: np.ndarray) -> list[Beat]:
        """Return one ``Beat`` per R-peak whose search window fits inside the signal."""
        sig = np.asarray(signal, dtype=np.float64)
        return self._delineate(self._lead_velocity(sig).sum(axis=1), r_peaks)

    def _delineate(self, vel: np.ndarray, r_peaks: np.ndarray) -> list[Beat]:
        fs = self.fs
        n = vel.shape[0]
        back = _ms_to_samples(QRS_BACK_MS, fs)
        forward = _ms_to_samples(QRS_FORWARD_MS, fs)
        quiet = _ms_to_samples(QUIET_MS, fs)
        half_vmax = _ms_to_samples(QRS_VMAX_MS, fs)
        floor = float(np.percentile(vel, FLOOR_PCT))

        beats: list[Beat] = []
        for r in np.asarray(r_peaks, dtype=np.int64).tolist():
            if r - back - quiet < 0 or r + forward + quiet >= n:
                continue
            vmax = float(np.max(vel[r - half_vmax : r + half_vmax + 1]))
            thr = floor + QRS_FRAC * (vmax - floor)

            onset = r - back
            for i in range(r, r - back - 1, -1):
                if np.all(vel[i - quiet + 1 : i + 1] < thr):
                    onset = i
                    break
            offset = r + forward
            for i in range(r, r + forward + 1):
                if np.all(vel[i : i + quiet] < thr):
                    offset = i
                    break
            beats.append(Beat(r_peak=r, qrs_onset=onset, qrs_offset=offset, p_onset=None))
        return beats

    def qrs_duration_ms(self, beats: list[Beat]) -> float:
        """Median QRS width in ms; NaN when no beat was delineated."""
        if not beats:
            return float("nan")
        widths = np.array([b.qrs_offset - b.qrs_onset for b in beats], dtype=np.float64)
        return float(np.median(widths)) / self.fs * 1000.0

    # ------------------------------------------------------------------ P waves

    def detect_p_onsets(
        self, signal: np.ndarray, beats: list[Beat], r_peaks: np.ndarray
    ) -> list[Beat]:
        """Return ``beats`` with ``p_onset`` set where a single DII P wave precedes the QRS."""
        fs = self.fs
        dii = self._lead_dii(signal)
        ii = _butter_filter(dii, fs, P_LOWPASS_ORDER, high=P_LOWPASS_HZ)
        gap = _ms_to_samples(P_GAP_MS, fs)
        max_back = _ms_to_samples(P_MAX_LOOKBACK_MS, fs)

        peaks = np.asarray(r_peaks, dtype=np.int64)
        rr = np.diff(peaks)
        median_rr = float(np.median(rr)) if rr.size else float("inf")

        found: dict[int, tuple[int, int]] = {}  # beat index -> (p_onset, p_peak)
        for k, beat in enumerate(beats):
            pos = int(np.searchsorted(peaks, beat.r_peak))
            rr_prev = float(peaks[pos] - peaks[pos - 1]) if 0 < pos < peaks.size else median_rr
            lookback = int(min(max_back, round(P_RR_FRAC * rr_prev)))
            onset = beat.qrs_onset
            start, end = onset - lookback, onset - gap
            if start < 0 or end - start < 3:
                continue
            baseline = float(np.median(ii[onset - gap : onset]))
            detection = self._single_p_wave(ii[start:end] - baseline)
            if detection is not None:
                pk, j = detection
                found[k] = (start + j, start + pk)

        if found:
            lags = {k: beats[k].qrs_onset - p_peak for k, (_, p_peak) in found.items()}
            median_lag = float(np.median(list(lags.values())))
            tol = _ms_to_samples(P_LAG_TOL_MS, fs)
            found = {k: v for k, v in found.items() if abs(lags[k] - median_lag) <= tol}

        return [
            Beat(b.r_peak, b.qrs_onset, b.qrs_offset, found[k][0]) if k in found else b
            for k, b in enumerate(beats)
        ]

    @staticmethod
    def _single_p_wave(x: np.ndarray) -> tuple[int, int] | None:
        """Return ``(peak, onset)`` indices into ``x`` for one clear P wave, else ``None``."""
        idx = int(np.argmax(np.abs(x)))
        amp = float(abs(x[idx]))
        if not np.isfinite(amp) or amp < P_MIN_AMP:
            return None
        y = np.sign(x[idx]) * x
        if float(np.max(-y)) >= P_OPPOSITE_FRAC * amp:
            return None
        found, _ = find_peaks(y, prominence=P_PROMINENCE_FRAC * amp)
        if found.size != 1:
            return None
        pk = int(found[0])
        ymin = float(np.min(y[:pk]))
        level = ymin + P_FRAC * (float(y[pk]) - ymin)
        j = pk
        while j > 0 and y[j] > level:
            j -= 1
        return pk, j

    def pr_interval_ms(self, beats: list[Beat], r_peak_count: int) -> float | None:
        """Median P-onset to QRS-onset interval in ms, or ``None`` with too few P waves."""
        with_p = [b for b in beats if b.p_onset is not None]
        if not with_p or len(with_p) < P_MIN_BEAT_FRAC * r_peak_count:
            return None
        pr = np.array([b.qrs_onset - b.p_onset for b in with_p], dtype=np.float64)
        return float(np.median(pr)) / self.fs * 1000.0

    # --------------------------------------------------------------- ST segment

    def analyze_st_segment(
        self, signal: np.ndarray, beats: list[Beat], qrs_ms: float
    ) -> tuple[float, str | None]:
        """Return ``(st_amp mV, lead name)`` from the candidate lead with the largest |median|."""
        sig = np.asarray(signal, dtype=np.float64)
        medians = self.st_lead_medians(sig, beats, self._lead_velocity(sig))
        excluded: tuple[str, ...] = ()
        if EXCLUDE_V1_V3_WHEN_WIDE_QRS and np.isfinite(qrs_ms) and qrs_ms >= WIDE_QRS_MS:
            excluded = V1_V3
        best_lead: str | None = None
        best = 0.0
        for name, value in zip(LEAD_NAMES, medians.tolist()):
            if name in excluded or not np.isfinite(value):
                continue
            if best_lead is None or abs(value) > abs(best):
                best_lead, best = name, float(value)
        return best, best_lead

    def st_lead_medians(
        self, sig: np.ndarray, beats: list[Beat], lead_vel: np.ndarray
    ) -> np.ndarray:
        """Per-lead median J+60 ms deviation over beats; NaN for leads without a usable beat."""
        fs = self.fs
        n, n_leads = sig.shape
        j_off = _ms_to_samples(ST_J_OFFSET_MS, fs)
        base_len = _ms_to_samples(ST_BASELINE_MS, fs)
        max_back = _ms_to_samples(LEAD_BACK_MS, fs)
        quiet = _ms_to_samples(QUIET_MS, fs)

        values: list[list[float]] = [[] for _ in range(n_leads)]
        for beat in beats:
            j60 = beat.qrs_offset + j_off
            if j60 >= n:
                continue
            onset, offset = beat.qrs_onset, beat.qrs_offset
            for lead in range(n_leads):
                v = lead_vel[:, lead]
                thr = LEAD_FRAC * float(np.max(v[onset : offset + 1]))
                lead_onset = onset - max_back
                for i in range(onset, onset - max_back - 1, -1):
                    if i - quiet + 1 >= 0 and np.all(v[i - quiet + 1 : i + 1] < thr):
                        lead_onset = i
                        break
                if lead_onset - base_len < 0:
                    continue
                baseline = float(np.median(sig[lead_onset - base_len : lead_onset, lead]))
                values[lead].append(float(sig[j60, lead]) - baseline)

        return np.array(
            [float(np.median(v)) if v else float("nan") for v in values], dtype=np.float64
        )

    # ------------------------------------------------------------------ measure

    def measure(self, signal: np.ndarray) -> MeasurementResult:
        """Compute DSP_Measurements plus the R-peaks, beats and ST lead behind them."""
        sig = np.asarray(signal, dtype=np.float64)
        dii = self._lead_dii(sig)
        unmeasurable = MeasurementResult(self.not_measurable(), (), (), None)
        if not np.all(np.isfinite(sig)) or float(np.ptp(dii)) < FLAT_PTP:
            return unmeasurable
        r_peaks = self.detect_r_peaks(sig)
        if r_peaks.size < MIN_R_PEAKS:
            return unmeasurable

        heart_rate, rr_irregularity = self.rate_measures(r_peaks)
        beats = self.delineate_qrs(sig, r_peaks)
        qrs_ms = self.qrs_duration_ms(beats)
        beats = self.detect_p_onsets(sig, beats, r_peaks)
        pr_ms = self.pr_interval_ms(beats, int(r_peaks.size))
        st_amp, st_lead = self.analyze_st_segment(sig, beats, qrs_ms)

        st_amp = round(_clamp(st_amp, ST_RANGE), 3)
        measurements = {
            "heart_rate": round(_clamp(heart_rate, HR_RANGE), 1),
            "qrs_ms": round(_clamp(qrs_ms, QRS_RANGE), 1),
            "pr_ms": None if pr_ms is None else round(_clamp(pr_ms, PR_RANGE), 1),
            "st_amp": st_amp,
            "st_status": _st_status(st_amp),
            "rr_irregularity": round(_clamp(rr_irregularity, (0.0, float("inf"))), 3),
            "measurable": True,
        }
        return MeasurementResult(
            measurements=measurements,
            r_peaks=tuple(int(r) for r in r_peaks.tolist()),
            beats=tuple(beats),
            st_lead=st_lead,
        )


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    """Clamp to ``bounds``; non-finite values become the ``not_measurable()`` default 0.0."""
    value = float(value)
    if not np.isfinite(value):
        return 0.0
    return float(min(max(value, bounds[0]), bounds[1]))


def _st_status(st_amp: float) -> str:
    if st_amp > ST_ELEVATION_MV:
        return "Elevation"
    if st_amp < ST_DEPRESSION_MV:
        return "Depression"
    return "Normal"
