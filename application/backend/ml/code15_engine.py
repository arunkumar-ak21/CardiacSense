"""CODE-15% 12-lead engine: weight resolution, configuration, preprocessing and inference.

Only the standard library is imported at module level. NumPy, SciPy and torch are imported
inside functions (torch only when it is available), so importing this module never loads
them before ``main.py`` has imported torch (Req 1.3).
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("vitalsense.code15_engine")

MODEL_DIR_ENV = "CODE15_MODEL_DIR"
BACKEND_DIR = Path(__file__).resolve().parent.parent
MODEL_SUBDIR = ("model",)  # CardiacSense/model/ (config.json, history.csv, model.pth)
MAX_PARENT_LEVELS = 6

SYNTH_SEED = MODEL_INIT_SEED = 1515

MODE_PRETRAINED = "PyTorch_ResNet1d_Pretrained_Model1"
MODE_FALLBACK = "Fallback_Heuristic"

DISCLAIMER = (
    "Heart age comes from a ResNet1d pre-trained on CODE-15% ECGs. Arrhythmia probabilities, "
    "fusion weights, the risk score and its attributions are uncalibrated research heuristics "
    "and are not clinically validated."
)


@dataclass(frozen=True)
class ResNetConfig:
    """Architecture settings read from ``config.json`` (Req 3.3)."""

    filters: tuple[int, ...]
    seq_lengths: tuple[int, ...]
    seq_length: int
    kernel_size: int
    dropout_rate: float

    @property
    def blocks_dim(self) -> list[tuple[int, int]]:
        return list(zip(self.filters, self.seq_lengths))

    @property
    def input_dim(self) -> tuple[int, int]:
        return (12, self.seq_length)


DEFAULT_CONFIG = ResNetConfig(
    filters=(64, 128, 196, 256, 320),
    seq_lengths=(4096, 1024, 256, 64, 16),
    seq_length=4096,
    kernel_size=17,
    dropout_rate=0.8,
)


# ---------------------------------------------------------------------------
# Weight resolution (Req 3.1, 3.2)
# ---------------------------------------------------------------------------

def _is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _absolute(path: Path) -> Path:
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        return Path(os.path.abspath(path))


def _model_dir_candidates(backend_dir: Path) -> list[Path]:
    """Candidates (b) and (c) of Req 3.1, in order."""
    base = _absolute(Path(backend_dir))
    # In-repo copy first (backend/models/code15_model1), then ../../model and the parent walk.
    candidates = [base / "models" / "code15_model1", base.parent.parent.joinpath(*MODEL_SUBDIR)]
    current = base
    for _ in range(MAX_PARENT_LEVELS + 1):  # backend_dir itself, then up to 6 parents
        candidates.append(current.joinpath(*MODEL_SUBDIR))
        parent = current.parent
        if parent == current:  # filesystem root
            break
        current = parent
    return candidates


def resolve_model_file(
    name: str, backend_dir: Path = BACKEND_DIR, env: Mapping[str, str] = os.environ
) -> Path | None:
    """Return the first regular file called ``name`` in the candidate directories, else None.

    Order: ``$CODE15_MODEL_DIR`` (when set and not empty), ``backend_dir/../../model``,
    then ``d/model`` for ``d`` in ``backend_dir`` and up to 6 parents. Never raises.
    """
    env_value = (env.get(MODEL_DIR_ENV) or "").strip()
    if env_value:
        env_dir = _absolute(Path(env_value))
        if not _is_dir(env_dir):
            logger.warning(
                "%s directory %s does not exist; looking for %s in the next candidate directory",
                MODEL_DIR_ENV, env_dir, name,
            )
        elif _is_file(env_dir / name):
            return env_dir / name
        else:
            logger.warning(
                "%s directory %s does not contain %s; looking in the next candidate directory",
                MODEL_DIR_ENV, env_dir, name,
            )

    for directory in _model_dir_candidates(backend_dir):
        candidate = directory / name
        if _is_file(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Config loading (Req 3.3, 3.4)
# ---------------------------------------------------------------------------

_CONFIG_KEYS = ("net_filter_size", "net_seq_lengh", "seq_length", "kernel_size", "dropout_rate")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(_is_int(v) and v > 0 for v in value)
    )


def _config_from_mapping(data: Mapping) -> ResNetConfig:
    """Validate the five config keys; raise ValueError naming the first failing rule."""
    missing = [k for k in _CONFIG_KEYS if k not in data]
    if missing:
        raise ValueError(f"missing key(s): {', '.join(missing)}")

    filters = data["net_filter_size"]
    seq_lengths = data["net_seq_lengh"]
    seq_length = data["seq_length"]
    kernel_size = data["kernel_size"]
    dropout_rate = data["dropout_rate"]

    if not _positive_int_list(filters):
        raise ValueError(f"net_filter_size must be a non-empty list of positive integers, got {filters!r}")
    if not _positive_int_list(seq_lengths):
        raise ValueError(f"net_seq_lengh must be a non-empty list of positive integers, got {seq_lengths!r}")
    if len(filters) != len(seq_lengths):
        raise ValueError(
            f"net_filter_size and net_seq_lengh must have equal length, got {len(filters)} and {len(seq_lengths)}"
        )
    if not (_is_int(seq_length) and seq_length > 0):
        raise ValueError(f"seq_length must be a positive integer, got {seq_length!r}")
    if not (_is_int(kernel_size) and kernel_size > 0 and kernel_size % 2 == 1):
        raise ValueError(f"kernel_size must be an odd positive integer, got {kernel_size!r}")
    if not (
        isinstance(dropout_rate, (int, float))
        and not isinstance(dropout_rate, bool)
        and math.isfinite(dropout_rate)
        and 0 <= dropout_rate < 1
    ):
        raise ValueError(f"dropout_rate must be a number in [0, 1), got {dropout_rate!r}")

    return ResNetConfig(
        filters=tuple(filters),
        seq_lengths=tuple(seq_lengths),
        seq_length=seq_length,
        kernel_size=kernel_size,
        dropout_rate=float(dropout_rate),
    )


def load_resnet_config(path: Path | None) -> tuple[ResNetConfig, str | None]:
    """Return ``(config, None)`` for a valid ``config.json``, else ``(DEFAULT_CONFIG, reason)``."""
    if path is None:
        reason = "config.json not found; using the built-in default configuration"
        logger.info(reason)
        return DEFAULT_CONFIG, reason

    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except OSError as exc:
        reason = f"could not read {path}: {type(exc).__name__}: {exc}"
    else:
        try:
            data = json.loads(text)
        except ValueError as exc:
            reason = f"{path} is not valid JSON: {exc}"
        else:
            if not isinstance(data, Mapping):
                reason = f"{path} must contain a JSON object, got {type(data).__name__}"
            else:
                try:
                    return _config_from_mapping(data), None
                except ValueError as exc:
                    reason = f"{path} has an invalid configuration: {exc}"

    logger.warning("%s; using the built-in default configuration", reason)
    return DEFAULT_CONFIG, reason


# ---------------------------------------------------------------------------
# ECG ingestion and preprocessing (Req 1.8, 4.1-4.20)
# ---------------------------------------------------------------------------

class ECGInputError(Exception):
    """A rejected ECG input. Routes turn it into ``{"detail": {"error", "detail", ...extra}}``."""

    def __init__(
        self, status_code: int, error: str, detail: str, extra: Mapping[str, object] | None = None
    ) -> None:
        super().__init__(f"{error}: {detail}")
        self.status_code = int(status_code)
        self.error = error
        self.detail = detail
        self.extra = dict(extra or {})

    def to_detail(self) -> dict[str, object]:
        """The body placed under the response's ``detail`` key."""
        return {"error": self.error, "detail": self.detail, **self.extra}


def _shape_text(shape: tuple[int, ...]) -> str:
    return "(" + ", ".join(str(int(d)) for d in shape) + ")"


class SignalPreprocessor:
    """Turns ``.npy``, ``.csv`` and HDF5 ECGs into the (4096, 12) float32 Canonical_Signal."""

    ACCEPTED = {".npy": "npy", ".csv": "csv", ".hdf5": "hdf5", ".h5": "hdf5"}
    MAX_BYTES = 20 * 1024 * 1024
    MIN_SAMPLE_RATE = 100.0
    MAX_SAMPLE_RATE = 2000.0
    TARGET_SAMPLE_RATE = 400.0
    TARGET_SAMPLES = 4096
    N_LEADS = 12
    MIN_SAMPLES = 400
    MAX_INVALID_FRACTION = 0.10
    UV_THRESHOLD = 50.0  # max |x| above this: microvolts -> millivolts
    V_THRESHOLD = 0.02  # 0 < max |x| below this: volts -> millivolts

    UPLOAD_RULES = (
        "accepted formats: .npy, .csv, .hdf5, .h5; maximum size: 20 MB (20971520 bytes); "
        "sample_rate: 100 to 2000 Hz"
    )
    H5PY_MISSING = "HDF5 support requires h5py; upload .npy or .csv instead"

    # -- step 1: extension, size and sample_rate (Req 4.15) -------------------

    def _invalid_upload(self, reason: str) -> ECGInputError:
        return ECGInputError(422, "invalid ECG upload", f"{reason}. {self.UPLOAD_RULES}")

    def _check_sample_rate(self, sample_rate: object) -> float | None:
        if sample_rate is None:
            return None
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, float)):
            raise self._invalid_upload(f"sample_rate must be a number, got {sample_rate!r}")
        rate = float(sample_rate)
        if not (math.isfinite(rate) and self.MIN_SAMPLE_RATE <= rate <= self.MAX_SAMPLE_RATE):
            raise self._invalid_upload(f"sample_rate {sample_rate!r} Hz is out of range")
        return rate

    def check_upload(self, filename: str, size: int | None, sample_rate: float | None) -> str:
        """Validate an upload before its contents are read; return the format name."""
        ext = os.path.splitext(filename or "")[1].lower()
        fmt = self.ACCEPTED.get(ext)
        if fmt is None:
            shown = ext or "(none)"
            raise self._invalid_upload(f"unsupported file extension {shown}")
        if size is not None and size > self.MAX_BYTES:
            raise self._invalid_upload(f"file is {size} bytes, larger than the limit")
        self._check_sample_rate(sample_rate)
        return fmt

    # -- step 2: file loading (Req 4.1-4.3, 4.14, 4.20) -----------------------

    @staticmethod
    def _parse_error(fmt: str, reason: str) -> ECGInputError:
        return ECGInputError(422, f"could not parse {fmt} file", reason)

    def load(self, data: bytes, fmt: str):
        """Parse file bytes into a numeric NumPy array (not yet validated for shape)."""
        loaders = {"npy": self._load_npy, "csv": self._load_csv, "hdf5": self._load_hdf5}
        loader = loaders.get(fmt)
        if loader is None:
            raise ECGInputError(
                422, "invalid ECG upload", f"unsupported format {fmt!r}. {self.UPLOAD_RULES}"
            )
        try:
            return loader(bytes(data))
        except ECGInputError:
            raise
        except Exception as exc:  # corrupt file of any kind
            raise self._parse_error(fmt, f"{type(exc).__name__}: {exc}") from exc

    def _load_npy(self, data: bytes):
        import io

        import numpy as np

        arr = np.load(io.BytesIO(data), allow_pickle=False)
        if not isinstance(arr, np.ndarray):  # an .npz archive
            close = getattr(arr, "close", None)
            if close is not None:
                close()
            raise self._parse_error("npy", "file is not a single NumPy array")
        if arr.dtype.kind not in "biuf":
            raise self._parse_error("npy", f"unsupported dtype {arr.dtype}; expected a numeric array")
        return arr

    def _load_csv(self, data: bytes):
        import csv
        import io

        import numpy as np

        text = data.decode("utf-8-sig")
        rows = [row for row in csv.reader(io.StringIO(text)) if any(f.strip() for f in row)]
        if not rows:
            raise self._parse_error("csv", "file has no rows")

        def numeric(field: str) -> bool:
            try:
                float(field)
            except ValueError:
                return False
            return True

        start = 0 if all(numeric(f) for f in rows[0]) else 1  # lead-name header
        body = rows[start:]
        if not body:
            raise self._parse_error("csv", "file has no data rows")

        values: list[list[float]] = []
        for offset, row in enumerate(body):
            line = start + offset + 1
            if len(row) != self.N_LEADS:
                raise self._parse_error(
                    "csv", f"row {line} has {len(row)} columns; expected {self.N_LEADS}"
                )
            try:
                values.append([float(f) for f in row])
            except ValueError:
                raise self._parse_error("csv", f"row {line} has a non-numeric field") from None
        return np.asarray(values, dtype=np.float64)

    def _load_hdf5(self, data: bytes):
        try:
            import h5py
        except ImportError:
            raise ECGInputError(422, "HDF5 support unavailable", self.H5PY_MISSING) from None
        import io

        with h5py.File(io.BytesIO(data), "r") as handle:
            dataset = handle.get("tracings")
            if not isinstance(dataset, h5py.Dataset):
                found: list[tuple[str, object]] = []

                def visit(name: str, obj: object) -> None:
                    if isinstance(obj, h5py.Dataset):
                        shape = obj.shape or ()
                        if (len(shape) >= 1 and shape[-1] == self.N_LEADS) or (
                            len(shape) >= 2 and shape[-2] == self.N_LEADS
                        ):
                            found.append((name, obj))

                handle.visititems(visit)
                if not found:
                    raise self._parse_error(
                        "hdf5", "no 'tracings' dataset and no dataset with 12 leads"
                    )
                dataset = sorted(found, key=lambda item: item[0])[0][1]
            arr = dataset[()]

        import numpy as np

        arr = np.asarray(arr)
        if arr.dtype.kind not in "biuf":
            raise self._parse_error("hdf5", f"unsupported dtype {arr.dtype}; expected a numeric array")
        return arr

    # -- steps 3-9: canonical signal (Req 4.4-4.13, 4.16-4.19) -----------------

    def _shape_error(self, shape: tuple[int, ...], reason: str) -> ECGInputError:
        return ECGInputError(
            422,
            "invalid ECG shape",
            f"received shape {_shape_text(shape)}; {reason}",
            {"shape": [int(d) for d in shape]},
        )

    def to_canonical(self, raw, sample_rate: float | None):
        """Return the (4096, 12) float32 Canonical_Signal for a loaded array."""
        import numpy as np

        rate = self._check_sample_rate(sample_rate)
        x = np.asarray(raw)
        received = tuple(x.shape)
        if x.dtype.kind not in "biuf":
            raise ECGInputError(
                422, "invalid ECG data", f"unsupported dtype {x.dtype}; expected a numeric array"
            )

        # 3. Leading-dimension squeeze.
        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]

        # 4. Shape validation and orientation: (n, 12) kept, (12, n) transposed.
        expected = f"expected (n, 12) or (12, n) with at least {self.MIN_SAMPLES} samples per lead"
        if x.ndim != 2:
            raise self._shape_error(received, expected)
        if x.shape[1] != self.N_LEADS:
            if x.shape[0] != self.N_LEADS:
                raise self._shape_error(received, expected)
            x = x.T
        if x.shape[0] < self.MIN_SAMPLES:
            raise self._shape_error(received, expected)

        # Precision in which the 0.02 threshold is read, so a float32 0.02 counts as 0.02.
        low = self.V_THRESHOLD
        if x.dtype.kind == "f" and x.dtype.itemsize < 8:
            low = float(x.dtype.type(self.V_THRESHOLD))
        x = np.array(x, dtype=np.float64, order="C")

        # 5. Non-finite rejection or repair.
        finite = np.isfinite(x)
        invalid = int(x.size - np.count_nonzero(finite))
        if invalid:
            fraction = invalid / x.size
            if invalid * 10 > x.size:
                raise ECGInputError(
                    422,
                    "too many invalid values",
                    f"{fraction:.4f} of the ECG values ({invalid} of {x.size}, {fraction:.2%}) are "
                    "NaN or infinite; at most 10% can be repaired",
                    {"invalid_fraction": round(fraction, 6)},
                )
            positions = np.arange(x.shape[0])
            for lead in range(self.N_LEADS):
                good = finite[:, lead]
                if good.all():
                    continue
                if not good.any():
                    x[:, lead] = 0.0
                    continue
                bad = ~good
                x[bad, lead] = np.interp(positions[bad], positions[good], x[good, lead])

        # 6. Unit scaling.
        peak = float(np.max(np.abs(x)))
        if peak > self.UV_THRESHOLD:
            x = x / 1000.0
        elif 0.0 < peak < low:
            x = x * 1000.0

        # 7. Resampling to 400 Hz.
        if rate is not None and rate != self.TARGET_SAMPLE_RATE:
            from scipy.signal import resample

            n_out = max(1, int(round(x.shape[0] * self.TARGET_SAMPLE_RATE / rate)))
            x = resample(x, n_out, axis=0)

        # 8. Centered crop or zero-pad.
        n = x.shape[0]
        target = self.TARGET_SAMPLES
        if n > target:
            start = (n - target) // 2
            x = x[start:start + target]
        elif n < target:
            padded = np.zeros((target, self.N_LEADS), dtype=np.float64)
            before = (target - n) // 2
            padded[before:before + n] = x
            x = padded

        # 9. Cast.
        return np.ascontiguousarray(x, dtype=np.float32)

    @staticmethod
    def to_model_array(canonical):
        """(4096, 12) Canonical_Signal -> contiguous (1, 12, 4096) float32 Model_Tensor array."""
        import numpy as np

        arr = np.asarray(canonical, dtype=np.float32)
        if arr.shape != (SignalPreprocessor.TARGET_SAMPLES, SignalPreprocessor.N_LEADS):
            raise ValueError(
                f"expected a canonical signal of shape (4096, 12), got {_shape_text(arr.shape)}"
            )
        return np.ascontiguousarray(arr.T[None])


# ---------------------------------------------------------------------------
# ECG sources and lead preview (Req 5)
# ---------------------------------------------------------------------------

SAMPLES_DIR = BACKEND_DIR.parent / "sample_code15_ecgs"
PREVIEW_POINTS = 400
PREVIEW_SAMPLE_RATE = 39.0625  # 400 values over 10.24 s
LEAD_NAMES = ["DI", "DII", "DIII", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CODE15_CODES = ("1dAVb", "RBBB", "LBBB", "SB", "AF", "ST")


def build_synthesis_profile(hr: float | None, flags: Mapping[str, bool]):
    """ECGProfile matching the submitted heart rate and CODE-15% flags (Req 5.4)."""
    from db.seed_and_samples import ECGProfile

    flags = flags or {}
    heart_rate = float(hr) if hr is not None else (50.0 if flags.get("SB") else 72.0)
    rhythm, conduction, qrs, pr, st = "sinus", "normal", 90.0, 160.0, 0.0
    if flags.get("AF"):
        rhythm, pr = "af", None
    if flags.get("LBBB"):
        conduction, qrs = "lbbb", 148.0
    elif flags.get("RBBB"):
        qrs = 130.0
    if flags.get("1dAVb") and not flags.get("AF"):
        pr = 245.0
    if flags.get("ST"):
        st = -0.15
    return ECGProfile(
        name="Synthesized from submitted vitals and flags",
        heart_rate=heart_rate, qrs_ms=qrs, pr_ms=pr, rhythm=rhythm,
        conduction=conduction, st_amp_mv=st, seed=SYNTH_SEED,
    )


def synthesize_from_flags(hr: float | None, flags: Mapping[str, bool]):
    """(4096, 12) float32 synthetic ECG for the request (deterministic, Req 5.8)."""
    import numpy as np

    from db.seed_and_samples import synthesize_12lead_ecg

    return np.asarray(synthesize_12lead_ecg(build_synthesis_profile(hr, flags))["signal"], dtype=np.float32)


def resolve_sample_path(name: str, samples_dir: Path = SAMPLES_DIR) -> Path | None:
    """Path of a sample file directly inside ``samples_dir``, or None when invalid/absent."""
    from companion_requests import RequestValidationError, validate_sample_filename

    try:
        validate_sample_filename(name)
    except (RequestValidationError, ValueError):
        return None
    path = Path(samples_dir) / name
    return path if _is_file(path) else None


def list_sample_files(samples_dir: Path = SAMPLES_DIR) -> list[str]:
    if not _is_dir(Path(samples_dir)):
        return []
    return sorted(
        p.name for p in Path(samples_dir).iterdir()
        if p.is_file() and p.suffix.lower() in SignalPreprocessor.ACCEPTED
    )


def build_leads_preview(canonical) -> dict[str, list[float]]:
    """12 leads x 400 points (39.0625 Hz) for the frontend plot (Req 5.6)."""
    import numpy as np
    from scipy.signal import resample_poly

    reduced = resample_poly(np.asarray(canonical, dtype=np.float64), 25, 256, axis=0)[:PREVIEW_POINTS]
    reduced = np.nan_to_num(reduced, nan=0.0, posinf=0.0, neginf=0.0)
    return {lead: np.round(reduced[:, i], 4).tolist() for i, lead in enumerate(LEAD_NAMES)}


# ---------------------------------------------------------------------------
# Code15Engine (Req 3.5-3.9, 7-11)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EngineInput:
    patient: Mapping            # Patient.to_dict()
    values: Mapping             # vitals + blood values (None when missing)
    flags: Mapping              # submitted CODE-15% flags
    canonical: object           # (4096, 12) float32 array
    source: str                 # "upload" | "sample" | "synthetic"
    filename: str | None
    fmt: str                    # "npy" | "csv" | "hdf5" | "synthetic"


@dataclass(frozen=True)
class EngineResult:
    predictions: dict
    shap: list
    heatmap: list
    code15_flags: dict
    ecg_metadata: dict


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def _checkpoint_safe_globals() -> list:
    """Allow-list needed to read model/model.pth with ``weights_only=True``."""
    import numpy as np

    try:
        from numpy._core.multiarray import scalar
    except ImportError:  # NumPy 1.x
        from numpy.core.multiarray import scalar
    allow: list = [(scalar, "numpy.core.multiarray.scalar"), scalar, np.dtype]
    float64_dtype = getattr(getattr(np, "dtypes", None), "Float64DType", None)
    if float64_dtype is not None:
        allow.append(float64_dtype)
    return allow


class Code15Engine:
    """Process-wide CODE-15% inference engine (use ``get_code15_engine()``)."""

    def __init__(self) -> None:
        import threading

        self.lock = threading.Lock()
        self.preprocessor = SignalPreprocessor()
        self.model = None
        self.fusion = None
        self.ready = False
        self.weights_loaded = False
        self.weights_path: str | None = None
        self.config_path: str | None = None
        self.model_version = "code15-unavailable"
        self._w_deep = None

        from ml.clinical_measurements import ClinicalMeasurer

        self.measurer = ClinicalMeasurer()

        if not _torch_available():
            logger.error("torch is not available; Code15Engine runs DSP-only fallbacks")
            return

        import numpy as np
        import torch

        from ml.multimodal_fusion import build_fusion_module
        from ml.resnet1d import ResNet1d

        config_file = resolve_model_file("config.json", BACKEND_DIR)
        self.config_path = str(config_file) if config_file else None
        config, _reason = load_resnet_config(config_file)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(MODEL_INIT_SEED)
            model = ResNet1d(
                input_dim=config.input_dim, blocks_dim=config.blocks_dim, n_classes=1,
                kernel_size=config.kernel_size, dropout_rate=config.dropout_rate,
            )
        fresh_state = {k: v.clone() for k, v in model.state_dict().items()}

        weights_file = resolve_model_file("model.pth", BACKEND_DIR)
        if weights_file is None:
            logger.warning("model.pth not found; using seeded random ResNet1d weights")
        else:
            try:
                with torch.serialization.safe_globals(_checkpoint_safe_globals()):
                    checkpoint = torch.load(str(weights_file), map_location="cpu", weights_only=True)
                state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
                own = model.state_dict()
                mismatched = [k for k, v in state.items() if k in own and tuple(own[k].shape) != tuple(v.shape)]
                if mismatched:
                    raise ValueError(f"shape mismatch for {mismatched[:3]}")
                model.load_state_dict(state, strict=True)
                self.weights_loaded = True
                self.weights_path = str(weights_file)
                logger.info("Loaded ResNet1d weights from %s", weights_file)
            except Exception as exc:
                logger.warning("Could not load %s (%s: %s); using seeded random weights",
                               weights_file, type(exc).__name__, exc)
                model.load_state_dict(fresh_state)
                self.weights_loaded = False

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model
        self.fusion = build_fusion_module()
        self._w_deep = np.random.default_rng(2024).normal(0.0, (1.0 / 320) ** 0.5, size=(6, 320))
        self.model_version = (
            f"code15-resnet1d:{'model-1' if self.weights_loaded else 'random-init-seed1515'}"
            "|fusion:untrained-v1"
        )
        self.ready = True

    # -- ECG source helpers ----------------------------------------------------

    def canonical_from_upload(self, filename: str, data: bytes, sample_rate: float | None):
        fmt = self.preprocessor.check_upload(filename, len(data), sample_rate)
        if not data:
            raise ECGInputError(422, "empty upload", "The uploaded ECG file is empty.")
        raw = self.preprocessor.load(data, fmt)
        return self.preprocessor.to_canonical(raw, sample_rate), fmt

    def canonical_from_sample(self, path: Path):
        fmt = self.preprocessor.check_upload(path.name, None, None)
        raw = self.preprocessor.load(path.read_bytes(), fmt)
        return self.preprocessor.to_canonical(raw, 400.0), fmt

    # -- inference --------------------------------------------------------------

    def _deep_terms(self, features) -> dict[str, float] | None:
        import numpy as np

        if not self.weights_loaded or self._w_deep is None:
            return None
        pooled = np.asarray(features, dtype=np.float64).mean(axis=-1).reshape(-1)  # (320,)
        pooled = pooled / (np.linalg.norm(pooled) + 1e-6)
        terms = np.tanh(self._w_deep @ pooled * 3.0)
        return {code: float(terms[i]) for i, code in enumerate(CODE15_CODES)}

    def predict(self, inp: EngineInput) -> EngineResult:
        import numpy as np

        from db.delta_engine import severity_tier
        from ml import code15_scoring as scoring
        from ml.saliency import compute_saliency

        canonical = np.asarray(inp.canonical, dtype=np.float32)
        x = SignalPreprocessor.to_model_array(canonical)  # (1, 12, 4096)
        age = float(inp.patient.get("age"))

        with self.lock:
            measured = self.measurer.measure(canonical)
            dsp = dict(measured.measurements)

            model_out, features, forward_ms = None, None, None
            if self.model is not None:
                import time

                import torch

                x_tensor = torch.from_numpy(x)  # (1, 12, 4096) float32
                t0 = time.perf_counter()
                with torch.no_grad():
                    # Same computation as model(x_tensor); also returns the (1, 320, 16) features.
                    out, feats = self.model.forward_with_features(x_tensor)
                forward_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                model_out = float(out[0, 0])
                features = feats[0].numpy()
            ran_pretrained = self.weights_loaded and model_out is not None and math.isfinite(model_out)

            heart_age, gap, age_source = scoring.resolve_heart_age(model_out, age, dsp, self.weights_loaded)
            deep = self._deep_terms(features) if features is not None else None
            probs = scoring.calibrate_arrhythmias(inp.flags, dsp, deep)
            present = scoring.present_codes(inp.flags, probs)
            cv, imputed = scoring.build_clinical_vector(inp.patient, inp.values, gap, dsp.get("st_amp"), present)
            clinical = scoring.clinical_risk(cv, present)
            shap = scoring.attribution_drivers(cv, present)

            neural, attention = None, None
            if self.fusion is not None and features is not None:
                try:
                    import torch

                    with torch.no_grad():
                        fused = self.fusion(
                            torch.from_numpy(features[None].astype(np.float32)),
                            torch.tensor([cv.as_list()], dtype=torch.float32),
                        )
                    neural = float(fused.neural_adjustment[0])
                    attention = np.round(fused.attention[0].numpy().astype(np.float64), 4).tolist()
                except Exception as exc:
                    logger.error("Fusion forward failed (%s); neural adjustment set to 0", type(exc).__name__)
                    neural, attention = None, None

            overall, clinical_part, neural_part = scoring.combine_risk(clinical, neural)
            heatmap = compute_saliency(x[0], dsp, list(measured.r_peaks), self.model)

        predictions = {
            "overall_risk_score": overall,
            "severity_tier": severity_tier(overall),
            "predicted_heart_age": heart_age,
            "chronological_age": int(age) if float(age).is_integer() else age,
            "heart_age_gap": gap,
            "heart_age_source": age_source,
            "arrhythmias": probs,
            "risk_components": {
                "clinical_risk_component": clinical_part,
                "neural_risk_adjustment": neural_part,
            },
            "cross_attention_weights": attention,
            "imputed_fields": imputed,
            "weights_loaded": self.weights_loaded,
            "model_execution_mode": MODE_PRETRAINED if ran_pretrained else MODE_FALLBACK,
            "model_execution": {
                "weights_path": self.weights_path,
                "input_tensor_shape": [1, 12, 4096],
                "feature_map_shape": [1, 320, 16] if features is not None else None,
                "resnet1d_raw_output_years": round(model_out, 3) if model_out is not None and math.isfinite(model_out) else None,
                "forward_pass_ms": forward_ms,
            },
            "model_version": self.model_version,
            "source": "code15_engine",
            "clinical_validation_disclaimer": DISCLAIMER,
        }
        ecg_metadata = {
            "filename": inp.filename,
            "source": inp.source,
            "format": inp.fmt,
            "sample_rate": 400,
            "n_samples": 4096,
            "lead_names": list(LEAD_NAMES),
            "units": "mV",
            "synthetic": inp.source in ("sample", "synthetic"),
            "dsp_measurements": dsp,
            "preview_sample_rate": PREVIEW_SAMPLE_RATE,
            "leads_preview": build_leads_preview(canonical),
        }
        return EngineResult(
            predictions=predictions,
            shap=shap,
            heatmap=heatmap,
            code15_flags=scoring.build_code15_flags(inp.flags, probs),
            ecg_metadata=ecg_metadata,
        )


_ENGINE: Code15Engine | None = None
_ENGINE_LOCK = __import__("threading").Lock()


def get_code15_engine() -> Code15Engine:
    """Singleton accessor: one Code15Engine (and one ResNet1d) per process."""
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = Code15Engine()
    return _ENGINE
