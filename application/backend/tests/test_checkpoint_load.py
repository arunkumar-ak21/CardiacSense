"""Task 1.5: safe, strict loading of the CODE-15% ResNet1d checkpoint (Req 2.1, 2.2, 3.5)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
CANDIDATES = (
    BACKEND_DIR.parents[1] / "model" / "model.pth",         # CardiacSense/model/
    BACKEND_DIR / "models" / "code15_model1" / "model.pth",  # optional local copy
)
BLOCKS = [(64, 4096), (128, 1024), (196, 256), (256, 64), (320, 16)]


def _checkpoint_path() -> Path:
    for path in CANDIDATES:
        if path.is_file():
            return path
    pytest.skip("model.pth not found under CardiacSense/model or backend/models/code15_model1")


def _safe_globals() -> list:
    try:
        from numpy._core.multiarray import scalar
    except ImportError:  # NumPy 1.x
        from numpy.core.multiarray import scalar
    allow: list = [(scalar, "numpy.core.multiarray.scalar"), scalar, np.dtype]
    float64_dtype = getattr(getattr(np, "dtypes", None), "Float64DType", None)
    if float64_dtype is not None:
        allow.append(float64_dtype)
    return allow


def test_checkpoint_loads_strictly_with_weights_only(torch_mod):
    torch = torch_mod
    from ml.resnet1d import ResNet1d

    path = _checkpoint_path()
    with torch.serialization.safe_globals(_safe_globals()):
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)

    assert isinstance(checkpoint, dict) and "model" in checkpoint
    state = checkpoint["model"]
    assert len(state) == 72

    model = ResNet1d((12, 4096), BLOCKS, 1, 17, 0.8)
    result = model.load_state_dict(state, strict=True)
    assert list(result.missing_keys) == []
    assert list(result.unexpected_keys) == []

    with torch.no_grad():
        out = model.eval()(torch.zeros(1, 12, 4096))
    assert tuple(out.shape) == (1, 1)
    assert bool(torch.isfinite(out).all())


def test_engine_reports_weights_loaded(torch_mod):
    _checkpoint_path()
    from ml.code15_engine import Code15Engine

    engine = Code15Engine()
    assert engine.weights_loaded is True
