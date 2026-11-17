"""
Runtime environment checks for the pure-PyTorch backend.

Validates: Requirements 1.1, 1.2, 1.6, 1.7
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent
REQUIREMENTS_TXT = BACKEND_DIR / "requirements.txt"

# Directory names never scanned for imports (caches and virtual environments).
_SKIP_DIRS = {
    "__pycache__",
    ".hypothesis",
    ".pytest_cache",
    ".git",
    "node_modules",
    "site-packages",
    "venv",
    ".venv",
    "env",
    ".env",
}

# Top-level module names / distribution names that are banned.
_BANNED_MODULES = {"tensorflow", "keras", "tf_keras"}
_BANNED_DIST_RE = re.compile(r"^(tensorflow|keras|tf[-_.]?keras)([-_.].*)?$")

_REQ_LINE_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*(.*)$")


# ── helpers ────────────────────────────────────────────────────────────────
def _version_tuple(text: str) -> tuple[int, ...]:
    """'2.15.0+cpu' -> (2, 15, 0). Stops at the first non-numeric component."""
    parts: list[int] = []
    for piece in text.split("+", 1)[0].split("."):
        m = re.match(r"\d+", piece)
        if not m:
            break
        parts.append(int(m.group()))
    return tuple(parts)


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_requirements() -> dict[str, str]:
    """Map normalized distribution name -> version specifier ('' when unpinned)."""
    reqs: dict[str, str] = {}
    for raw in REQUIREMENTS_TXT.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        m = _REQ_LINE_RE.match(line)
        assert m, f"unparseable requirements line: {raw!r}"
        spec = m.group(3).split(";", 1)[0].replace(" ", "")
        reqs[_normalize(m.group(1))] = spec
    return reqs


def _is_skipped(path: Path) -> bool:
    rel_parts = path.relative_to(BACKEND_DIR).parts[:-1]
    return any(part in _SKIP_DIRS for part in rel_parts)


def _venv_roots() -> list[Path]:
    """Directories under backend/ that are virtual environments (have pyvenv.cfg)."""
    return [cfg.parent for cfg in BACKEND_DIR.rglob("pyvenv.cfg")]


def _backend_py_files() -> list[Path]:
    venvs = _venv_roots()
    files = []
    for path in BACKEND_DIR.rglob("*.py"):
        if _is_skipped(path):
            continue
        if any(v in path.parents for v in venvs):
            continue
        files.append(path)
    return sorted(files)


def _banned_imports(source: str, filename: str) -> list[str]:
    """Return 'line N: <module>' for every import naming tensorflow/keras."""
    tree = ast.parse(source, filename=filename)
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _BANNED_MODULES:
                    hits.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and node.module.split(".")[0] in _BANNED_MODULES:
                hits.append(f"line {node.lineno}: from {node.module} import ...")
    return hits


# ── torch runtime ──────────────────────────────────────────────────────────
def test_torch_imports_and_is_at_least_2_6():
    import torch

    version = _version_tuple(torch.__version__)
    assert version[:2] >= (2, 6), f"torch {torch.__version__} is older than 2.6"


def test_version_tuple_parsing():
    assert _version_tuple("2.15.0+cpu") == (2, 15, 0)
    assert _version_tuple("2.6.0") == (2, 6, 0)
    assert _version_tuple("2.10.0.dev20250101") == (2, 10, 0)
    assert _version_tuple("2.6")[:2] >= (2, 6)
    assert _version_tuple("2.5.1")[:2] < (2, 6)


# ── requirements.txt ───────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("package", "required_spec"),
    [
        ("torch", ">=2.6"),
        ("numpy", ">=1.26.4"),
        ("h5py", ">=3.11.0"),
        ("scipy", None),
        ("sqlalchemy", None),
        ("python-multipart", None),
    ],
)
def test_requirements_lists_package(package: str, required_spec: str | None):
    reqs = _parse_requirements()
    assert package in reqs, f"{package} missing from requirements.txt"
    if required_spec is not None:
        specs = reqs[package].split(",")
        assert required_spec in specs, (
            f"{package} should be pinned '{required_spec}', found '{reqs[package]}'"
        )


def test_requirements_has_no_tensorflow_or_keras():
    banned = [name for name in _parse_requirements() if _BANNED_DIST_RE.match(name)]
    assert not banned, f"banned packages in requirements.txt: {banned}"


@pytest.mark.parametrize(
    "name",
    ["tensorflow", "tensorflow-cpu", "tensorflow-gpu", "tensorflow-intel", "keras", "tf-keras", "tf_keras"],
)
def test_banned_distribution_pattern_matches_variants(name: str):
    assert _BANNED_DIST_RE.match(_normalize(name))


@pytest.mark.parametrize("name", ["torch", "kerastuner-free-lib", "tensorboard", "scipy"])
def test_banned_distribution_pattern_ignores_others(name: str):
    # 'kerastuner-free-lib' starts with 'keras' but not as its own name segment.
    assert not _BANNED_DIST_RE.match(_normalize(name))


# ── source scan ────────────────────────────────────────────────────────────
def test_no_backend_module_imports_tensorflow_or_keras():
    files = _backend_py_files()
    assert files, "no Python files found under backend/"
    offenders: dict[str, list[str]] = {}
    for path in files:
        source = path.read_text(encoding="utf-8-sig")
        hits = _banned_imports(source, str(path))
        if hits:
            offenders[str(path.relative_to(BACKEND_DIR))] = hits
    assert not offenders, f"tensorflow/keras imports found: {offenders}"


def test_import_scanner_detects_banned_imports():
    src = (
        "import os\n"
        "import tensorflow as tf\n"
        "from keras.layers import Dense\n"
        "import tensorflow.compat.v1\n"
        "from tf_keras import models\n"
        "s = 'import tensorflow'\n"  # string literal: not an import
        "from . import keras_utils\n"  # relative import: not the keras package
    )
    hits = _banned_imports(src, "<sample>")
    assert len(hits) == 4
    assert all("line 6" not in h and "line 7" not in h for h in hits)
