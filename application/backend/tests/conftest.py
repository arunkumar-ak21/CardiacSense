"""
Shared pytest configuration for the CODE-15% multimodal engine tests.

Order matters here. Everything at module level runs before pytest collects any
test module, so the environment is fixed before ``db`` (or anything that imports
it) is loaded:

1. ``backend/`` and ``backend/tests/`` go on ``sys.path``.
2. ``COMPANION_DB_PATH`` points at a per-session temp file, so tests never touch
   ``backend/companion.db``.
3. ``OPENAI_API_KEY`` and ``GEMINI_API_KEY`` are blanked. They are set to ``""``
   rather than deleted so that ``load_dotenv()`` in ``main.py`` (which never
   overrides an existing variable) cannot bring real keys back from ``.env``.
4. The Hypothesis profile ``code15`` (100 examples, no deadline) is loaded.

This module never imports ``torch``, ``main``, ``db`` or ``genai`` at import
time. Fixtures that need them import lazily.
"""
from __future__ import annotations

import atexit
import importlib
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest
from hypothesis import HealthCheck, settings

TESTS_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TESTS_DIR.parent

for _p in (str(BACKEND_DIR), str(TESTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Per-session database file (must be set before any ``db`` import) ──────
_SESSION_TMP = Path(tempfile.mkdtemp(prefix="code15_tests_"))
SESSION_DB_PATH = _SESSION_TMP / "companion_test.db"
os.environ["COMPANION_DB_PATH"] = str(SESSION_DB_PATH)
atexit.register(shutil.rmtree, _SESSION_TMP, ignore_errors=True)

# ── No LLM provider is configured during tests ─────────────────────────────
for _key in ("OPENAI_API_KEY", "GEMINI_API_KEY"):
    os.environ[_key] = ""

# ── Hypothesis profile ─────────────────────────────────────────────────────
settings.register_profile(
    "code15",
    max_examples=100,
    deadline=None,
    # Some strategies synthesize full 12-lead ECGs with NumPy; generation time is
    # expected and should not fail the health check.
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("code15")


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════
@pytest.fixture
def torch_mod():
    """The ``torch`` module; the test is skipped when torch is not installed."""
    return pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _instant_fallback_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Set ``genai.narrative.FALLBACK_TOKEN_DELAY = 0`` so fallback streams finish
    instantly. ``genai.narrative`` is pure Python (no torch, no ``main``); if it
    cannot be imported yet, the fixture does nothing.
    """
    try:
        narrative = importlib.import_module("genai.narrative")
    except Exception:  # module missing or broken: nothing to patch
        return
    monkeypatch.setattr(narrative, "FALLBACK_TOKEN_DELAY", 0, raising=False)


# Names in ``db.database`` that other modules may have imported by value
# (``from db.database import SessionLocal``) and that a reload replaces.
_DB_REBIND_NAMES = ("DB_PATH", "DATABASE_URL", "engine", "SessionLocal")


def _point_database_at(db_path: Path) -> Any:
    """
    Point ``db.database`` at ``db_path`` (taken from ``COMPANION_DB_PATH``).

    ``importlib.reload`` re-runs the module in the same namespace, so functions
    such as ``get_db``, ``session_scope`` and ``init_db`` (including ones captured
    by FastAPI ``Depends``) pick up the new engine automatically. Two details are
    handled explicitly:

    * The declarative ``Base`` is kept, because the ORM models in ``db.models``
      are registered on it and ``init_db`` creates tables from its metadata.
    * Loaded backend modules holding the old ``engine``/``SessionLocal``/
      ``DB_PATH``/``DATABASE_URL`` objects by value are rebound to the new ones.
    """
    os.environ["COMPANION_DB_PATH"] = str(db_path)
    if "db.database" not in sys.modules:
        return importlib.import_module("db.database")

    database = sys.modules["db.database"]
    old = {name: getattr(database, name, None) for name in _DB_REBIND_NAMES}
    old_base = database.Base
    try:
        old["engine"].dispose()
    except Exception:
        pass

    database = importlib.reload(database)
    database.Base = old_base

    backend_prefix = str(BACKEND_DIR).lower()
    for mod in list(sys.modules.values()):
        if mod is None or mod is database:
            continue
        mod_file = str(getattr(mod, "__file__", "") or "").lower()
        if not mod_file.startswith(backend_prefix) or mod_file.startswith(str(TESTS_DIR).lower()):
            continue
        namespace = getattr(mod, "__dict__", {})
        for name in _DB_REBIND_NAMES:
            if name in namespace and old[name] is not None and namespace[name] is old[name]:
                setattr(mod, name, getattr(database, name))
    return database


@contextmanager
def _client_on(db_path: Path) -> Iterator[Any]:
    from fastapi.testclient import TestClient

    database = _point_database_at(db_path)
    import main  # noqa: PLC0415  (lazy: main imports torch first)

    try:
        with TestClient(main.app) as client:
            yield client
    finally:
        try:
            database.engine.dispose()
        except Exception:
            pass


@pytest.fixture
def fresh_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """
    A ``TestClient(main.app)`` (lifespan started) backed by a brand-new SQLite
    file. After the test, ``db.database`` points back at the session database.
    """
    monkeypatch.setenv("COMPANION_DB_PATH", str(tmp_path / "companion_fresh.db"))
    try:
        with _client_on(tmp_path / "companion_fresh.db") as client:
            yield client
    finally:
        _point_database_at(SESSION_DB_PATH)
