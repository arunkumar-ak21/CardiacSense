"""
Longitudinal companion database package.

Importing this package is lightweight (SQLAlchemy only). The demo seeder and ECG
sample generator live in ``db.seed_and_samples`` and are imported on demand so
numpy/scipy never load before torch in ``main.py`` (Windows DLL import-order rule).
"""
from .database import (
    BACKEND_DIR,
    DATABASE_URL,
    DB_PATH,
    Base,
    SessionLocal,
    engine,
    get_db,
    init_db,
    run_in_session,
    session_scope,
)
from .delta_engine import CODE15_LABELS, compute_visit_deltas, severity_tier
from .models import LEAD_NAMES, Patient, Visit, json_dumps, json_loads

__all__ = [
    "BACKEND_DIR",
    "CODE15_LABELS",
    "DATABASE_URL",
    "DB_PATH",
    "LEAD_NAMES",
    "Base",
    "Patient",
    "SessionLocal",
    "Visit",
    "compute_visit_deltas",
    "engine",
    "get_db",
    "init_db",
    "json_dumps",
    "json_loads",
    "run_in_session",
    "session_scope",
    "severity_tier",
]
