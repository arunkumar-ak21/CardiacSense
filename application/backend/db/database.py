"""
SQLite engine and session management for the longitudinal companion database.

Design notes
------------
* Uses the synchronous ``sqlite3`` driver through SQLAlchemy 2.x. This avoids the
  ``aiosqlite`` / greenlet bridge entirely, which is the usual source of crashes
  and "database is locked" errors on Windows.
* Every new DBAPI connection enables ``journal_mode=WAL`` (readers never block the
  single writer), ``synchronous=NORMAL``, ``foreign_keys=ON`` and a 30 s busy
  timeout so concurrent FastAPI requests wait instead of failing.
* ``check_same_thread=False`` lets FastAPI's threadpool (sync ``def`` routes using
  ``Depends(get_db)``) use pooled connections created on other threads.
* ``async def`` routes should call :func:`run_in_session`, which runs the DB work
  in a worker thread so the event loop is never blocked.

The database file lives at ``backend/companion.db`` regardless of the current
working directory. Set ``COMPANION_DB_PATH`` to override the location.
"""
from __future__ import annotations

import asyncio
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

BACKEND_DIR = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("COMPANION_DB_PATH", str(BACKEND_DIR / "companion.db"))).resolve()
DATABASE_URL = f"sqlite:///{DB_PATH.as_posix()}"

T = TypeVar("T")


class Base(DeclarativeBase):
    """Declarative base shared by all ORM models."""


engine: Engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30},
    pool_pre_ping=True,
)


@event.listens_for(engine, "connect")
def _configure_sqlite_connection(dbapi_connection: Any, _connection_record: Any) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=30000")
    finally:
        cursor.close()


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    class_=Session,
)


def get_db() -> Iterator[Session]:
    """FastAPI dependency: ``def route(db: Session = Depends(get_db))``."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for scripts and background jobs (commit or rollback)."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


async def run_in_session(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """
    Run ``fn(session, *args, **kwargs)`` in a worker thread inside a transaction.

    Use from ``async def`` routes so blocking SQLite I/O never stalls the event loop::

        patients = await run_in_session(lambda db: [p.to_dict() for p in db.query(Patient)])
    """

    def _work() -> T:
        with session_scope() as db:
            return fn(db, *args, **kwargs)

    return await asyncio.to_thread(_work)


def init_db(seed: bool = False) -> dict[str, Any]:
    """
    Create all tables (idempotent). When ``seed`` is True, also insert the demo
    patients if the ``patients`` table is empty.

    Returns a small status dict with the database path and seeding result.
    """
    from . import models  # noqa: F401  (registers tables on Base.metadata)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)

    status: dict[str, Any] = {"db_path": str(DB_PATH), "seed": None}
    if seed:
        from .seed_and_samples import seed_initial_database

        db = SessionLocal()
        try:
            status["seed"] = seed_initial_database(db)
        finally:
            db.close()
    return status
