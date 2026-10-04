"""
db/session.py
-------------
SQLAlchemy engine and session factory wired to the configured DATABASE_URL.

Usage (FastAPI dependency injection)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    from app.db.session import get_db

    @router.get("/example")
    def example(db: Session = Depends(get_db)):
        ...

The session is committed on success and rolled back + closed on any exception,
ensuring connections are always returned to the pool.
"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

engine = create_engine(
    settings.database_url,
    # Pool configuration suitable for a single-process FastAPI service.
    pool_pre_ping=True,       # detect stale connections before use
    pool_size=5,
    max_overflow=10,
    echo=settings.debug,      # SQL logging in debug mode only
)


# ---------------------------------------------------------------------------
# Declarative base (shared by all ORM models)
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models in this project."""


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


def get_db() -> Generator[Session, None, None]:
    """
    Yield a database session for the duration of a single request.

    The session is closed in the ``finally`` block so the underlying
    connection is always returned to the pool, even when an exception is
    raised inside a view function.
    """
    db: Session = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Convenience helper (used by the health endpoint)
# ---------------------------------------------------------------------------


def check_db_connection() -> bool:
    """
    Return ``True`` if a lightweight ``SELECT 1`` query succeeds.

    Raises no exceptions – all errors are caught and ``False`` is returned
    so callers can handle the result without a try/except.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
