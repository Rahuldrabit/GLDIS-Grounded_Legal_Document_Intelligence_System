"""Database session factory and helpers."""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import Session, sessionmaker

from core.config import get_settings
from db.models import Base

_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        connect_args = {}
        engine_kwargs = {}
        if settings.database_url.startswith("sqlite"):
            connect_args = {"check_same_thread": False}
            if settings.database_url.endswith(":memory:") or settings.database_url == "sqlite:///:memory:":
                engine_kwargs["poolclass"] = StaticPool
        _engine = create_engine(
            settings.database_url,
            connect_args=connect_args,
            **engine_kwargs,
            echo=settings.debug,
        )
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(), autocommit=False, autoflush=False
        )
    return _SessionLocal


def create_tables() -> None:
    """Create all tables (idempotent)."""
    Base.metadata.create_all(bind=get_engine())


def get_db() -> Session:
    """FastAPI dependency that yields a DB session and closes it after."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
