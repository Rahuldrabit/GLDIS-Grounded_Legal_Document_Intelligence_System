"""Database session factory and helpers."""
from __future__ import annotations

import logging

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.pool import StaticPool
from sqlalchemy.orm import Session, sessionmaker

from core.config import get_settings
from db.models import Base

_engine = None
_SessionLocal = None
logger = logging.getLogger(__name__)


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
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    _ensure_file_content_column(engine)


def _ensure_file_content_column(engine) -> None:
    """Backfill schema for deployments created before documents.file_content existed."""
    inspector = inspect(engine)
    if "documents" not in inspector.get_table_names():
        return

    existing_cols = {col["name"] for col in inspector.get_columns("documents")}
    if "file_content" in existing_cols:
        return

    if engine.dialect.name == "postgresql":
        ddl = "ALTER TABLE documents ADD COLUMN file_content BYTEA"
    else:
        ddl = "ALTER TABLE documents ADD COLUMN file_content BLOB"

    try:
        with engine.begin() as conn:
            conn.execute(text(ddl))
        logger.info("Added missing documents.file_content column.")
    except Exception as exc:
        logger.warning(f"Could not add documents.file_content column automatically: {exc}")


def get_db() -> Session:
    """FastAPI dependency that yields a DB session and closes it after."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
