"""
Shared pytest fixtures for GLDIS test suite.
"""
from __future__ import annotations

import os
import uuid
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# ── Point at an in-memory SQLite DB for tests ─────────────────────────────────
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
os.environ.setdefault("VLM_ENABLED", "false")
os.environ.setdefault("UPLOAD_DIR", tempfile.mkdtemp())
os.environ.setdefault("FAISS_INDEX_PATH", tempfile.mkdtemp())
os.environ.setdefault("BM25_INDEX_PATH", tempfile.mkdtemp())
os.environ.setdefault("FEEDBACK_STORE_PATH", tempfile.mkdtemp())

from db.models import Base
from core.schemas import Chunk, EvidenceChunk


@pytest.fixture(scope="session")
def db_engine():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine) -> Generator[Session, None, None]:
    SessionLocal = sessionmaker(bind=db_engine, autocommit=False, autoflush=False)
    session = SessionLocal()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def sample_text() -> str:
    return """
    COMMERCIAL LEASE AGREEMENT

    This Commercial Lease Agreement ("Agreement") is entered into as of January 15, 2024,
    by and between Acme Properties LLC ("Landlord") and Beta Corp ("Tenant").

    Case No: CV-2024-00123

    ARTICLE 1 — PREMISES
    Landlord hereby leases to Tenant the property located at
    500 Main Street, New York, NY 10001.

    ARTICLE 2 — TERM
    The lease term shall commence on February 1, 2024 and expire on January 31, 2026.

    ARTICLE 3 — RENT
    Tenant shall pay monthly rent of $12,500.00 (USD) due on the first of each month.
    Late payments are subject to a penalty of $500.00 per day.

    ARTICLE 4 — OBLIGATIONS
    Tenant must maintain the premises in good condition.
    Tenant shall not sublet without written consent of Landlord.
    Landlord agrees to provide heating and cooling systems.

    Signed: ____________________
    Date: January 15, 2024
    """


@pytest.fixture
def sample_chunks(sample_text) -> list[Chunk]:
    doc_id = str(uuid.uuid4())
    from preprocessing.chunker import SemanticChunker
    chunker = SemanticChunker(target_tokens=200, overlap_tokens=30)
    return chunker.chunk_text(sample_text, document_id=doc_id)


@pytest.fixture
def sample_evidence() -> list[EvidenceChunk]:
    doc_id = str(uuid.uuid4())
    return [
        EvidenceChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=doc_id,
            text="The lease term shall commence on February 1, 2024 and expire on January 31, 2026.",
            page=1,
            section="ARTICLE 2 — TERM",
            score=0.92,
        ),
        EvidenceChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=doc_id,
            text="Tenant shall pay monthly rent of $12,500.00 (USD) due on the first of each month.",
            page=1,
            section="ARTICLE 3 — RENT",
            score=0.88,
        ),
        EvidenceChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=doc_id,
            text="Tenant must maintain the premises in good condition.",
            page=1,
            section="ARTICLE 4 — OBLIGATIONS",
            score=0.75,
        ),
    ]


@pytest.fixture
def tmp_pdf(tmp_path) -> str:
    """Create a minimal plain-text 'document' for pipeline tests."""
    f = tmp_path / "test_doc.txt"
    f.write_text(
        "NOTICE OF ARBITRATION\n\nCase No: ARB-2024-999\n\n"
        "Claimant: Alpha Ltd\nRespondent: Beta Inc\n\n"
        "The arbitration hearing shall commence on March 15, 2024.\n"
        "The parties must submit evidence by February 28, 2024.\n"
    )
    return str(f)
