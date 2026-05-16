"""SQLAlchemy ORM models for all GLDIS database tables."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


# ──────────────────────────────────────────────────────────────────────────────
class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=_uuid)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    mime_type = Column(String, default="application/octet-stream")
    file_size = Column(Integer, default=0)
    status = Column(String, default="uploaded")  # DocumentStatus values
    page_count = Column(Integer, default=0)
    upload_time = Column(DateTime, default=datetime.utcnow)
    processed_time = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    ocr_blocks = relationship("OCRBlock", back_populates="document", cascade="all, delete-orphan")
    structured_fields = relationship("StructuredField", back_populates="document", cascade="all, delete-orphan")
    drafts = relationship("Draft", back_populates="document", cascade="all, delete-orphan")


# ──────────────────────────────────────────────────────────────────────────────
class OCRBlock(Base):
    """Stage 1 — Persistent coordinate-grounded block extraction (MinerU/VLM-guided)."""
    __tablename__ = "ocr_blocks"

    block_id = Column(String, primary_key=True, default=_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    page = Column(Integer, nullable=False)
    block_type = Column(String, default="text")  # OCRBlockType value
    text = Column(Text, nullable=False)
    bbox = Column(JSON, nullable=False)  # [x1, y1, x2, y2] in 300DPI pixels
    confidence = Column(Float, default=1.0)
    ocr_engine = Column(String, default="vlm")  # OCREngine value
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="ocr_blocks")


# ──────────────────────────────────────────────────────────────────────────────
class Chunk(Base):
    __tablename__ = "chunks"

    chunk_id = Column(String, primary_key=True, default=_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    text = Column(Text, nullable=False)
    page = Column(Integer, nullable=True)
    section = Column(String, nullable=True)
    token_count = Column(Integer, default=0)
    start_char = Column(Integer, nullable=True)
    end_char = Column(Integer, nullable=True)
    chunk_index = Column(Integer, default=0)
    block_ids = Column(JSON, default=list)  # List of OCRBlock IDs for coordinate grounding
    bbox_union = Column(JSON, default=list)  # [x1, y1, x2, y2] union of contributing blocks
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="chunks")


# ──────────────────────────────────────────────────────────────────────────────
class StructuredField(Base):
    __tablename__ = "structured_fields"

    id = Column(String, primary_key=True, default=_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    field = Column(String, nullable=False)
    value = Column(Text, nullable=False)
    confidence = Column(Float, default=1.0)
    source_chunk_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="structured_fields")


# ──────────────────────────────────────────────────────────────────────────────
class Draft(Base):
    __tablename__ = "drafts"

    draft_id = Column(String, primary_key=True, default=_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    query = Column(Text, nullable=True)
    generated_text = Column(Text, nullable=False)
    evidence_used = Column(JSON, default=list)   # list of chunk_id strings
    citations = Column(JSON, default=list)        # list of citation dicts
    grounding_score = Column(Float, default=0.0)
    model_used = Column(String, default="gpt-4.1")
    draft_type = Column(String, default="case_summary_memo", nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="drafts")
    edits = relationship("Edit", back_populates="draft", cascade="all, delete-orphan")


# ──────────────────────────────────────────────────────────────────────────────
class Edit(Base):
    __tablename__ = "edits"

    edit_id = Column(String, primary_key=True, default=_uuid)
    draft_id = Column(String, ForeignKey("drafts.draft_id"), nullable=False)
    original_text = Column(Text, nullable=False)
    edited_text = Column(Text, nullable=False)
    edit_diff = Column(Text, nullable=True)
    feedback_type = Column(String, default="other")
    reviewer_comment = Column(Text, nullable=True)
    reviewer_id = Column(String, default="anonymous")
    retrieved_evidence = Column(JSON, default=list)  # evidence chunks used
    created_at = Column(DateTime, default=datetime.utcnow)

    draft = relationship("Draft", back_populates="edits")


# ──────────────────────────────────────────────────────────────────────────────
class FewShotExample(Base):
    """Stores high-quality (evidence → corrected draft) pairs for prompt learning."""
    __tablename__ = "few_shot_examples"

    id = Column(String, primary_key=True, default=_uuid)
    evidence_context = Column(Text, nullable=False)
    original_draft = Column(Text, nullable=False)
    corrected_draft = Column(Text, nullable=False)
    feedback_type = Column(String, default="other")
    quality_score = Column(Float, default=1.0)
    use_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


# ──────────────────────────────────────────────────────────────────────────────
class VLMQueryLog(Base):
    """Stores VLM queries and responses for the multimodal data flywheel."""
    __tablename__ = "vlm_query_logs"

    id = Column(String, primary_key=True, default=_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    image_path = Column(String, nullable=False)
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    model_used = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document")
