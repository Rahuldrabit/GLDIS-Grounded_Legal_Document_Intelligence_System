"""
Shared Pydantic schemas used across the entire pipeline.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────

class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class FeedbackType(str, Enum):
    MISSING_FACT = "missing_fact"
    INCORRECT_FACT = "incorrect_fact"
    POOR_STRUCTURE = "poor_structure"
    HALLUCINATION = "hallucination"
    IRRELEVANT_EVIDENCE = "irrelevant_evidence"
    OTHER = "other"


class DraftType(str, Enum):
    CASE_SUMMARY_MEMO = "case_summary_memo"  # Default: combined Case Fact Summary + Internal Memo


class OCREngine(str, Enum):
    PYMUPDF = "pymupdf"
    TESSERACT = "tesseract"
    PADDLEOCR = "paddleocr"
    VLM = "vlm"


class OCRBlockType(str, Enum):
    """MinerU-style block classification for document layout understanding."""
    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    FIGURE = "figure"
    FOOTER = "footer"
    HEADER = "header"
    STAMP = "stamp"
    SIGNATURE = "signature"
    ANNOTATION = "annotation"
    UNKNOWN = "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# OCR / Text Extraction
# ──────────────────────────────────────────────────────────────────────────────

class TextBlock(BaseModel):
    text: str
    bbox: List[float] = Field(default_factory=list, description="[x1, y1, x2, y2] in 300DPI pixel coordinates")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class OCRBlock(BaseModel):
    """Persistent block-level OCR result with coordinate grounding (Stage 1)."""
    block_id: str
    document_id: str
    page: int
    block_type: OCRBlockType = OCRBlockType.TEXT
    text: str
    bbox: List[float] = Field(description="[x1, y1, x2, y2] in 300DPI pixel coordinates")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    ocr_engine: OCREngine = OCREngine.VLM
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PageOCRResult(BaseModel):
    page: int
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    engine: OCREngine
    blocks: List[TextBlock] = Field(default_factory=list)


class DocumentOCRResult(BaseModel):
    document_id: str
    pages: List[PageOCRResult]
    total_text: str = ""
    vlm_logs: List[Dict[str, Any]] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        if not self.total_text:
            self.total_text = "\n\n".join(p.text for p in self.pages)


# ──────────────────────────────────────────────────────────────────────────────
# Chunks
# ──────────────────────────────────────────────────────────────────────────────

class ChunkMetadata(BaseModel):
    document_id: str
    chunk_id: str
    page: Optional[int] = None
    section: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    page: Optional[int] = None
    section: Optional[str] = None
    token_count: int = 0
    block_ids: List[str] = Field(default_factory=list, description="OCRBlock IDs that contribute to this chunk (coordinate grounding)")
    bbox_union: List[float] = Field(default_factory=list, description="Union bbox of contributing blocks [x1, y1, x2, y2]")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Structured Entity Extraction
# ──────────────────────────────────────────────────────────────────────────────

class ExtractedField(BaseModel):
    field: str
    value: Any
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_chunk_id: Optional[str] = None


class StructuredExtraction(BaseModel):
    document_id: str
    case_number: Optional[str] = None
    plaintiff: Optional[str] = None
    defendant: Optional[str] = None
    dates: List[str] = Field(default_factory=list)
    obligations: List[str] = Field(default_factory=list)
    monetary_values: List[str] = Field(default_factory=list)
    jurisdictions: List[str] = Field(default_factory=list)
    addresses: List[str] = Field(default_factory=list)
    parties: List[str] = Field(default_factory=list)
    deadlines: List[str] = Field(default_factory=list)
    raw_fields: List[ExtractedField] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────────────────────────────────────

class EvidenceChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    page: Optional[int] = None
    section: Optional[str] = None
    score: float = 0.0
    retrieval_method: str = "hybrid"
    block_ids: List[str] = Field(default_factory=list, description="OCRBlock IDs for coordinate grounding")
    bbox_union: List[float] = Field(default_factory=list, description="Union bbox for highlighting")


# ──────────────────────────────────────────────────────────────────────────────
# Drafts
# ──────────────────────────────────────────────────────────────────────────────

class DraftRequest(BaseModel):
    document_id: str
    query: Optional[str] = "Generate a case fact summary and internal memo draft."
    top_k: int = Field(default=5, ge=1, le=20)
    draft_type: str = DraftType.CASE_SUMMARY_MEMO


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    page: Optional[int] = None
    excerpt: str = ""


class DraftResponse(BaseModel):
    draft_id: str
    document_id: str
    generated_text: str
    citations: List[Citation] = Field(default_factory=list)
    evidence_chunks: List[EvidenceChunk] = Field(default_factory=list)
    grounding_score: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ──────────────────────────────────────────────────────────────────────────────
# Feedback
# ──────────────────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    draft_id: str
    original_draft: str
    edited_draft: str
    feedback_type: FeedbackType = FeedbackType.OTHER
    reviewer_comment: Optional[str] = None
    reviewer_id: Optional[str] = "anonymous"


class FeedbackRecord(BaseModel):
    feedback_id: str
    draft_id: str
    original_draft: str
    edited_draft: str
    edit_diff: str
    feedback_type: FeedbackType
    reviewer_comment: Optional[str] = None
    reviewer_id: str
    retrieved_evidence: List[EvidenceChunk] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ──────────────────────────────────────────────────────────────────────────────
# API Response wrappers
# ──────────────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    message: str


class ProcessResponse(BaseModel):
    document_id: str
    status: DocumentStatus
    chunks_created: int
    entities_extracted: int
    message: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class DocumentListResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    page_count: int
    upload_time: datetime


class DocumentDetailResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    page_count: int
    file_size: int
    chunks_count: int
    upload_time: datetime
    processed_time: Optional[datetime] = None
    error_message: Optional[str] = None
    structured_fields: List[Dict[str, Any]] = Field(default_factory=list)


class SearchResultItem(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    page: Optional[int] = None
    section: Optional[str] = None
    score: float
    retrieval_method: str


class SearchResultResponse(BaseModel):
    query: str
    document_id: Optional[str] = None
    results_count: int
    results: List[SearchResultItem]


class EvidenceChunkResponse(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    page: Optional[int] = None
    section: Optional[str] = None
    token_count: int
    chunk_index: int
