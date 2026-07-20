from __future__ import annotations

import uuid

from core.config import get_settings
from db import models
from core.schemas import DocumentStatus
from ingestion import orchestrator


class _Engine:
    value = "mock-ocr"


class _Page:
    page = 1
    text = "Sample extracted text"
    engine = _Engine()


class _OCRResult:
    total_text = "Sample extracted text"
    pages = [_Page()]
    vlm_logs = []


class _Chunk:
    def __init__(self, document_id: str):
        self.chunk_id = str(uuid.uuid4())
        self.document_id = document_id
        self.text = "Sample extracted text"
        self.page = 1
        self.section = "section"
        self.token_count = 3

    def model_dump(self):
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "page": self.page,
            "section": self.section,
            "token_count": self.token_count,
        }


def test_run_pipeline_uses_db_blob_when_upload_file_missing(db_session, monkeypatch, tmp_path):
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    get_settings.cache_clear()
    get_settings().ensure_dirs()

    doc_id = str(uuid.uuid4())
    doc = models.Document(
        id=doc_id,
        filename=f"{doc_id}.txt",
        original_filename="source.txt",
        mime_type="text/plain",
        file_size=19,
        file_content=b"Sample extracted text",
        status=DocumentStatus.UPLOADED.value,
    )
    db_session.add(doc)
    db_session.commit()

    monkeypatch.setattr(orchestrator, "is_vlm_blocks_available", lambda: False)
    monkeypatch.setattr(orchestrator._ocr, "extract", lambda **kwargs: _OCRResult())
    monkeypatch.setattr(orchestrator._layout_parser, "parse", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator._chunker, "chunk_text", lambda text, document_id: [_Chunk(document_id)])
    monkeypatch.setattr(orchestrator._rule_extractor, "extract_all", lambda chunks: [])
    monkeypatch.setattr(orchestrator._ner_extractor, "extract_all", lambda chunks: [])
    monkeypatch.setattr(orchestrator, "persist_ocr_blocks", lambda *args, **kwargs: {})
    monkeypatch.setattr(orchestrator, "link_chunks_to_blocks", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator, "index_blocks_in_graph", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator._retriever, "add_chunks", lambda chunks: None)

    summary = orchestrator.run_pipeline(doc_id, db_session)
    assert summary["chunks_created"] == 1

    db_session.refresh(doc)
    assert doc.status == DocumentStatus.READY.value

