"""Routes: Document upload, process, list, delete."""
from __future__ import annotations

import uuid
from pathlib import Path

from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import update

from core.config import get_settings
from core.schemas import DocumentStatus, ProcessResponse, UploadResponse, DocumentDetailResponse, DocumentListResponse
from db import models
from db.session import get_db
from ingestion.orchestrator import run_pipeline

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
):
    """Upload a document (PDF, PNG, JPEG, TIFF). Returns a document_id."""
    settings = get_settings()
    allowed = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".txt"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {allowed}")

    doc_id = str(uuid.uuid4())
    safe_name = f"{doc_id}{ext}"
    dest = Path(settings.upload_dir) / safe_name

    try:
        file_bytes = await file.read()
        with open(dest, "wb") as f:
            f.write(file_bytes)
        file_size = len(file_bytes)
    except Exception as exc:
        raise HTTPException(500, f"File storage failed: {exc}")

    doc = models.Document(
        id=doc_id,
        filename=safe_name,
        original_filename=file.filename,
        mime_type=file.content_type or "application/octet-stream",
        file_size=file_size,
        file_content=file_bytes,
        status=DocumentStatus.UPLOADED.value,
    )
    db.add(doc)
    db.commit()

    return UploadResponse(
        document_id=doc_id,
        filename=file.filename,
        status=DocumentStatus.UPLOADED,
        message="Document uploaded. Call POST /api/documents/{id}/process to begin processing.",
    )


from fastapi import Header
import os
import tempfile
from vlm.parser import VLMParser

@router.post("/vlm-extract")
async def stateless_vlm_extract(
    file: UploadFile = File(...),
    x_llm_provider: str = Header(None, alias="X-LLM-Provider"),
    x_llm_api_key: str = Header(None, alias="X-LLM-Api-Key"),
    x_llm_model: str = Header(None, alias="X-LLM-Model"),
    x_llm_base_url: str = Header(None, alias="X-LLM-Base-Url"),
):
    """Stateless VLM extraction endpoint. Does not save to DB."""
    allowed = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {allowed}")

    try:
        file_bytes = await file.read()
        
        # Write to a temporary file
        fd, temp_path = tempfile.mkstemp(suffix=ext)
        with os.fdopen(fd, 'wb') as f:
            f.write(file_bytes)
            
        parser = VLMParser()
        result = parser.parse_document_image(
            temp_path,
            header_provider=x_llm_provider,
            header_api_key=x_llm_api_key,
            header_model=x_llm_model,
            header_base_url=x_llm_base_url
        )
        
        os.remove(temp_path)
        
        if "error" in result and result["error"] and result["error"] != "No VLM client available.":
            raise HTTPException(500, f"Extraction failed: {result['error']}")
            
        return result
    except Exception as exc:
        raise HTTPException(500, f"Stateless extraction failed: {exc}")


@router.post("/{document_id}/process", response_model=ProcessResponse)
def process_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Trigger the full processing pipeline for an uploaded document."""
    doc = db.query(models.Document).filter(models.Document.id == document_id).first()
    if not doc:
        raise HTTPException(404, f"Document {document_id} not found.")

    status_before_processing = str(doc.status)

    # Atomic state transition: prevent double-enqueue under concurrency.
    if doc.status == DocumentStatus.READY.value:
        return ProcessResponse(
            document_id=document_id,
            status=DocumentStatus.READY,
            chunks_created=0,
            entities_extracted=0,
            message="Document already processed.",
        )

    result = db.execute(
        update(models.Document)
        .where(
            models.Document.id == document_id,
            models.Document.status.notin_([
                DocumentStatus.PROCESSING.value,
                DocumentStatus.READY.value,
            ]),
        )
        .values(status=DocumentStatus.PROCESSING.value, error_message=None)
    )
    db.commit()

    if result.rowcount == 0:
        # Re-check status for correct error reporting
        doc = db.query(models.Document).filter(models.Document.id == document_id).first()
        if not doc:
            raise HTTPException(404, f"Document {document_id} not found.")
        if doc.status == DocumentStatus.PROCESSING.value:
            raise HTTPException(409, "Document is already being processed.")
        if doc.status == DocumentStatus.READY.value:
            return ProcessResponse(
                document_id=document_id,
                status=DocumentStatus.READY,
                chunks_created=0,
                entities_extracted=0,
                message="Document already processed.",
            )
        raise HTTPException(409, f"Document cannot be processed in status: {doc.status}")

    # Run pipeline in background
    def _run(doc_id: str, prev_status: str):
        from db.session import get_session_factory
        SessionLocal = get_session_factory()
        bg_db = SessionLocal()
        try:
            run_pipeline(doc_id, bg_db, previous_status=prev_status)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error(f"Pipeline failed for {doc_id}: {exc}")
        finally:
            bg_db.close()

    background_tasks.add_task(_run, document_id, status_before_processing)

    return ProcessResponse(
        document_id=document_id,
        status=DocumentStatus.PROCESSING,
        chunks_created=0,
        entities_extracted=0,
        message="Processing started in background. Poll GET /api/documents/{id} for status.",
    )


@router.post("/{document_id}/process/sync", response_model=ProcessResponse)
def process_document_sync(document_id: str, db: Session = Depends(get_db)):
    """
    Synchronous processing (blocks until complete).
    Useful for testing and small documents.
    """
    doc = db.query(models.Document).filter(models.Document.id == document_id).first()
    if not doc:
        raise HTTPException(404, f"Document {document_id} not found.")

    try:
        summary = run_pipeline(document_id, db)
    except Exception as exc:
        raise HTTPException(500, f"Processing failed: {exc}")

    return ProcessResponse(
        document_id=document_id,
        status=DocumentStatus.READY,
        chunks_created=summary["chunks_created"],
        entities_extracted=summary["entities_extracted"],
        message=f"Processing complete. {summary['chunks_created']} chunks, {summary['entities_extracted']} entities.",
    )


@router.get("/{document_id}", response_model=DocumentDetailResponse)
def get_document(document_id: str, db: Session = Depends(get_db)):
    """Get document status, metadata, and extracted fields."""
    doc = db.query(models.Document).filter(models.Document.id == document_id).first()
    if not doc:
        raise HTTPException(404, f"Document {document_id} not found.")

    fields = (
        db.query(models.StructuredField)
        .filter(models.StructuredField.document_id == document_id)
        .all()
    )
    chunks_count = (
        db.query(models.Chunk)
        .filter(models.Chunk.document_id == document_id)
        .count()
    )

    return {
        "document_id": doc.id,
        "filename": doc.original_filename,
        "status": doc.status,
        "page_count": doc.page_count,
        "file_size": doc.file_size,
        "chunks_count": chunks_count,
        "upload_time": doc.upload_time,
        "processed_time": doc.processed_time,
        "error_message": doc.error_message,
        "structured_fields": [
            {"field": f.field, "value": f.value, "confidence": f.confidence}
            for f in fields[:50]
        ],
    }


@router.get("", response_model=List[DocumentListResponse])
def list_documents(db: Session = Depends(get_db)):
    """List all documents with basic metadata."""
    docs = db.query(models.Document).order_by(models.Document.upload_time.desc()).all()
    return [
        {
            "document_id": d.id,
            "filename": d.original_filename,
            "status": d.status,
            "page_count": d.page_count,
            "upload_time": d.upload_time,
        }
        for d in docs
    ]


@router.delete("/{document_id}")
def delete_document(document_id: str, db: Session = Depends(get_db)):
    """Delete a document and remove it from all indices."""
    doc = db.query(models.Document).filter(models.Document.id == document_id).first()
    if not doc:
        raise HTTPException(404, f"Document {document_id} not found.")

    # Remove from indices
    from retrieval.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever()
    retriever.delete_document(document_id)

    # Remove file
    settings = get_settings()
    file_path = Path(settings.upload_dir) / doc.filename
    if file_path.exists():
        file_path.unlink()

    # Cascade delete from DB
    db.delete(doc)
    db.commit()

    return {"message": f"Document {document_id} deleted.", "document_id": document_id}
