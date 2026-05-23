"""
API Routes: Document Processing Pipeline
Executes Preprocessing, OCR, Chunking, Extraction, and Indexing.
(In production, this would be a background Celery task).
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy import update
from sqlalchemy.orm import Session

from core.schemas import DocumentStatus, ProcessResponse
from db.models import Document
from db.session import get_db

from ingestion.orchestrator import run_pipeline
from db.session import get_session_factory

logger = logging.getLogger(__name__)

router = APIRouter()

def _run_pipeline_task(document_id: str, previous_status: str | None) -> None:
    """Background task wrapper that uses a fresh DB session."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        run_pipeline(document_id, db, previous_status=previous_status)
    except Exception as exc:
        logger.error(f"Processing failed for {document_id}: {exc}")
    finally:
        db.close()


@router.post("/{document_id}", response_model=ProcessResponse)
async def process_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
) -> ProcessResponse:
    """
    Trigger processing pipeline for an uploaded document.
    """
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    status_before_processing = str(doc.status)

    if doc.status == DocumentStatus.READY.value:
        return ProcessResponse(
            document_id=document_id,
            status=DocumentStatus.READY,
            chunks_created=0,
            entities_extracted=0,
            message="Document already processed.",
        )

    # Atomic state transition: prevent double-enqueue under concurrency.
    result = db.execute(
        update(Document)
        .where(
            Document.id == document_id,
            Document.status.notin_([
                DocumentStatus.PROCESSING.value,
                DocumentStatus.READY.value,
            ]),
        )
        .values(status=DocumentStatus.PROCESSING.value, error_message=None)
    )
    db.commit()

    if result.rowcount == 0:
        # Someone else is processing or it became ready.
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc and doc.status == DocumentStatus.PROCESSING.value:
            raise HTTPException(status_code=409, detail="Document is already being processed")
        return ProcessResponse(
            document_id=document_id,
            status=DocumentStatus.READY,
            chunks_created=0,
            entities_extracted=0,
            message="Document already processed.",
        )

    # In a real app we'd dispatch to Celery. Here we use FastAPI BackgroundTasks.
    background_tasks.add_task(_run_pipeline_task, document_id, status_before_processing)
    
    return ProcessResponse(
        document_id=document_id,
        status=DocumentStatus.PROCESSING,
        chunks_created=0,
        entities_extracted=0,
        message="Processing started in background."
    )
