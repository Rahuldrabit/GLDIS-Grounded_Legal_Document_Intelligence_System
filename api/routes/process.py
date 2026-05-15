"""
API Routes: Document Processing Pipeline
Executes Preprocessing, OCR, Chunking, Extraction, and Indexing.
(In production, this would be a background Celery task).
"""
from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from core.config import get_settings
from core.schemas import DocumentStatus, ProcessResponse
from db.models import Document, Chunk, StructuredField
from db.session import get_db

from preprocessing.pipeline import preprocess_pdf_pages, preprocess_image
from preprocessing.layout_parser import LayoutParser
from preprocessing.chunker import SemanticChunker
from ocr.hybrid_ocr import HybridOCR
from extraction import RuleBasedExtractor, NERExtractor
from retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

router = APIRouter()

def process_document_task(document_id: str, db: Session):
    """
    The main processing pipeline. Runs synchronously here for simplicity,
    but should be async/background in production.
    """
    settings = get_settings()
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        logger.error(f"Document {document_id} not found for processing.")
        return
        
    doc.status = DocumentStatus.PROCESSING.value
    db.commit()
    
    file_path = Path(settings.upload_dir) / doc.filename
    if not file_path.exists():
        doc.status = DocumentStatus.FAILED.value
        doc.error_message = "File not found on disk."
        db.commit()
        return

    try:
        # 1. Preprocessing (Skipping rendering to images for simple test, assuming HybridOCR handles PDF)
        # 2. OCR
        logger.info(f"Starting OCR for {document_id}")
        ocr_engine = HybridOCR()
        ocr_result = ocr_engine.extract(str(file_path))
        doc.page_count = len(ocr_result.pages)
        
        # 3. Layout Parsing
        logger.info(f"Parsing layout for {document_id}")
        layout_parser = LayoutParser()
        layout = layout_parser.parse(ocr_result.total_text, document_id)
        
        # 4. Chunking
        logger.info(f"Chunking {document_id}")
        chunker = SemanticChunker()
        chunks_schemas = chunker.chunk_layout(layout, document_id)
        
        # Fallback if layout chunker produced nothing (e.g., no headings found)
        if not chunks_schemas:
            chunks_schemas = chunker.chunk_text(ocr_result.total_text, document_id)
            
        # DB chunks
        db_chunks = []
        for c in chunks_schemas:
            db_chunk = Chunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                text=c.text,
                page=c.page,
                section=c.section,
                token_count=c.token_count,
            )
            db.add(db_chunk)
            db_chunks.append(db_chunk)
            
        # 5. Extraction
        logger.info(f"Extracting entities for {document_id}")
        rule_ext = RuleBasedExtractor()
        ner_ext = NERExtractor()
        
        fields = rule_ext.extract_all(chunks_schemas)
        fields.extend(ner_ext.extract_all(chunks_schemas))
        
        for f in fields:
            db_field = StructuredField(
                document_id=document_id,
                field=f.field,
                value=f.value,
                confidence=f.confidence,
                source_chunk_id=f.source_chunk_id
            )
            db.add(db_field)
            
        # 6. Indexing (Vector + BM25)
        logger.info(f"Indexing {document_id}")
        retriever = HybridRetriever()
        retriever.add_chunks(chunks_schemas)
        
        # 7. Finalize
        doc.status = DocumentStatus.READY.value
        doc.processed_time = datetime.utcnow()
        db.commit()
        
        logger.info(f"Document {document_id} processed successfully. "
                    f"Created {len(chunks_schemas)} chunks and {len(fields)} fields.")
                    
    except Exception as exc:
        logger.error(f"Processing failed for {document_id}: {exc}")
        db.rollback()
        # Reload doc
        doc = db.query(Document).filter(Document.id == document_id).first()
        doc.status = DocumentStatus.FAILED.value
        doc.error_message = str(exc)
        db.commit()


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
        
    if doc.status == DocumentStatus.READY.value:
        return ProcessResponse(
            document_id=document_id,
            status=DocumentStatus.READY,
            chunks_created=0,
            entities_extracted=0,
            message="Document already processed."
        )

    # In a real app we'd dispatch to Celery. Here we use FastAPI BackgroundTasks
    background_tasks.add_task(process_document_task, document_id, db)
    
    return ProcessResponse(
        document_id=document_id,
        status=DocumentStatus.PROCESSING,
        chunks_created=0,
        entities_extracted=0,
        message="Processing started in background."
    )
