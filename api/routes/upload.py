"""
API Routes: File Upload
Handles initial document ingestion and metadata creation.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from core.config import get_settings
from core.schemas import DocumentStatus, UploadResponse
from db.models import Document
from db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> UploadResponse:
    """
    Ingest a document (PDF or Image).
    Returns a document ID for further processing.
    """
    settings = get_settings()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing")
        
    ext = Path(file.filename).suffix.lower()
    if ext not in [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"]:
        raise HTTPException(status_code=400, detail="Unsupported file format.")
        
    doc_id = str(uuid4())
    stored_filename = f"{doc_id}{ext}"
    stored_path = upload_dir / stored_filename
    
    try:
        with stored_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        file_size = os.path.getsize(stored_path)
        
        # Create DB record
        doc = Document(
            id=doc_id,
            filename=stored_filename,
            original_filename=file.filename,
            mime_type=file.content_type,
            file_size=file_size,
            status=DocumentStatus.UPLOADED.value
        )
        
        db.add(doc)
        db.commit()
        db.refresh(doc)
        
        logger.info(f"Document uploaded: {doc_id} ({file.filename})")
        
        return UploadResponse(
            document_id=doc_id,
            filename=file.filename,
            status=DocumentStatus.UPLOADED,
            message="File uploaded successfully. Ready for processing."
        )
        
    except Exception as exc:
        logger.error(f"Upload failed: {exc}")
        if stored_path.exists():
            stored_path.unlink()
        raise HTTPException(status_code=500, detail=str(exc))
