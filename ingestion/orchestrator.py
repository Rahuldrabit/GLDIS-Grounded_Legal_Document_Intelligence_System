"""
Document Ingestion Orchestrator — End-to-End Pipeline (Steps 3–18)

Pipeline stages:
  1. Preprocessing  — PDF→image, deskew, denoise, CLAHE
  2. VLM or OCR    — VLM-first for images; OCR chain fallback
  3. Layout Parsing — Rule-based heading/clause detection (OCR path only)
  4. Chunking       — Section-aware semantic chunking
  5. Extraction     — Rule-based regex + spaCy NER
  6. Indexing       — FAISS dense + BM25 sparse

Status transitions: uploaded → processing → ready | failed
"""
from __future__ import annotations

import logging
import tempfile
import os
import json
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from core.config import get_settings
from core.schemas import DocumentStatus, StructuredExtraction
from db import models
from extraction.ner_extractor import NERExtractor
from extraction.rule_based import RuleBasedExtractor
from ocr.hybrid_ocr import HybridOCR
from preprocessing.chunker import SemanticChunker
from preprocessing.layout_parser import LayoutParser
from preprocessing.pipeline import preprocess_image, pdf_to_images
from retrieval.hybrid_retriever import HybridRetriever
from ocr.block_extraction_adapter import extract_vlm_with_blocks, is_vlm_blocks_available
from ingestion.ocr_block_handler import (
    persist_ocr_blocks,
    link_chunks_to_blocks,
    index_blocks_in_graph,
)

logger = logging.getLogger(__name__)

# Module-level singletons (loaded once per process)
_ocr = HybridOCR()
_layout_parser = LayoutParser()
_chunker = SemanticChunker()
_rule_extractor = RuleBasedExtractor()
_ner_extractor = NERExtractor()
_retriever = HybridRetriever()


def _set_status(db: Session, doc: models.Document, status: DocumentStatus, error: Optional[str] = None):
    doc.status = status.value
    if error:
        doc.error_message = error
    elif status != DocumentStatus.FAILED:
        # Clear previous errors once we move out of failed state
        doc.error_message = None
    db.commit()
    logger.info(f"Document {doc.id} → {status.value}")


def run_pipeline(document_id: str, db: Session, *, previous_status: Optional[str] = None) -> dict:
    """Execute the full ingestion pipeline for a document.

    Returns summary dict:
      {document_id, page_count, chunks_created, entities_extracted, ocr_engines_used}
    """

    doc = db.query(models.Document).filter(models.Document.id == document_id).first()
    if not doc:
        raise ValueError(f"Document {document_id} not found.")

    settings = get_settings()
    file_path_obj = Path(settings.upload_dir) / doc.filename
    temp_source_file: Optional[str] = None
    if not file_path_obj.exists():
        if doc.file_content:
            suffix = Path(doc.filename).suffix or ".tmp"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(doc.file_content)
                temp_source_file = tmp_file.name
            file_path_obj = Path(temp_source_file)
            logger.info(f"[{document_id}] Source file restored from DB blob into temp storage.")
        else:
            missing_path = str(file_path_obj)
            _set_status(db, doc, DocumentStatus.FAILED, f"File not found: {missing_path}")
            raise FileNotFoundError(missing_path)
    file_path = str(file_path_obj)

    status_before_processing = previous_status or doc.status
    _set_status(db, doc, DocumentStatus.PROCESSING)

    ext = Path(file_path).suffix.lower()
    is_pdf = ext == ".pdf"
    is_image = ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp")

    # Retry cleanup: if a previous run FAILED, delete partial DB/index state so the retry is idempotent.
    if status_before_processing == DocumentStatus.FAILED.value:
        try:
            db.query(models.Chunk).filter(models.Chunk.document_id == document_id).delete(synchronize_session=False)
            db.query(models.StructuredField).filter(models.StructuredField.document_id == document_id).delete(synchronize_session=False)
            db.query(models.OCRBlock).filter(models.OCRBlock.document_id == document_id).delete(synchronize_session=False)
            db.query(models.VLMQueryLog).filter(models.VLMQueryLog.document_id == document_id).delete(synchronize_session=False)
            db.commit()
        except Exception as exc:
            logger.warning(f"[{document_id}] Cleanup of partial DB state failed: {exc}")
            db.rollback()

        try:
            _retriever.delete_document(document_id)
        except Exception as exc:
            logger.warning(f"[{document_id}] Cleanup of index state failed: {exc}")

    artifact_dir = Path(settings.upload_dir).parent / "artifacts" / document_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    image_paths: list[str] = []
    tmp_dir: Optional[str] = None

    try:
        # ── Stage 1: Preprocessing ─────────────────────────────────────────
        logger.info(f"[{document_id}] Stage 1: Preprocessing")
        try:
            if is_pdf:
                raw_images = pdf_to_images(file_path, dpi=300)
                if raw_images:
                    tmp_dir = tempfile.mkdtemp()
                    for i, img in enumerate(raw_images):
                        try:
                            proc = preprocess_image(img)
                        except Exception:
                            proc = img
                        img_path = os.path.join(tmp_dir, f"page_{i + 1}.png")
                        proc.save(img_path)
                        image_paths.append(img_path)
            elif is_image:
                image_paths = [file_path]
        except Exception as exc:
            logger.warning(f"[{document_id}] Preprocessing warning: {exc}. Continuing with raw file.")

        # ── Stage 2: VLM-First → OCR Fallback ─────────────────────────────
        logger.info(f"[{document_id}] Stage 2: Document understanding (VLM-first)")
        try:
            if is_vlm_blocks_available() and image_paths:
                logger.info(f"[{document_id}] Using block-guided VLM extraction")
                ocr_result, vlm_logs = extract_vlm_with_blocks(file_path, image_paths)
            else:
                ocr_result = _ocr.extract(
                    file_path=file_path,
                    image_paths=image_paths if image_paths else None,
                )
                vlm_logs = getattr(ocr_result, "vlm_logs", []) or []

            full_text = ocr_result.total_text
            page_count = len(ocr_result.pages)
            doc.page_count = page_count
            db.commit()

            if vlm_logs:
                for log_entry in vlm_logs:
                    db.add(models.VLMQueryLog(
                        document_id=document_id,
                        image_path=log_entry.get("image_path", ""),
                        prompt=log_entry.get("prompt", ""),
                        response=log_entry.get("response", ""),
                        model_used=log_entry.get("model_used", ""),
                    ))
                db.commit()

            try:
                block_lookup = persist_ocr_blocks(db, document_id, ocr_result.pages)
            except Exception as _e:
                logger.warning(f"[{document_id}] Persisting OCRBlocks failed: {_e}")
                block_lookup = {}

            try:
                with open(artifact_dir / "ocr.json", "w", encoding="utf-8") as f:
                    f.write(ocr_result.model_dump_json(indent=2))
            except Exception as e:
                logger.warning(f"[{document_id}] Failed to save ocr.json: {e}")

        except Exception as exc:
            _set_status(db, doc, DocumentStatus.FAILED, f"OCR/VLM failed: {exc}")
            raise

        if not full_text.strip():
            _set_status(db, doc, DocumentStatus.FAILED, "No text extracted from document.")
            raise ValueError("Empty OCR/VLM output.")

        # ── Stage 3: Layout Parsing ───────────────────────────────────────
        logger.info(f"[{document_id}] Stage 3: Layout parsing")
        try:
            page_marked_text = ""
            for page in ocr_result.pages:
                page_marked_text += f"\n--- PAGE {page.page} ---\n{page.text}\n"
            layout = _layout_parser.parse(page_marked_text, document_id=document_id)
            if layout:
                try:
                    with open(artifact_dir / "layout.json", "w", encoding="utf-8") as f:
                        if hasattr(layout, "model_dump_json"):
                            f.write(layout.model_dump_json(indent=2))
                        elif hasattr(layout, "model_dump"):
                            json.dump(layout.model_dump(), f, indent=2)
                        else:
                            json.dump(layout.__dict__, f, indent=2, default=str)
                except Exception as e:
                    logger.warning(f"[{document_id}] Failed to save layout.json: {e}")
        except Exception as exc:
            logger.warning(f"[{document_id}] Layout parsing failed: {exc}. Using flat text.")
            layout = None

        # ── Stage 4: Chunking ─────────────────────────────────────────────
        logger.info(f"[{document_id}] Stage 4: Semantic chunking")
        try:
            if layout and getattr(layout, "blocks", None):
                chunks = _chunker.chunk_layout(layout, document_id=document_id)
            else:
                chunks = _chunker.chunk_text(full_text, document_id=document_id)
        except Exception as exc:
            _set_status(db, doc, DocumentStatus.FAILED, f"Chunking failed: {exc}")
            raise

        for i, chunk in enumerate(chunks):
            db.add(models.Chunk(
                chunk_id=chunk.chunk_id,
                document_id=document_id,
                text=chunk.text,
                page=chunk.page,
                section=chunk.section,
                token_count=chunk.token_count,
                chunk_index=i,
            ))
        db.commit()
        logger.info(f"[{document_id}] {len(chunks)} chunks persisted.")

        try:
            with open(artifact_dir / "chunks.json", "w", encoding="utf-8") as f:
                json.dump([c.model_dump() for c in chunks], f, indent=2)
        except Exception as e:
            logger.warning(f"[{document_id}] Failed to save chunks.json: {e}")

        try:
            link_chunks_to_blocks(db, document_id, chunks, ocr_result.pages, block_lookup)
        except Exception as e:
            logger.warning(f"[{document_id}] Chunk→block linkage failed: {e}. Continuing.")

        # ── Stage 5: Entity Extraction ────────────────────────────────────
        logger.info(f"[{document_id}] Stage 5: Entity extraction")
        entities_count = 0
        try:
            rule_fields = _rule_extractor.extract_all(chunks)
            ner_fields = _ner_extractor.extract_all(chunks)
            all_fields = rule_fields + ner_fields

            seen = set()
            for field in all_fields:
                key = (field.field, str(field.value).lower()[:50])
                if key in seen:
                    continue
                seen.add(key)
                db.add(models.StructuredField(
                    document_id=document_id,
                    field=field.field,
                    value=str(field.value),
                    confidence=field.confidence,
                    source_chunk_id=field.source_chunk_id,
                ))
                entities_count += 1
            db.commit()
        except Exception as exc:
            logger.warning(f"[{document_id}] Entity extraction warning: {exc}")

        try:
            index_blocks_in_graph(document_id, chunks, filename=doc.original_filename)
        except Exception as e:
            logger.warning(f"[{document_id}] Graph indexing failed: {e}. Continuing.")

        # ── Stage 6: Indexing (FAISS + BM25) ──────────────────────────────
        logger.info(f"[{document_id}] Stage 6: Indexing (FAISS + BM25)")
        try:
            _retriever.add_chunks(chunks)
        except Exception as exc:
            logger.warning(f"[{document_id}] Indexing warning: {exc}. Continuing.")

        from datetime import datetime
        doc.processed_time = datetime.utcnow()
        _set_status(db, doc, DocumentStatus.READY)

        summary = {
            "document_id": document_id,
            "page_count": page_count,
            "chunks_created": len(chunks),
            "entities_extracted": entities_count,
            "ocr_engines_used": list({p.engine.value for p in ocr_result.pages}),
        }
        logger.info(f"[{document_id}] Pipeline complete: {summary}")
        return summary

    except Exception:
        try:
            db.refresh(doc)
        except Exception:
            pass
        if getattr(doc, "status", None) != DocumentStatus.FAILED.value:
            _set_status(db, doc, DocumentStatus.FAILED, "Pipeline failed")
        raise

    finally:
        if tmp_dir:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if temp_source_file:
            Path(temp_source_file).unlink(missing_ok=True)


def get_structured_extraction(document_id: str, db: Session) -> StructuredExtraction:
    """Assemble StructuredExtraction from persisted DB fields."""
    fields = (
        db.query(models.StructuredField)
        .filter(models.StructuredField.document_id == document_id)
        .all()
    )
    extraction = StructuredExtraction(document_id=document_id)
    for f in fields:
        val = f.value
        fname = f.field
        if fname == "case_number" and not extraction.case_number:
            extraction.case_number = val
        elif fname == "date":
            extraction.dates.append(val)
        elif fname == "monetary_value":
            extraction.monetary_values.append(val)
        elif fname == "jurisdiction":
            extraction.jurisdictions.append(val)
        elif fname == "address":
            extraction.addresses.append(val)
        elif fname == "deadline":
            extraction.deadlines.append(val)
        elif fname in ("person", "organisation"):
            extraction.parties.append(val)
        elif fname == "obligation":
            extraction.obligations.append(val)
    return extraction
