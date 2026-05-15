"""
Orchestrator extension for OCRBlock persistence and chunk→block linkage.

Implements Stage 1 completion tasks:
- Persist OCRBlock records to DB
- Link chunks to blocks (block_ids + bbox_union)
- Index in Neo4j graph store
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime

from sqlalchemy.orm import Session

from core.schemas import OCRBlockType, PageOCRResult, TextBlock
from db import models

logger = logging.getLogger(__name__)


def persist_ocr_blocks(
    db: Session,
    document_id: str,
    pages: List[PageOCRResult],
) -> Dict[str, str]:
    """
    Persist OCRBlock records for each block in each page.
    
    Returns a mapping of (page, block_index) → block_id for chunk linkage.
    """
    block_lookup: Dict[tuple, str] = {}  # (page, block_idx) -> block_id
    block_ids_created = []
    
    for page in pages:
        for block_idx, block in enumerate(page.blocks):
            try:
                # Infer block type if not available
                block_type = getattr(block, 'block_type', 'text') or 'text'
                if block_type not in [bt.value for bt in OCRBlockType]:
                    block_type = 'text'
                
                # Create OCRBlock record
                import uuid
                block_id = str(uuid.uuid4())
                
                ocr_block = models.OCRBlock(
                    block_id=block_id,
                    document_id=document_id,
                    page=page.page,
                    block_type=block_type,
                    text=block.text,
                    bbox=block.bbox,  # Store as [x1, y1, x2, y2]
                    confidence=block.confidence,
                    ocr_engine=page.engine.value,
                    created_at=datetime.utcnow(),
                )
                db.add(ocr_block)
                
                # Track for chunk linkage
                block_lookup[(page.page, block_idx)] = block_id
                block_ids_created.append(block_id)
                
            except Exception as e:
                logger.warning(f"Failed to persist OCRBlock on page {page.page}: {e}")
    
    db.commit()
    logger.info(f"Persisted {len(block_ids_created)} OCRBlocks for document {document_id}")
    
    return block_lookup


def link_chunks_to_blocks(
    db: Session,
    document_id: str,
    chunks: List,  # List[core.schemas.Chunk]
    pages: List[PageOCRResult],
    block_lookup: Dict[tuple, str],
) -> None:
    """
    Update persisted Chunk records with:
    - block_ids: list of contributing OCRBlock IDs
    - bbox_union: union of contributing block bboxes
    
    Args:
        db: Database session
        document_id: Document ID
        chunks: List of Chunk objects (from chunker)
        pages: PageOCRResult list with block info
        block_lookup: Mapping (page, block_idx) → block_id
    """
    import json
    
    for chunk in chunks:
        try:
            # Find all blocks that intersect with this chunk's text
            contributing_block_ids: List[str] = []
            contributing_bboxes: List[List[float]] = []
            
            # Simple heuristic: match by page + text overlap
            chunk_page = getattr(chunk, 'page', 1)
            chunk_text_lower = chunk.text.lower()[:100]  # First 100 chars
            
            # Look through blocks on the chunk's page
            for page in pages:
                if page.page != chunk_page:
                    continue
                
                for block_idx, block in enumerate(page.blocks):
                    # Check text overlap
                    block_text_lower = block.text.lower()[:100]
                    if (block_text_lower in chunk_text_lower or
                        chunk_text_lower in block_text_lower or
                        _text_similarity(chunk_text_lower, block_text_lower) > 0.5):
                        
                        block_id = block_lookup.get((page.page, block_idx))
                        if block_id:
                            contributing_block_ids.append(block_id)
                            contributing_bboxes.append(block.bbox)
            
            # Compute union bbox
            bbox_union = _union_bboxes(contributing_bboxes) if contributing_bboxes else [0, 0, 0, 0]
            
            # Update Chunk record
            db_chunk = (
                db.query(models.Chunk)
                .filter(models.Chunk.chunk_id == chunk.chunk_id)
                .first()
            )
            if db_chunk:
                db_chunk.block_ids = json.dumps(contributing_block_ids)
                db_chunk.bbox_union = json.dumps(bbox_union)
            
        except Exception as e:
            logger.warning(f"Failed to link chunk {chunk.chunk_id} to blocks: {e}")
    
    db.commit()
    logger.info(f"Linked chunks to blocks for document {document_id}")


def _text_similarity(text_a: str, text_b: str) -> float:
    """
    Simple Jaccard similarity between two texts (by character).
    """
    if not text_a or not text_b:
        return 0.0
    
    set_a = set(text_a)
    set_b = set(text_b)
    
    if not set_a and not set_b:
        return 1.0
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    return intersection / union if union > 0 else 0.0


def _union_bboxes(bboxes: List[List[float]]) -> List[float]:
    """
    Compute union bbox from a list of bboxes [x1, y1, x2, y2].
    """
    if not bboxes:
        return [0, 0, 0, 0]
    
    x1_vals = [bbox[0] for bbox in bboxes if len(bbox) >= 2]
    y1_vals = [bbox[1] for bbox in bboxes if len(bbox) >= 2]
    x2_vals = [bbox[2] for bbox in bboxes if len(bbox) >= 4]
    y2_vals = [bbox[3] for bbox in bboxes if len(bbox) >= 4]
    
    if not x1_vals or not y1_vals or not x2_vals or not y2_vals:
        return [0, 0, 0, 0]
    
    return [
        min(x1_vals),
        min(y1_vals),
        max(x2_vals),
        max(y2_vals),
    ]


def index_blocks_in_graph(
    document_id: str,
    chunks: List,
) -> None:
    """
    Index chunks and their block references in Neo4j graph store.
    """
    try:
        from retrieval.graph_store import get_graph_store
        
        graph_store = get_graph_store()
        if graph_store:
            graph_store.index_document(document_id)
            graph_store.index_chunks(chunks)
            logger.info(f"Indexed blocks/chunks for {document_id} in Neo4j")
    except Exception as e:
        logger.warning(f"Graph indexing failed: {e}. Continuing without Neo4j.")
