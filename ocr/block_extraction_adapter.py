"""
Block-guided extraction integration for HybridOCR.

Provides a seamless bridge between MinerU layout detection and
VLM-based block extraction, returning results in the standard
DocumentOCRResult format.
"""

import logging
from typing import List, Optional, Tuple

from core.schemas import DocumentOCRResult, PageOCRResult, OCREngine
from ocr.block_extractor import extract_blocks_with_vlm, MinerULayoutBlock
from preprocessing.pipeline import extract_layout_blocks_mineru

logger = logging.getLogger(__name__)


def extract_vlm_with_blocks(
    pdf_path: str,
    image_paths: List[str],
) -> Tuple[DocumentOCRResult, List[dict]]:
    """
    Extract using MinerU-guided block extraction via VLM.
    
    Orchestrates:
    1. MinerU layout detection (PDF → 300DPI images + layout blocks)
    2. VLM crop-guided extraction (InternVL2.5 on each block)
    3. Aggregation into DocumentOCRResult
    
    Args:
        pdf_path: Path to PDF document
        image_paths: Pre-rendered page images (300 DPI) from pdf_to_images()
    
    Returns:
        - DocumentOCRResult with all pages and blocks having real bboxes
        - List[dict]: Raw extraction logs for data flywheel
    """
    import os
    
    # Detect layout blocks
    layout_blocks_per_page: List[List[MinerULayoutBlock]] = []
    try:
        detected = extract_layout_blocks_mineru(pdf_path)
        if detected:
            layout_blocks_per_page = detected
            logger.info(f"MinerU detected {len(detected)} pages with layout blocks")
    except Exception as e:
        logger.warning(f"MinerU layout detection failed: {e}. Proceeding without block guidance.")
    
    # If MinerU failed or is disabled, create dummy full-page blocks for each page
    if not layout_blocks_per_page and image_paths:
        logger.debug("Creating full-page blocks as MinerU fallback")
        from PIL import Image
        for img_path in image_paths:
            try:
                img = Image.open(img_path)
                full_page_block = MinerULayoutBlock(
                    block_type="page",
                    bbox=[0, 0, img.width, img.height],
                    text="",
                    confidence=1.0,
                )
                layout_blocks_per_page.append([full_page_block])
            except Exception as e:
                logger.warning(f"Failed to create full-page block for {img_path}: {e}")
    
    # Extract blocks with VLM
    pages, vlm_logs = extract_blocks_with_vlm(image_paths, layout_blocks_per_page)
    
    # Assemble result
    document_id = os.path.splitext(os.path.basename(pdf_path))[0]
    result = DocumentOCRResult(
        document_id=document_id,
        pages=pages,
        vlm_logs=vlm_logs,
    )
    
    return result, vlm_logs


def is_vlm_blocks_available() -> bool:
    """
    Check if VLM block extraction is available.
    Requires: vlm_enabled + (mineru_available OR fallback to full-page blocks).
    """
    from core.config import get_settings
    
    settings = get_settings()
    return settings.vlm_enabled


def extract_vlm_blocks_fallback(
    image_paths: List[str],
) -> Tuple[List[PageOCRResult], List[dict]]:
    """
    Full-page VLM extraction fallback when MinerU is not available.
    
    Treats each page as a single block to maintain compatibility.
    """
    from ocr.vlm_extractor import extract_with_vlm
    
    pages, vlm_logs = extract_with_vlm(image_paths)
    return pages, vlm_logs
