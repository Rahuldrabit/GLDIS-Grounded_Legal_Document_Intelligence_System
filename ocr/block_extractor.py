"""
Block-guided VLM extraction module.

Implements MinerU-guided block extraction where each layout block is
processed separately with InternVL2.5, yielding TextBlock entries with
real pixel coordinates [x1, y1, x2, y2] at 300 DPI.
"""

import base64
import logging
import os
import tempfile
from typing import List, Optional, Tuple, Dict, Any

from PIL import Image

from core.config import get_settings
from core.schemas import (
    PageOCRResult,
    TextBlock,
    OCREngine,
)
from llm.client import chat_completion, resolve_llm_config

logger = logging.getLogger(__name__)

# Block-specific VLM prompt (crop-level extraction)
VLM_BLOCK_EXTRACTION_PROMPT = """You are an expert document analyst. Analyze this document block and extract its text content.

Respond in JSON format:
{
    "text": "<extracted text>",
    "confidence": <0.0 to 1.0>,
    "block_quality": "<high|medium|low>"
}

Extract all text visible in the block. Preserve structure (bullets, numbering, line breaks).
If the block is mostly empty/blank, set text to empty string and confidence to 0.0.
"""


def _encode_image(image_path: str) -> str:
    """Encode image to base64."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        return ""


def _crop_image_region(
    image_path: str,
    bbox: List[float],
) -> Optional[str]:
    """
    Crop image to bbox region [x1, y1, x2, y2] and save to temp file.
    
    Args:
        image_path: Path to full-page image.
        bbox: [x1, y1, x2, y2] in pixel coordinates.
    
    Returns:
        Path to cropped temp image, or None if cropping failed.
    """
    try:
        img = Image.open(image_path)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Clamp to image bounds
        x1 = max(0, min(x1, img.width))
        y1 = max(0, min(y1, img.height))
        x2 = max(0, min(x2, img.width))
        y2 = max(0, min(y2, img.height))
        
        if x1 >= x2 or y1 >= y2:
            logger.warning(f"Invalid bbox after clamping: {[x1, y1, x2, y2]}")
            return None
        
        cropped = img.crop((x1, y1, x2, y2))
        
        # Save to temp file
        fd, temp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cropped.save(temp_path, "PNG")
        
        return temp_path
    except Exception as e:
        logger.error(f"Image cropping failed: {e}")
        return None


def _parse_vlm_response(response_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON response from VLM.
    
    Returns:
        Dict with 'text' and 'confidence' keys, or None if parse fails.
    """
    import json
    
    try:
        # Try to extract JSON from response
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = response_text[start:end]
            return json.loads(json_str)
    except Exception as e:
        logger.debug(f"JSON parse failed: {e}")
    
    return None


class MinerULayoutBlock:
    """Minimal MinerU layout block representation."""
    
    def __init__(
        self,
        block_type: str,  # text, heading, table, figure, header, footer, stamp, signature, annotation, unknown
        bbox: List[float],  # [x1, y1, x2, y2]
        text: str = "",
        confidence: float = 0.9,
    ):
        self.block_type = block_type
        self.bbox = bbox
        self.text = text
        self.confidence = confidence


def extract_blocks_with_vlm(
    image_paths: List[str],
    layout_blocks_per_page: List[List[MinerULayoutBlock]],
) -> Tuple[List[PageOCRResult], List[Dict[str, Any]]]:
    """
    Extract text from MinerU-guided blocks using VLM crops.
    
    For each block, crop the image and call InternVL2.5 with block-specific prompt.
    Returns PageOCRResult with TextBlock entries that have real bbox coordinates.
    
    Args:
        image_paths: List of page image paths (300 DPI).
        layout_blocks_per_page: List where each element is a list of MinerULayoutBlock objects.
    
    Returns:
        - List[PageOCRResult]: One per page with TextBlock entries having real bboxes.
        - List[dict]: Raw VLM logs.
    """
    results: List[PageOCRResult] = []
    logs: List[Dict[str, Any]] = []
    
    settings = get_settings()
    if not settings.vlm_enabled:
        logger.warning("VLM disabled; block extraction unavailable.")
        return results, logs
    
    config = resolve_llm_config(mode="vision")
    if not config.base_url:
        logger.warning("VLM endpoint not configured; block extraction unavailable.")
        return results, logs
    
    # Process each page
    for page_num, img_path in enumerate(image_paths, start=1):
        if not os.path.exists(img_path):
            logger.warning(f"Image not found: {img_path}")
            continue
        
        blocks_for_page = (
            layout_blocks_per_page[page_num - 1]
            if (page_num - 1) < len(layout_blocks_per_page)
            else []
        )
        
        page_text_parts = []
        page_blocks: List[TextBlock] = []
        confidences = []
        
        # Extract each block
        for block in blocks_for_page:
            try:
                # Crop image to block bbox
                crop_path = _crop_image_region(img_path, block.bbox)
                if not crop_path:
                    logger.debug(
                        f"Failed to crop block type={block.block_type} "
                        f"on page {page_num}; skipping"
                    )
                    continue
                
                # Encode cropped image
                b64_crop = _encode_image(crop_path)
                if not b64_crop:
                    continue
                
                # Call VLM
                try:
                    response = chat_completion(
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": VLM_BLOCK_EXTRACTION_PROMPT},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{b64_crop}",
                                            "detail": "low",
                                        },
                                    },
                                ],
                            }
                        ],
                        mode="vision",
                        model=config.model,
                        max_tokens=500,
                        temperature=0.1,
                    )
                    
                    raw_content = response.choices[0].message.content or ""
                    parsed = _parse_vlm_response(raw_content)
                    
                    if parsed:
                        text = parsed.get("text", "").strip()
                        confidence = float(parsed.get("confidence", 0.8))
                    else:
                        text = raw_content.strip()
                        confidence = 0.5
                    
                    # Only add non-empty blocks
                    if text:
                        page_text_parts.append(text)
                        confidences.append(confidence)
                        page_blocks.append(
                            TextBlock(
                                text=text,
                                bbox=block.bbox,  # REAL bbox from MinerU
                                confidence=confidence,
                            )
                        )
                        logs.append(
                            {
                                "page": page_num,
                                "block_type": block.block_type,
                                "bbox": block.bbox,
                                "response": raw_content,
                                "model_used": config.model,
                                "confidence": confidence,
                            }
                        )
                    
                except Exception as exc:
                    logger.warning(
                        f"VLM call failed for block on page {page_num}: {exc}"
                    )
                
                # Clean up temp crop
                if crop_path and os.path.exists(crop_path):
                    try:
                        os.remove(crop_path)
                    except OSError:
                        pass
                        
            except Exception as exc:
                logger.warning(
                    f"Block extraction failed on page {page_num}: {exc}"
                )
        
        # Build page result
        page_text = "\n\n".join(page_text_parts) if page_text_parts else ""
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        
        results.append(
            PageOCRResult(
                page=page_num,
                text=page_text,
                confidence=avg_conf,
                engine=OCREngine.VLM,
                blocks=page_blocks,
            )
        )
        
        logger.debug(
            f"Extracted {len(page_blocks)} blocks from page {page_num}"
        )
    
    return results, logs


def extract_with_vlm_guided_blocks(
    image_paths: List[str],
    layout_blocks_per_page: List[List[MinerULayoutBlock]],
) -> Tuple[List[PageOCRResult], List[Dict[str, Any]]]:
    """
    Alias for extract_blocks_with_vlm for backward compatibility.
    
    Args:
        image_paths: List of page image paths (300 DPI).
        layout_blocks_per_page: List of layout blocks per page.
    
    Returns:
        - List[PageOCRResult]: One per page with TextBlock entries.
        - List[dict]: Raw VLM logs.
    """
    return extract_blocks_with_vlm(image_paths, layout_blocks_per_page)
