"""
VLM Extractor — Primary Document Understanding Engine (Step 7–9)

Routes document images through a Vision-Language Model (Qwen2.5-VL or
GPT-4.1 Vision) to extract:
  1. Full text with layout preservation
  2. Headings and sections
  3. Dates, entities, obligations
  4. Tables (as Markdown)
  5. Handwritten notes and signatures

Returns structured PageOCRResult objects and raw VLM log entries.
Falls back gracefully when VLM is unavailable or confidence is low.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from core.config import get_settings
from llm.client import chat_completion, resolve_llm_config
from core.schemas import OCREngine, PageOCRResult, TextBlock

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Prompt (Step 8 spec)
# ──────────────────────────────────────────────────────────────────────────────

VLM_EXTRACTION_PROMPT = """Analyze this legal-style document page carefully.

Extract ALL of the following:
1. Full text (preserve line breaks and formatting)
2. Headings and section titles
3. Sections with their content
4. Dates (all formats)
5. Legal entities (person names, organizations, jurisdictions)
6. Tables (format as Markdown tables)
7. Handwritten notes or annotations (if any)
8. Signature blocks (if any)
9. Key obligations (sentences containing "shall", "must", "agrees to", etc.)

Return a JSON object ONLY — no explanation, no markdown wrapper. Format exactly:
{
  "document_type": "string (e.g. contract, notice, court_filing, lease, memo)",
  "text": "full extracted text of the entire page",
  "sections": [
    {"title": "string", "content": "string"}
  ],
  "entities": {
    "person_names": ["string"],
    "organizations": ["string"],
    "dates": ["string"],
    "jurisdictions": ["string"]
  },
  "tables": ["markdown table string"],
  "handwritten_notes": ["string"],
  "signatures": ["string"],
  "key_obligations": ["string"],
  "confidence": 0.95
}"""


# Block-guided extraction prompt (for MinerU-cropped regions)
VLM_BLOCK_EXTRACTION_PROMPT = """Extract text and metadata from this document region carefully.

Return a JSON object:
{
  "text": "extracted text from this block",
  "block_type_detected": "text|heading|table|signature|annotation|unknown",
  "confidence": 0.95
}"""


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _encode_image(image_path: str) -> str:
    """Base64-encode an image file for API transmission."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _crop_image_region(image_path: str, bbox: List[float]) -> Optional[str]:
    """
    Crop an image to a bounding box and return the base64-encoded result.
    bbox: [x1, y1, x2, y2] in pixel coordinates.
    Returns path to the temp cropped image, or None on failure.
    """
    try:
        from PIL import Image
        import tempfile
        
        img = Image.open(image_path)
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cropped = img.crop((x1, y1, x2, y2))
        
        # Save to temp file and return encoded
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            cropped.save(tmp.name)
            return tmp.name
    except Exception as exc:
        logger.warning(f"Image cropping failed: {exc}")
        return None


def _parse_vlm_response(content: str) -> Optional[Dict[str, Any]]:
    """
    Parse VLM JSON response, handling common model output variations:
    - Bare JSON object
    - JSON wrapped in ```json ... ``` blocks
    - JSON with trailing commentary
    """
    content = content.strip()

    # Strip markdown code fence if present
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    content = content.strip()

    # Try direct parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try extracting the first JSON object via regex
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _vlm_result_to_page(
    page_num: int,
    parsed: Dict[str, Any],
    raw_content: str,
) -> PageOCRResult:
    """Convert a parsed VLM JSON response to a PageOCRResult."""
    text = parsed.get("text", raw_content)
    confidence = float(parsed.get("confidence", 0.9))

    # Build rich text: prepend section structure so downstream chunking
    # can detect heading boundaries
    enriched_parts = [text]
    sections = parsed.get("sections", [])
    if sections:
        enriched_parts = []
        for sec in sections:
            title = sec.get("title", "")
            body = sec.get("content", "")
            if title:
                enriched_parts.append(f"\n\n{title}\n{body}")
            else:
                enriched_parts.append(body)

    # Append obligations as a distinct block for extraction
    obligations = parsed.get("key_obligations", [])
    if obligations:
        enriched_parts.append("\n\nKey Obligations:\n" + "\n".join(f"- {o}" for o in obligations))

    enriched_text = "\n".join(enriched_parts).strip() or text

    return PageOCRResult(
        page=page_num,
        text=enriched_text,
        confidence=confidence,
        engine=OCREngine.VLM,
        blocks=[TextBlock(text=enriched_text, bbox=[0, 0, 0, 0], confidence=confidence)],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main extractor
# ──────────────────────────────────────────────────────────────────────────────

def extract_with_vlm(
    image_paths: List[str],
    use_json_mode: bool = True,
) -> Tuple[List[PageOCRResult], List[dict]]:
    """
    Primary VLM extraction path (Steps 7–9).

    Sends each page image to a vision-language model (Qwen2.5-VL via LM Studio
    or GPT-4.1 Vision via OpenAI) and receives structured JSON with full text
    + metadata extraction.

    Args:
        image_paths: List of pre-rendered page image paths.
        use_json_mode: Request JSON response format from the API.

    Returns:
        - List[PageOCRResult]: One per page.
        - List[dict]: Raw VLM query logs for the data flywheel.
    """
    settings = get_settings()
    results: List[PageOCRResult] = []
    logs: List[dict] = []

    if not image_paths:
        return results, logs

    provider_override = settings.llm_provider if isinstance(getattr(settings, "llm_provider", ""), str) else ""
    openai_api_key = settings.openai_api_key if isinstance(getattr(settings, "openai_api_key", ""), str) else ""
    if not settings.vlm_enabled and not provider_override.strip() and not openai_api_key:
        logger.warning(
            "VLM extraction skipped: no VLM endpoint configured. "
            "Set VLM_ENABLED=true and VLM_API_BASE, or set OPENAI_API_KEY."
        )
        return results, logs

    config = resolve_llm_config(mode="vision")

    if not config.base_url:
        logger.warning(
            "VLM extraction skipped: no VLM endpoint configured. "
            "Set VLM_ENABLED=true and VLM_API_BASE, or set OPENAI_API_KEY."
        )
        return results, logs

    logger.info(f"VLM client → provider={config.provider} base_url={config.base_url} model={config.model}")

    # ── Process each page ────────────────────────────────────────────────────
    for page_num, img_path in enumerate(image_paths, start=1):
        logger.info(f"VLM processing page {page_num}/{len(image_paths)}: {img_path}")
        try:
            b64 = _encode_image(img_path)

            req_kwargs: dict = {}
            if use_json_mode and config.supports_json_mode:
                req_kwargs["response_format"] = {"type": "json_object"}

            response = chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": VLM_EXTRACTION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                mode="vision",
                model=config.model,
                max_tokens=config.max_tokens,
                temperature=0.1,
                response_format=req_kwargs.get("response_format"),
            )

            raw_content = response.choices[0].message.content or ""
            parsed = _parse_vlm_response(raw_content)

            if parsed:
                page_result = _vlm_result_to_page(page_num, parsed, raw_content)
            else:
                # VLM returned non-JSON — use raw text but mark low confidence
                logger.warning(f"VLM page {page_num}: JSON parse failed, using raw text.")
                page_result = PageOCRResult(
                    page=page_num,
                    text=raw_content,
                    confidence=0.5,
                    engine=OCREngine.VLM,
                    blocks=[TextBlock(text=raw_content, bbox=[0, 0, 0, 0], confidence=0.5)],
                )

            results.append(page_result)
            logs.append({
                "image_path": img_path,
                "prompt": VLM_EXTRACTION_PROMPT,
                "response": raw_content,
                "model_used": config.model,
                "parsed_ok": parsed is not None,
                "confidence": page_result.confidence,
            })

        except Exception as exc:
            logger.error(f"VLM extraction failed on page {page_num}: {exc}")
            results.append(PageOCRResult(
                page=page_num, text="", confidence=0.0,
                engine=OCREngine.VLM, blocks=[],
            ))

    return results, logs


def is_vlm_available() -> bool:
    """Quick check: is the VLM endpoint reachable and configured?"""
    settings = get_settings()
    if not settings.vlm_enabled:
        return False
    try:
        import httpx
        r = httpx.get(f"{settings.vlm_api_base}/models", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False
