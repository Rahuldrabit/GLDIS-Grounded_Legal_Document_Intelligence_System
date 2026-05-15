"""
VLM Parser — Structured JSON extraction from document images (Steps 7–9).

This module is the standalone VLM inference class used by both the ingestion
orchestrator and the vlm/ API sub-application. It wraps the shared
extract_with_vlm() function and provides a class-based interface for reuse.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import get_settings
from llm.client import chat_completion, resolve_llm_config
from ocr.vlm_extractor import VLM_EXTRACTION_PROMPT, _encode_image, _parse_vlm_response

logger = logging.getLogger(__name__)


class VLMParser:
    """
    Vision-Language Model parser for legal document images.

    Connects to Qwen2.5-VL (via LM Studio) or GPT-4.1 Vision (via OpenAI)
    and returns a structured JSON dict with extracted text, entities, and layout.

    Usage:
        parser = VLMParser()
        result = parser.parse_document_image("path/to/page.png")
        # result["text"], result["sections"], result["entities"], ...
    """

    def __init__(self):
        settings = get_settings()
        self.settings = settings
        self._client = None
        self._model = settings.vlm_model
        self._init_client()

    def _init_client(self):
        provider_override = self.settings.llm_provider if isinstance(getattr(self.settings, "llm_provider", ""), str) else ""
        openai_api_key = self.settings.openai_api_key if isinstance(getattr(self.settings, "openai_api_key", ""), str) else ""
        if not self.settings.vlm_enabled and not provider_override.strip() and not openai_api_key:
            logger.warning("No VLM endpoint configured.")
            return

        config = resolve_llm_config(mode="vision")
        if not config.base_url:
            logger.warning("No VLM endpoint configured.")
            return

        self._model = config.model
        self._client = object()
        logger.info(f"VLMParser using provider={config.provider} model={self._model}")

    def parse_document_image(self, image_path: str) -> Dict[str, Any]:
        """
        Send an image to the VLM and return a structured dict.

        Returns:
            Dict with keys: document_type, text, sections, entities,
            tables, handwritten_notes, signatures, key_obligations, confidence.
            On failure: {"error": str, "text": ""}.
        """
        if self._client is None:
            return {"error": "No VLM client available.", "text": ""}

        logger.info(f"VLMParser: parsing {Path(image_path).name} with {self._model}")

        try:
            b64 = _encode_image(image_path)

            config = resolve_llm_config(mode="vision")

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
                model=self._model,
                max_tokens=config.max_tokens,
                temperature=0.1,
                response_format={"type": "json_object"} if config.supports_json_mode else None,
            )

            raw = response.choices[0].message.content or ""
            parsed = _parse_vlm_response(raw)

            if parsed is None:
                logger.warning(f"VLMParser: JSON parse failed for {image_path}")
                return {"error": "JSON parse failed", "text": raw}

            return parsed

        except Exception as exc:
            logger.error(f"VLMParser failed for {image_path}: {exc}")
            return {"error": str(exc), "text": ""}

    def parse_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Parse multiple page images in sequence."""
        return [self.parse_document_image(p) for p in image_paths]
