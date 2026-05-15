"""
Reasoner client adapter for Stage 3 (DeepSeek-R1 reasoning model).

Calls the configured `reasoning_api_base`/`reasoning_model` endpoint
with an OpenAI-compatible chat/completions call and returns the full
text response from the model.
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any

import httpx

from core.config import get_settings

logger = logging.getLogger(__name__)


def call_reasoner(messages: List[Dict[str, Any]], temperature: float = 0.0, max_tokens: int = 1500) -> str:
    settings = get_settings()
    base = getattr(settings, "reasoning_api_base", None)
    model = getattr(settings, "reasoning_model", None)
    api_key = getattr(settings, "llm_api_key", None)

    if not base or not model:
        raise RuntimeError("Reasoner endpoint not configured")

    url = base.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, headers=headers, json=data)
            resp.raise_for_status()
            return resp.text
    except Exception as exc:
        logger.warning(f"Reasoner API call failed: {exc}")
        raise
