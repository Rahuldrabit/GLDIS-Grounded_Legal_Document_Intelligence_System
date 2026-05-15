"""
Verifier module for Drafts.

Uses the configured `verifier_api_base` and `verifier_model` to check a
generated draft against evidence and produce a correction list. The
verifier returns a JSON object with `corrections`: an array of
{"claim": "...", "issue": "...", "suggested_fix": "..."}.
"""
from __future__ import annotations

import json
import logging
from typing import List, Dict, Any

import httpx

from core.config import get_settings
from core.schemas import EvidenceChunk

logger = logging.getLogger(__name__)


VERIFIER_PROMPT = (
    "You are a verification assistant. Given a generated draft and the provided evidence passages, "
    "identify factual claims that are not directly supported by the evidence or that contradict it. "
    "Respond with a JSON object: {\n  \"corrections\": [ { \"claim\": \"...\", \"issue\": \"...\", \"suggested_fix\": \"...\" } ]\n}\n"
)


def _call_verifier_api(payload: Dict[str, Any]) -> str:
    settings = get_settings()
    base = getattr(settings, "verifier_api_base", None)
    model = getattr(settings, "verifier_model", None)
    api_key = getattr(settings, "llm_api_key", None)

    if not base or not model:
        raise RuntimeError("Verifier endpoint not configured")

    url = base.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data = {
        "model": model,
        "messages": payload.get("messages", []),
        "temperature": payload.get("temperature", 0.0),
        "max_tokens": payload.get("max_tokens", 512),
    }

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, headers=headers, json=data)
        resp.raise_for_status()
        return resp.text


def verify_draft(
    draft_text: str,
    evidence_chunks: List[EvidenceChunk],
) -> List[Dict[str, Any]]:
    """
    Verify a draft and return a list of corrections.
    Each correction is a dict with keys: claim, issue, suggested_fix.
    """
    settings = get_settings()
    if not settings.verifier_enabled:
        logger.info("Verifier disabled in config; skipping verification.")
        return []

    # Build evidence block
    evid = []
    for c in evidence_chunks[:20]:
        evid.append({"chunk_id": c.chunk_id, "page": c.page, "text": c.text[:1000]})

    messages = [
        {"role": "system", "content": VERIFIER_PROMPT},
        {"role": "user", "content": "Here is the generated draft:\n\n" + draft_text},
        {"role": "user", "content": "Evidence passages (JSON):\n" + json.dumps(evid)},
        {"role": "user", "content": "Produce ONLY a JSON object with field `corrections` as described."},
    ]

    try:
        raw = _call_verifier_api({"messages": messages, "temperature": 0.0, "max_tokens": 800})
        # Try to extract JSON from response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = raw[start:end]
            parsed = json.loads(json_str)
            return parsed.get("corrections", [])
    except Exception as exc:
        logger.warning(f"Verifier API failed: {exc}")

    return []
