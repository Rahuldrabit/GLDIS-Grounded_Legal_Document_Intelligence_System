"""
DPO export helper: collect feedback edit pairs and export JSONL for
offline preference optimization / training.

Each line is a JSON object with fields: {"original": ..., "edited": ..., "metadata": {...}}
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from core.config import get_settings
from core.schemas import FeedbackRecord

logger = logging.getLogger(__name__)

EXPORT_PATH = Path(get_settings().feedback_store_path) / "dpo_export.jsonl"


def export_feedback_for_dpo(records: List[FeedbackRecord]) -> Path:
    EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPORT_PATH, "w", encoding="utf-8") as fh:
        for r in records:
            obj: Dict[str, Any] = {
                "original": r.original_draft,
                "edited": r.edited_draft,
                "metadata": {
                    "feedback_id": r.feedback_id,
                    "feedback_type": r.feedback_type.value,
                    "reviewer_id": r.reviewer_id,
                    "created_at": r.created_at.isoformat(),
                },
            }
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
    logger.info(f"Exported {len(records)} feedback records to {EXPORT_PATH}")
    return EXPORT_PATH
