"""
Lightweight adapter for Mem0 feedback memory. If `mem0` package
is available, use it; otherwise fall back to a file-based JSONL store.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, List

from core.config import get_settings

logger = logging.getLogger(__name__)

MEM0_PATH = Path(get_settings().feedback_store_path) / "mem0.jsonl"


class Mem0Store:
    def __init__(self):
        self.path = MEM0_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._enabled = False
        try:
            import mem0  # type: ignore

            self._client = mem0.Client()
            self._enabled = True
        except Exception:
            logger.info("mem0 not installed; using file-based feedback store")

    def save(self, record: Dict[str, Any]) -> None:
        if self._enabled:
            try:
                self._client.save(record)
                return
            except Exception as exc:
                logger.warning(f"mem0 client save failed: {exc}")

        # File fallback
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def all(self) -> List[Dict[str, Any]]:
        if self._enabled:
            try:
                return list(self._client.all())
            except Exception as exc:
                logger.warning(f"mem0 client all() failed: {exc}")
                return []

        if not self.path.exists():
            return []
        out = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        return out
