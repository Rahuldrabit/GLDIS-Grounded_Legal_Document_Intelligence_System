"""
Stage 6 — Rule-Based Entity Extraction
Extracts dates, IDs, monetary values, phone numbers, and other
structured legal fields using compiled regex patterns.
"""
from __future__ import annotations

import re
import logging
from typing import List
from core.schemas import ExtractedField

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Compiled patterns
# ──────────────────────────────────────────────────────────────────────────────

_PATTERNS: dict[str, re.Pattern] = {
    "date": re.compile(
        r"\b(?:"
        r"\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}"           # 12/01/2024
        r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}"  # January 1, 2024
        r"|\d{4}[\/\-]\d{2}[\/\-]\d{2}"                     # 2024-01-12
        r")\b",
        re.IGNORECASE,
    ),
    "case_number": re.compile(
        r"\b(?:Case\s*(?:No\.?|Number|#)?\s*:?\s*)"
        r"([A-Z0-9\-\/]{4,20})\b",
        re.IGNORECASE,
    ),
    "monetary_value": re.compile(
        r"\b(?:USD|EUR|GBP|INR|\$|€|£|₹)?\s*"
        r"\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*"
        r"(?:USD|EUR|GBP|INR|dollars?|euros?|pounds?)?\b",
        re.IGNORECASE,
    ),
    "phone_number": re.compile(
        r"\b(?:\+\d{1,3}[\s\-]?)?"
        r"(?:\(?\d{3}\)?[\s\-]?)?"
        r"\d{3}[\s\-]?\d{4}\b"
    ),
    "email": re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    ),
    "jurisdiction": re.compile(
        r"\b(?:State\s+of|District\s+of|County\s+of|Province\s+of|Court\s+of)\s+"
        r"([A-Z][a-zA-Z\s]{2,30})\b"
    ),
    "address": re.compile(
        r"\b\d{1,5}\s+[A-Za-z0-9\s,\.]{5,60}"
        r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way)\b",
        re.IGNORECASE,
    ),
    "deadline": re.compile(
        r"\b(?:due|by|no\s+later\s+than|deadline|within\s+\d+\s+days?|"
        r"on\s+or\s+before)\s+"
        r"(?:\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})",
        re.IGNORECASE,
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Extractor
# ──────────────────────────────────────────────────────────────────────────────

class RuleBasedExtractor:
    """
    Extracts structured fields from text using deterministic regex patterns.
    Returns a flat list of ExtractedField objects.
    """

    def extract(self, text: str, chunk_id: str | None = None) -> List[ExtractedField]:
        fields: List[ExtractedField] = []

        for field_type, pattern in _PATTERNS.items():
            try:
                matches = pattern.findall(text)
                for match in matches:
                    value = match if isinstance(match, str) else " ".join(match).strip()
                    value = value.strip()
                    if not value or len(value) < 2:
                        continue
                    fields.append(ExtractedField(
                        field=field_type,
                        value=value,
                        confidence=0.85,   # Rule-based = high but not perfect
                        source_chunk_id=chunk_id,
                    ))
            except Exception as exc:
                logger.warning(f"Pattern '{field_type}' failed: {exc}")

        logger.debug(f"Rule extraction: {len(fields)} fields from chunk {chunk_id}")
        return fields

    def extract_all(
        self, chunks: list, field_prefix: str = ""
    ) -> List[ExtractedField]:
        """Run extraction over a list of Chunk objects."""
        all_fields: List[ExtractedField] = []
        for chunk in chunks:
            all_fields.extend(self.extract(chunk.text, chunk_id=chunk.chunk_id))
        return all_fields
