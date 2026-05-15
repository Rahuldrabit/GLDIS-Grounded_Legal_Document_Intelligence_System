"""
Stage 4 — Layout Parsing
Detects headings, sections, clause boundaries, and paragraph hierarchy
from raw OCR text to produce structured document layout.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LayoutBlock:
    block_type: str          # heading | clause | paragraph | table | signature | annotation
    text: str
    level: int = 0           # heading depth (1=H1, 2=H2, etc.)
    page: Optional[int] = None
    section_label: Optional[str] = None
    children: List["LayoutBlock"] = field(default_factory=list)


@dataclass
class DocumentLayout:
    document_id: str
    blocks: List[LayoutBlock] = field(default_factory=list)

    def flat_sections(self) -> List[dict]:
        """Flatten layout into a list of dicts for downstream chunking."""
        result = []
        current_heading = None
        for block in self.blocks:
            if block.block_type == "heading":
                current_heading = block.text.strip()
            result.append({
                "block_type": block.block_type,
                "text": block.text,
                "page": block.page,
                "section": current_heading,
                "level": block.level,
            })
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Heading / clause detection patterns
# ──────────────────────────────────────────────────────────────────────────────

# Matches: "1.", "1.1", "1.1.1", "Article 2", "Section 3", "CLAUSE IV"
_NUMBERED_HEADING = re.compile(
    r"^(?:(?:Article|Section|Clause|CLAUSE|SECTION|ARTICLE)\s+[\dIVXivx]+\.?|"
    r"\d+(?:\.\d+)*\.?)\s+[A-Z]",
    re.MULTILINE,
)

# All-caps short lines — likely headings (e.g. "TERMS AND CONDITIONS")
_ALLCAPS_HEADING = re.compile(r"^[A-Z][A-Z\s\-\(\)]{4,60}$")

# Signature indicators
_SIGNATURE_PATTERN = re.compile(
    r"(?:Signed|Signature|Signed by|Witness|Notary|____+)",
    re.IGNORECASE,
)

# Table indicators (simple heuristic: line contains multiple tab/pipe separators)
_TABLE_ROW = re.compile(r"(\t|  {3,}|\|)")

# Legal clause number (standalone)
_CLAUSE_NUM = re.compile(r"^\s*\d+(?:\.\d+)+\s+\S")


def _classify_line(line: str) -> str:
    """Classify a single text line into a block type."""
    stripped = line.strip()
    if not stripped:
        return "blank"
    if _SIGNATURE_PATTERN.search(stripped):
        return "signature"
    if _TABLE_ROW.search(stripped) and stripped.count("  ") > 2:
        return "table"
    if _NUMBERED_HEADING.match(stripped) or _CLAUSE_NUM.match(stripped):
        return "heading"
    if _ALLCAPS_HEADING.match(stripped) and len(stripped.split()) <= 8:
        return "heading"
    return "paragraph"


def _heading_level(text: str) -> int:
    """Determine heading depth: e.g. '1.' → 1, '1.2.' → 2, '1.2.3' → 3."""
    m = re.match(r"^(\d+(?:\.\d+)*)", text.strip())
    if m:
        return m.group(1).count(".") + 1
    if re.match(r"^(Article|Section|SECTION|ARTICLE|Clause|CLAUSE)", text.strip(), re.IGNORECASE):
        return 1
    return 1


# ──────────────────────────────────────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────────────────────────────────────

class LayoutParser:
    """
    Rule-based layout parser.
    Works on plain text (OCR output) and produces a DocumentLayout.
    """

    def parse(self, text: str, document_id: str, page_map: Optional[dict] = None) -> DocumentLayout:
        """
        Args:
            text: Full document text (all pages concatenated, or single page).
            document_id: Used to populate DocumentLayout.
            page_map: Optional dict mapping char_offset → page_number.
        """
        layout = DocumentLayout(document_id=document_id)
        lines = text.splitlines()

        buffer: List[str] = []
        buffer_type: str = "paragraph"
        char_offset = 0

        def flush_buffer(btype: str, page: Optional[int]):
            joined = " ".join(l for l in buffer if l.strip())
            if joined.strip():
                level = _heading_level(joined) if btype == "heading" else 0
                layout.blocks.append(LayoutBlock(
                    block_type=btype,
                    text=joined,
                    level=level,
                    page=page,
                ))

        current_page = 1
        for line in lines:
            # Simple page marker emitted by OCR pipeline (e.g. "--- PAGE 3 ---")
            page_match = re.match(r"---\s*PAGE\s+(\d+)\s*---", line.strip(), re.IGNORECASE)
            if page_match:
                flush_buffer(buffer_type, current_page)
                buffer = []
                current_page = int(page_match.group(1))
                continue

            ltype = _classify_line(line)

            if ltype == "blank":
                if buffer:
                    flush_buffer(buffer_type, current_page)
                    buffer = []
                    buffer_type = "paragraph"
                continue

            if ltype != buffer_type and buffer:
                flush_buffer(buffer_type, current_page)
                buffer = []

            buffer_type = ltype
            buffer.append(line)
            char_offset += len(line) + 1

        if buffer:
            flush_buffer(buffer_type, current_page)

        logger.info(
            f"Layout parsed: {len(layout.blocks)} blocks "
            f"({sum(1 for b in layout.blocks if b.block_type=='heading')} headings)"
        )
        return layout
