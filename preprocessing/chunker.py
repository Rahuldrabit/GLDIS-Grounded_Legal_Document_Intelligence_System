"""
Stage 5 — Hybrid Semantic Chunking
Splits documents into retrieval-optimised chunks using heading/clause
boundaries combined with token-count limits and overlap.
"""
from __future__ import annotations

import logging
import re
import uuid
from typing import List, Optional

from core.config import get_settings
from core.schemas import Chunk
from preprocessing.layout_parser import DocumentLayout, LayoutBlock

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Token counting (cheap approximation — no tokeniser dependency required)
# ──────────────────────────────────────────────────────────────────────────────

def _approx_tokens(text: str) -> int:
    """~4 chars per token is a reasonable estimate for English legal text."""
    return max(1, len(text) // 4)


def _split_into_sentences(text: str) -> List[str]:
    """Split text on sentence boundaries while preserving legal clause numbers."""
    # Avoid splitting on 'No. 1', 'Art. 3', abbreviations like 'Corp.'
    parts = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if p.strip()]


# ──────────────────────────────────────────────────────────────────────────────
# Core chunker
# ──────────────────────────────────────────────────────────────────────────────

class SemanticChunker:
    """
    Hybrid semantic chunker:
    1. Respects heading / clause boundaries from LayoutParser.
    2. Splits oversized sections further by sentence, respecting target_tokens.
    3. Adds overlap between chunks to maintain context continuity.
    """

    def __init__(
        self,
        target_tokens: Optional[int] = None,
        overlap_tokens: Optional[int] = None,
    ):
        settings = get_settings()
        self.target_tokens = target_tokens or settings.chunk_target_tokens
        self.overlap_tokens = overlap_tokens or settings.chunk_overlap_tokens

    # ── Public API ───────────────────────────────────────────────────────────

    def chunk_layout(self, layout: DocumentLayout, document_id: str) -> List[Chunk]:
        """Create chunks from a parsed DocumentLayout."""
        flat = layout.flat_sections()
        raw_segments = [
            (item["text"], item.get("page"), item.get("section"))
            for item in flat
            if item["text"].strip()
        ]
        return self._build_chunks(raw_segments, document_id)

    def chunk_text(
        self,
        text: str,
        document_id: str,
        page: Optional[int] = None,
        section: Optional[str] = None,
    ) -> List[Chunk]:
        """Create chunks from plain text (no layout info available)."""
        sentences = _split_into_sentences(text)
        segments = [(s, page, section) for s in sentences]
        return self._build_chunks(segments, document_id)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _build_chunks(
        self,
        segments: List[tuple],    # (text, page, section)
        document_id: str,
    ) -> List[Chunk]:
        """
        Pack segments into chunks respecting target_tokens.
        Adds token-level overlap between consecutive chunks.
        """
        chunks: List[Chunk] = []
        current_texts: List[str] = []
        current_tokens = 0
        current_page: Optional[int] = None
        current_section: Optional[str] = None
        chunk_index = 0

        def emit_chunk():
            nonlocal chunk_index
            joined = " ".join(current_texts)
            if not joined.strip():
                return
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                text=joined,
                page=current_page,
                section=current_section,
                token_count=_approx_tokens(joined),
                metadata={
                    "chunk_index": chunk_index,
                    "source": "layout_chunker",
                },
            ))
            chunk_index += 1

        for text, page, section in segments:
            seg_tokens = _approx_tokens(text)

            # Section boundary — emit current buffer, start fresh
            if section and section != current_section and current_texts:
                emit_chunk()
                # Carry overlap from end of previous chunk
                overlap_text = self._overlap_text(current_texts)
                current_texts = [overlap_text] if overlap_text else []
                current_tokens = _approx_tokens(overlap_text)
                current_section = section
                current_page = page

            # Segment alone exceeds target → split further by sentence
            if seg_tokens > self.target_tokens:
                sub_chunks = self._split_large_segment(text, page, section, document_id, chunk_index)
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
                # Carry tail as overlap
                if sub_chunks:
                    tail = sub_chunks[-1].text
                    current_texts = [self._overlap_text([tail])]
                    current_tokens = _approx_tokens(current_texts[0])
                continue

            # Adding segment would overflow target → flush and start new chunk
            if current_tokens + seg_tokens > self.target_tokens and current_texts:
                emit_chunk()
                overlap_text = self._overlap_text(current_texts)
                current_texts = [overlap_text] if overlap_text else []
                current_tokens = _approx_tokens(overlap_text)

            current_texts.append(text)
            current_tokens += seg_tokens
            if page:
                current_page = page
            if section:
                current_section = section

        # Flush remaining
        emit_chunk()

        logger.info(f"Chunking complete: {len(chunks)} chunks for document {document_id}")
        return chunks

    def _overlap_text(self, texts: List[str]) -> str:
        """Return the last overlap_tokens worth of words from current buffer."""
        joined = " ".join(texts)
        words = joined.split()
        overlap_word_count = max(1, self.overlap_tokens // 4)  # ~4 chars/word
        return " ".join(words[-overlap_word_count:]) if len(words) > overlap_word_count else joined

    def _split_large_segment(
        self,
        text: str,
        page: Optional[int],
        section: Optional[str],
        document_id: str,
        start_index: int,
    ) -> List[Chunk]:
        """Recursively split a segment that exceeds target_tokens."""
        sentences = _split_into_sentences(text)
        buffer: List[str] = []
        buf_tokens = 0
        sub_chunks: List[Chunk] = []
        idx = start_index

        for sent in sentences:
            stokens = _approx_tokens(sent)
            if buf_tokens + stokens > self.target_tokens and buffer:
                joined = " ".join(buffer)
                sub_chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    text=joined,
                    page=page,
                    section=section,
                    token_count=_approx_tokens(joined),
                    metadata={"chunk_index": idx, "source": "split_large"},
                ))
                idx += 1
                overlap = self._overlap_text(buffer)
                buffer = [overlap] if overlap else []
                buf_tokens = _approx_tokens(overlap)
            buffer.append(sent)
            buf_tokens += stokens

        if buffer:
            joined = " ".join(buffer)
            sub_chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                text=joined,
                page=page,
                section=section,
                token_count=_approx_tokens(joined),
                metadata={"chunk_index": idx, "source": "split_large"},
            ))

        return sub_chunks
