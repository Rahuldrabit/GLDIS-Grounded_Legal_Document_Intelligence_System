"""
Post-hoc grounding verification.
Scores how well a generated draft is supported by the retrieved evidence.
"""
from __future__ import annotations

import logging
import re
from typing import List, Tuple

from core.schemas import Citation, EvidenceChunk

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Citation extraction
# ──────────────────────────────────────────────────────────────────────────────

_CITATION_RE = re.compile(
    r"\[Source:\s*([a-f0-9\-]{36}),?\s*(?:p\.(\d+))?\]",
    re.IGNORECASE,
)


def extract_citations(
    text: str, evidence_chunks: List[EvidenceChunk]
) -> List[Citation]:
    """
    Parse inline [Source: chunk_id, p.X] markers from generated text.
    Returns Citation objects validated against the evidence_chunks list.
    """
    chunk_lookup = {c.chunk_id: c for c in evidence_chunks}
    citations: List[Citation] = []
    seen: set = set()

    for m in _CITATION_RE.finditer(text):
        chunk_id = m.group(1)
        page = int(m.group(2)) if m.group(2) else None

        if chunk_id in seen:
            continue
        seen.add(chunk_id)

        chunk = chunk_lookup.get(chunk_id)
        if chunk:
            # Grab a short excerpt centred on the citation
            start = max(0, m.start() - 100)
            excerpt = text[start : m.start()].strip()[-120:]
            citations.append(Citation(
                chunk_id=chunk_id,
                document_id=chunk.document_id,
                page=page or chunk.page,
                excerpt=excerpt,
            ))
        else:
            # Unknown chunk_id referenced — still record it
            citations.append(Citation(
                chunk_id=chunk_id,
                document_id="unknown",
                page=page,
                excerpt="",
            ))

    return citations


# ──────────────────────────────────────────────────────────────────────────────
# Sentence-level grounding score
# ──────────────────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, stripping headings and empty lines."""
    lines = text.splitlines()
    sentences = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("---"):
            continue
        # Remove markdown bold/italic markers
        line = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", line)
        # Split long lines on sentence boundaries
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", line)
        sentences.extend([p.strip() for p in parts if len(p.strip()) > 20])
    return sentences


def _token_overlap(a: str, b: str) -> float:
    """Jaccard token overlap between two strings."""
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def compute_grounding_score(
    generated_text: str,
    evidence_chunks: List[EvidenceChunk],
    threshold: float = 0.15,
) -> Tuple[float, List[str]]:
    """
    Compute a grounding score in [0, 1] by checking what fraction of
    generated sentences are supported by the evidence.

    Also returns a list of ungrounded sentences.

    Strategy:
    - Split generated text into content sentences
    - For each sentence, compute max Jaccard overlap with any evidence chunk
    - Sentence is "grounded" if max overlap >= threshold OR it contains a [Source:] citation
    - Score = grounded_count / total_count
    """
    if not evidence_chunks:
        return 0.0, []

    sentences = _split_sentences(generated_text)
    if not sentences:
        return 1.0, []  # Empty output is vacuously grounded

    evidence_texts = [c.text for c in evidence_chunks]
    grounded_count = 0
    ungrounded: List[str] = []

    for sent in sentences:
        # Fast path: sentence contains an explicit citation
        if _CITATION_RE.search(sent):
            grounded_count += 1
            continue

        # Compute overlap with each evidence chunk
        max_overlap = max(
            _token_overlap(sent, ev) for ev in evidence_texts
        )

        if max_overlap >= threshold:
            grounded_count += 1
        else:
            ungrounded.append(sent)

    score = grounded_count / len(sentences)
    logger.info(
        f"Grounding score: {score:.2f} "
        f"({grounded_count}/{len(sentences)} sentences supported)"
    )
    return round(score, 4), ungrounded
