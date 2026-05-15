"""
Stage 9a — BM25 Sparse Retrieval
Keyword-based retrieval using Okapi BM25 with legal-aware tokenization.
Persistent index via pickle.
"""
from __future__ import annotations

import logging
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.config import get_settings

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Legal-aware tokenizer
# ──────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "under",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "no", "only", "own", "same", "than", "too",
    "very", "just", "because", "if", "when", "while", "this", "that",
    "these", "those", "it", "its", "he", "she", "they", "them", "his",
    "her", "their", "our", "we", "you", "your", "my", "me", "us",
}


def _tokenize(text: str) -> List[str]:
    """
    Legal-aware tokenizer:
    - Lowercases
    - Preserves clause numbers (e.g., '3.1.2')
    - Preserves case IDs (e.g., 'CV-2024-0042')
    - Removes stopwords
    - Strips very short tokens
    """
    text = text.lower()
    # Split on whitespace and punctuation, but preserve periods in numbers
    # and hyphens in case IDs
    tokens = re.findall(r"\b(?:\d+(?:\.\d+)+|\w+(?:-\w+)*|\w+)\b", text)
    return [t for t in tokens if t not in _STOPWORDS and len(t) >= 2]


# ──────────────────────────────────────────────────────────────────────────────
# BM25 Index
# ──────────────────────────────────────────────────────────────────────────────

class BM25Store:
    """
    Persistent BM25 index for sparse keyword retrieval.
    
    Layout on disk:
        <bm25_index_path>/
            bm25_index.pkl   — serialized BM25 model + metadata
    """

    INDEX_FILE = "bm25_index.pkl"

    def __init__(self):
        settings = get_settings()
        self.index_dir = Path(settings.bm25_index_path)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._bm25 = None
        self._corpus_tokens: List[List[str]] = []
        self._meta: List[dict] = []
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        path = self.index_dir / self.INDEX_FILE
        if path.exists():
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                self._corpus_tokens = data["corpus_tokens"]
                self._meta = data["meta"]
                self._rebuild_index()
                logger.info(f"BM25 index loaded: {len(self._meta)} documents")
            except Exception as exc:
                logger.warning(f"Could not load BM25 index: {exc}")

    def _save(self):
        try:
            path = self.index_dir / self.INDEX_FILE
            with open(path, "wb") as f:
                pickle.dump({
                    "corpus_tokens": self._corpus_tokens,
                    "meta": self._meta,
                }, f)
        except Exception as exc:
            logger.error(f"BM25 save failed: {exc}")

    def _rebuild_index(self):
        """Rebuild BM25 from stored corpus tokens."""
        if not self._corpus_tokens:
            self._bm25 = None
            return
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._corpus_tokens)
        except ImportError:
            logger.warning("rank-bm25 not installed; using simple overlap fallback.")
            self._bm25 = None

    # ── Index operations ─────────────────────────────────────────────────────

    def add_chunks(self, chunks: list) -> int:
        """Add chunks to the BM25 index."""
        if not chunks:
            return 0

        for chunk in chunks:
            tokens = _tokenize(chunk.text)
            self._corpus_tokens.append(tokens)
            self._meta.append({
                "chunk_id":    chunk.chunk_id,
                "document_id": chunk.document_id,
                "text":        chunk.text,
                "page":        chunk.page,
                "section":     chunk.section,
            })

        self._rebuild_index()
        self._save()
        logger.info(f"Added {len(chunks)} chunks to BM25 (total: {len(self._meta)})")
        return len(chunks)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[dict, float]]:
        """Search the BM25 index. Returns list of (metadata_dict, score)."""
        if not self._meta:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        # Prefer rank_bm25 if available
        if self._bm25 is not None:
            scores = self._bm25.get_scores(query_tokens)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    results.append((self._meta[idx], float(scores[idx])))
            return results

        # Fallback: simple token-overlap scoring (fast, deterministic)
        scores = []
        qset = set(query_tokens)
        for tokens in self._corpus_tokens:
            tset = set(tokens)
            overlap = len(qset & tset)
            scores.append(float(overlap))

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self._meta[idx], float(scores[idx])))
        return results

    def delete_document(self, document_id: str) -> int:
        """Remove all entries for a document and rebuild index."""
        keep = [
            (tokens, meta)
            for tokens, meta in zip(self._corpus_tokens, self._meta)
            if meta["document_id"] != document_id
        ]
        removed = len(self._meta) - len(keep)
        if removed == 0:
            return 0

        if keep:
            self._corpus_tokens, self._meta = zip(*keep)
            self._corpus_tokens = list(self._corpus_tokens)
            self._meta = list(self._meta)
        else:
            self._corpus_tokens = []
            self._meta = []

        self._rebuild_index()
        self._save()
        logger.info(f"Deleted {removed} entries from BM25 for document {document_id}")
        return removed

    @property
    def total_documents(self) -> int:
        return len(self._meta)
