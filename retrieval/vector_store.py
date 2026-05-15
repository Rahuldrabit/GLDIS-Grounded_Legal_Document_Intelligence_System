"""
Stages 7-8 — Embedding Generation + FAISS Vector Store
Converts text chunks into dense vector embeddings and manages
a persistent FAISS index for similarity search.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.config import get_settings
from core.schemas import Chunk

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Embedding model (lazy-loaded singleton)
# ──────────────────────────────────────────────────────────────────────────────

_EMBED_MODEL = None

def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
    try:
        from sentence_transformers import SentenceTransformer
        settings = get_settings()
        _EMBED_MODEL = SentenceTransformer(settings.embedding_model)
        logger.info(f"Embedding model loaded: {settings.embedding_model}")
    except ImportError:
        logger.error("sentence-transformers not installed. Embeddings unavailable.")
    return _EMBED_MODEL


def embed_texts(texts: List[str], batch_size: int = 32) -> Optional[np.ndarray]:
    """Return (N, D) float32 array of embeddings, or None on failure."""
    model = _get_embed_model()
    if model is None:
        return None
    settings = get_settings()
    embeddings = model.encode(
        texts,
        batch_size=batch_size or settings.embedding_batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,   # cosine similarity via dot product
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# FAISS Index Manager
# ──────────────────────────────────────────────────────────────────────────────

class FAISSStore:
    """
    Persistent FAISS index with a sidecar metadata JSON file.

    Layout on disk:
        <faiss_index_path>/
            index.faiss          — FAISS flat L2 index
            metadata.pkl         — list of chunk metadata dicts
    """

    INDEX_FILE = "index.faiss"
    META_FILE  = "metadata.pkl"

    def __init__(self):
        settings = get_settings()
        self.index_dir = Path(settings.faiss_index_path)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._index = None
        self._meta: List[dict] = []
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        try:
            import faiss
            idx_path = self.index_dir / self.INDEX_FILE
            meta_path = self.index_dir / self.META_FILE
            if idx_path.exists() and meta_path.exists():
                self._index = faiss.read_index(str(idx_path))
                with open(meta_path, "rb") as f:
                    self._meta = pickle.load(f)
                logger.info(f"FAISS index loaded: {self._index.ntotal} vectors")
        except Exception as exc:
            logger.warning(f"Could not load FAISS index: {exc}")

    def _save(self):
        try:
            import faiss
            faiss.write_index(self._index, str(self.index_dir / self.INDEX_FILE))
            with open(self.index_dir / self.META_FILE, "wb") as f:
                pickle.dump(self._meta, f)
        except Exception as exc:
            logger.error(f"FAISS save failed: {exc}")

    # ── Index operations ─────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Chunk]) -> int:
        """Embed and add chunks to the index. Returns number added."""
        if not chunks:
            return 0
        import faiss

        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts)
        if embeddings is None:
            logger.error("Embedding generation failed; skipping FAISS add.")
            return 0

        dim = embeddings.shape[1]

        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)  # Inner product = cosine (normalized)
            logger.info(f"Created new FAISS index dim={dim}")

        self._index.add(embeddings)

        for chunk in chunks:
            self._meta.append({
                "chunk_id":    chunk.chunk_id,
                "document_id": chunk.document_id,
                "text":        chunk.text,
                "page":        chunk.page,
                "section":     chunk.section,
            })

        self._save()
        logger.info(f"Added {len(chunks)} chunks to FAISS (total: {self._index.ntotal})")
        return len(chunks)

    def search(
        self, query: str, top_k: int = 10
    ) -> List[Tuple[dict, float]]:
        """
        Search the index for the top_k most similar chunks.
        Returns list of (metadata_dict, score) tuples.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        q_emb = embed_texts([query])
        if q_emb is None:
            return []

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._meta):
                continue
            results.append((self._meta[idx], float(score)))

        return results

    def delete_document(self, document_id: str) -> int:
        """Remove all vectors for a document (rebuilds index)."""
        if self._index is None:
            return 0
        import faiss

        keep_indices = [
            i for i, m in enumerate(self._meta)
            if m["document_id"] != document_id
        ]
        removed = len(self._meta) - len(keep_indices)

        if removed == 0:
            return 0

        # Rebuild keeping only the retained vectors
        old_meta = self._meta
        dim = self._index.d

        all_vecs = np.zeros((self._index.ntotal, dim), dtype=np.float32)
        self._index.reconstruct_n(0, self._index.ntotal, all_vecs)

        kept_vecs = all_vecs[keep_indices]
        self._index = faiss.IndexFlatIP(dim)
        if len(kept_vecs):
            self._index.add(kept_vecs)
        self._meta = [old_meta[i] for i in keep_indices]
        self._save()

        logger.info(f"Deleted {removed} vectors for document {document_id}")
        return removed

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0
