"""
Stage 9b — Hybrid Retriever
Combines BM25 (sparse) and FAISS (dense) with Reciprocal Rank Fusion,
plus optional cross-encoder reranking.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from core.config import get_settings
from core.schemas import Chunk, EvidenceChunk
from retrieval.bm25_retriever import BM25Store
from retrieval.graph_store import get_graph_store
from retrieval.vector_store import FAISSStore

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Reciprocal Rank Fusion
# ──────────────────────────────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    ranked_lists: List[List[tuple]],  # each: list of (chunk_id, score)
    k: int = 60,
    weights: Optional[List[float]] = None,
) -> List[tuple]:
    """
    Merge multiple ranked lists using RRF.
    Returns sorted list of (chunk_id, fused_score).
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: Dict[str, float] = {}
    for w, ranked in zip(weights, ranked_lists):
        for rank, (chunk_id, _score) in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + w / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ──────────────────────────────────────────────────────────────────────────────
# Cross-encoder reranker (optional)
# ──────────────────────────────────────────────────────────────────────────────

_RERANKER = None

def _get_reranker():
    """Lazily load a cross-encoder model for reranking."""
    global _RERANKER
    if _RERANKER is not None:
        return _RERANKER
    try:
        from sentence_transformers import CrossEncoder
        _RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        logger.info("Cross-encoder reranker loaded.")
    except Exception as exc:
        logger.warning(f"Cross-encoder unavailable: {exc}. Skipping reranking.")
    return _RERANKER


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid Retriever
# ──────────────────────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Unified retriever combining:
    1. FAISS dense vector search
    2. BM25 sparse keyword search
    3. Reciprocal Rank Fusion
    4. Optional cross-encoder reranking
    """

    def __init__(
        self,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        use_reranker: bool = False,
    ):
        self.faiss_store = FAISSStore()
        self.bm25_store = BM25Store()
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.use_reranker = use_reranker
        self._meta_cache: Dict[str, dict] = {}

    def add_chunks(self, chunks: List[Chunk]) -> int:
        """Add chunks to both indices."""
        n_faiss = self.faiss_store.add_chunks(chunks)
        n_bm25 = self.bm25_store.add_chunks(chunks)

        # Cache metadata for retrieval
        for chunk in chunks:
            self._meta_cache[chunk.chunk_id] = {
                "chunk_id":    chunk.chunk_id,
                "document_id": chunk.document_id,
                "text":        chunk.text,
                "page":        chunk.page,
                "section":     chunk.section,
            }

        logger.info(f"Indexed {n_faiss} chunks (FAISS) + {n_bm25} chunks (BM25)")
        return max(n_faiss, n_bm25)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
        document_id: Optional[str] = None,
    ) -> List[EvidenceChunk]:
        """
        Retrieve evidence chunks using hybrid search.
        
        Args:
            query: The search query or drafting prompt.
            top_k: Number of candidates to retrieve from each index.
            rerank_top_k: Number of final results after reranking.
            document_id: Optional filter to search within a specific document.
        """
        settings = get_settings()
        top_k = top_k or settings.retrieval_top_k
        rerank_top_k = rerank_top_k or settings.rerank_top_k

        # 1. Retrieve from both indices
        faiss_results = self.faiss_store.search(query, top_k=top_k)
        bm25_results = self.bm25_store.search(query, top_k=top_k)

        # Filter by document_id if specified
        if document_id:
            faiss_results = [(m, s) for m, s in faiss_results if m["document_id"] == document_id]
            bm25_results = [(m, s) for m, s in bm25_results if m["document_id"] == document_id]

        # 2. Convert to (chunk_id, score) format for RRF
        faiss_ranked = [(m["chunk_id"], s) for m, s in faiss_results]
        bm25_ranked = [(m["chunk_id"], s) for m, s in bm25_results]

        # Build lookup: chunk_id → metadata
        meta_lookup = {}
        for m, _ in faiss_results + bm25_results:
            meta_lookup[m["chunk_id"]] = m

        # 3. Reciprocal Rank Fusion
        fused = _reciprocal_rank_fusion(
            [faiss_ranked, bm25_ranked],
            weights=[self.dense_weight, self.sparse_weight],
        )

        # 4. Optional cross-encoder reranking
        if self.use_reranker and len(fused) > rerank_top_k:
            fused = self._rerank(query, fused, meta_lookup, rerank_top_k)
        else:
            fused = fused[:rerank_top_k]

        # 5. Build EvidenceChunk objects
        evidence = []
        for chunk_id, score in fused:
            meta = meta_lookup.get(chunk_id, self._meta_cache.get(chunk_id, {}))
            if not meta:
                continue
            evidence.append(EvidenceChunk(
                chunk_id=chunk_id,
                document_id=meta.get("document_id", ""),
                text=meta.get("text", ""),
                page=meta.get("page"),
                section=meta.get("section"),
                score=score,
                retrieval_method="hybrid_rrf",
            ))

        logger.info(
            f"Hybrid search: {len(faiss_results)} dense + {len(bm25_results)} sparse "
            f"→ {len(evidence)} results"
        )
        
        # Optional: GraphRAG neighborhood expansion
        try:
            settings = get_settings()
            if getattr(settings, "graphrag_enabled", False):
                graph_store = get_graph_store()
                if graph_store and graph_store.enabled:
                    top_chunk_ids = [cid for cid, _ in fused]
                    hops = getattr(settings, "graph_expand_hops", 1)
                    logger.info(f"GraphRAG: expanding {len(top_chunk_ids)} chunks by {hops} hops")
                    expanded = graph_store.expand_chunk_neighborhood(top_chunk_ids, hops=hops, document_id=document_id)
                    # Merge expanded chunks, avoiding duplicates
                    existing_ids = {e.chunk_id for e in evidence}
                    for ex in expanded:
                        if ex.chunk_id in existing_ids:
                            continue
                        # Try to fill metadata if missing by checking caches
                        if not ex.text:
                            meta = self._meta_cache.get(ex.chunk_id)
                            if not meta:
                                # Fallback: search FAISSStore meta
                                for m in getattr(self.faiss_store, "_meta", []):
                                    if m.get("chunk_id") == ex.chunk_id:
                                        meta = m
                                        break
                            if meta:
                                ex.text = meta.get("text", "")
                                ex.document_id = meta.get("document_id", ex.document_id)
                                ex.page = meta.get("page", ex.page)
                                ex.section = meta.get("section", ex.section)
                        evidence.append(ex)
                    logger.info(f"GraphRAG expansion added {len(expanded)} neighbors")
        except Exception as exc:
            logger.warning(f"GraphRAG expansion failed: {exc}")

        return evidence

    def _rerank(
        self,
        query: str,
        candidates: List[tuple],
        meta_lookup: dict,
        top_k: int,
    ) -> List[tuple]:
        """Rerank candidates using a cross-encoder model."""
        reranker = _get_reranker()
        if reranker is None:
            return candidates[:top_k]

        pairs = []
        valid_ids = []
        for chunk_id, _ in candidates:
            meta = meta_lookup.get(chunk_id, {})
            text = meta.get("text", "")
            if text:
                pairs.append((query, text))
                valid_ids.append(chunk_id)

        if not pairs:
            return candidates[:top_k]

        try:
            scores = reranker.predict(pairs)
            reranked = sorted(
                zip(valid_ids, scores),
                key=lambda x: x[1],
                reverse=True,
            )
            return [(cid, float(s)) for cid, s in reranked[:top_k]]
        except Exception as exc:
            logger.warning(f"Reranking failed: {exc}")
            return candidates[:top_k]

    def delete_document(self, document_id: str) -> int:
        """Remove a document from both indices."""
        n1 = self.faiss_store.delete_document(document_id)
        n2 = self.bm25_store.delete_document(document_id)
        # Clean meta cache
        self._meta_cache = {
            k: v for k, v in self._meta_cache.items()
            if v.get("document_id") != document_id
        }
        return max(n1, n2)

    @property
    def total_indexed(self) -> int:
        return self.faiss_store.total_vectors
