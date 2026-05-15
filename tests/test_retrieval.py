"""Tests for hybrid retrieval: FAISS, BM25, RRF (Steps 19–23)."""
from __future__ import annotations
import uuid
import pytest
from core.schemas import Chunk


def _make_chunks(texts, doc_id="test-doc"):
    return [
        Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id=doc_id,
            text=t,
            page=i + 1,
            section=f"Section {i + 1}",
            token_count=len(t.split()),
        )
        for i, t in enumerate(texts)
    ]


def test_bm25_add_and_search(tmp_path, monkeypatch):
    monkeypatch.setenv("BM25_INDEX_PATH", str(tmp_path))
    from core.config import get_settings
    get_settings.cache_clear()
    from retrieval.bm25_retriever import BM25Store
    store = BM25Store()
    chunks = _make_chunks([
        "The tenant shall pay monthly rent of twelve thousand dollars.",
        "Landlord agrees to maintain heating systems in the building.",
        "Case No. CV-2024-001 is pending before the court.",
    ])
    store.add_chunks(chunks)
    results = store.search("monthly rent payment", top_k=3)
    assert len(results) >= 1
    assert "rent" in results[0][0]["text"].lower() or "tenant" in results[0][0]["text"].lower()


def test_bm25_empty_search(tmp_path, monkeypatch):
    monkeypatch.setenv("BM25_INDEX_PATH", str(tmp_path))
    from core.config import get_settings
    get_settings.cache_clear()
    from retrieval.bm25_retriever import BM25Store
    store = BM25Store()
    assert store.search("anything", top_k=5) == []


def test_bm25_delete_document(tmp_path, monkeypatch):
    monkeypatch.setenv("BM25_INDEX_PATH", str(tmp_path))
    from core.config import get_settings
    get_settings.cache_clear()
    from retrieval.bm25_retriever import BM25Store
    store = BM25Store()
    doc_id = "delete-me"
    store.add_chunks(_make_chunks(["This should be deleted."], doc_id=doc_id))
    removed = store.delete_document(doc_id)
    assert removed == 1


def test_rrf_merges_lists():
    from retrieval.hybrid_retriever import _reciprocal_rank_fusion
    list_a = [("c1", 0.9), ("c2", 0.8), ("c3", 0.7)]
    list_b = [("c2", 0.95), ("c4", 0.85), ("c1", 0.6)]
    fused = _reciprocal_rank_fusion([list_a, list_b])
    ids = [f[0] for f in fused]
    assert "c1" in ids[:3] and "c2" in ids[:3]


def test_rrf_single_list():
    from retrieval.hybrid_retriever import _reciprocal_rank_fusion
    fused = _reciprocal_rank_fusion([[("a", 1.0), ("b", 0.8)]])
    assert fused[0][0] == "a"
