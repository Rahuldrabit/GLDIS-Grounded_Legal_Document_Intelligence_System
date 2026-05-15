"""Routes: Document search and evidence retrieval."""
from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.schemas import EvidenceChunk
from db import models
from db.session import get_db
from retrieval.hybrid_retriever import HybridRetriever

router = APIRouter(prefix="/api/search", tags=["retrieval"])

_retriever = HybridRetriever()


@router.post("")
def search(
    query: str,
    document_id: Optional[str] = None,
    top_k: int = 5,
    db: Session = Depends(get_db),
):
    """
    Search across all indexed documents (or a specific document).
    Returns grounded evidence passages ranked by relevance.
    """
    if not query.strip():
        raise HTTPException(400, "Query cannot be empty.")

    results = _retriever.search(
        query=query,
        top_k=top_k * 2,
        rerank_top_k=top_k,
        document_id=document_id,
    )

    return {
        "query": query,
        "document_id": document_id,
        "results_count": len(results),
        "results": [
            {
                "chunk_id": r.chunk_id,
                "document_id": r.document_id,
                "text": r.text,
                "page": r.page,
                "section": r.section,
                "score": r.score,
                "retrieval_method": r.retrieval_method,
            }
            for r in results
        ],
    }


@router.get("/chunk/{chunk_id}")
def get_chunk(chunk_id: str, db: Session = Depends(get_db)):
    """Retrieve a specific evidence chunk by ID."""
    chunk = db.query(models.Chunk).filter(models.Chunk.chunk_id == chunk_id).first()
    if not chunk:
        raise HTTPException(404, f"Chunk {chunk_id} not found.")
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "text": chunk.text,
        "page": chunk.page,
        "section": chunk.section,
        "token_count": chunk.token_count,
        "chunk_index": chunk.chunk_index,
    }
