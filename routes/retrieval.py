"""Routes: Document search and evidence retrieval."""
from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.schemas import EvidenceChunk, SearchResultResponse, EvidenceChunkResponse
from core.rate_limit import rate_limit_guard
from db import models
from db.session import get_db
from retrieval.hybrid_retriever import HybridRetriever

router = APIRouter(prefix="/api/search", tags=["retrieval"])

_retriever = HybridRetriever()


@router.post(
    "",
    response_model=SearchResultResponse,
    dependencies=[rate_limit_guard(scope="llm_search")],
)
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


from pydantic import BaseModel
from typing import List, Optional
from fastapi import Header

class StatelessQueryRequest(BaseModel):
    query: str
    document_text: str
    top_k: int = 5

@router.post("/stateless-query")
def stateless_query(
    req: StatelessQueryRequest,
    x_llm_provider: Optional[str] = Header(None, alias="X-LLM-Provider"),
    x_llm_api_key: Optional[str] = Header(None, alias="X-LLM-Api-Key"),
    x_llm_model: Optional[str] = Header(None, alias="X-LLM-Model"),
):
    """
    Stateless query directly against provided document text.
    For simplicity in stateless mode, we chunk the text on the fly, 
    perform simple keyword/substring search, and generate an answer.
    """
    if not req.query.strip() or not req.document_text.strip():
        raise HTTPException(400, "Query and document text are required.")

    # 1. Simple in-memory chunking
    # Split by double newlines or large blocks
    paragraphs = [p.strip() for p in req.document_text.split('\n\n') if p.strip()]
    
    # 2. Simple retrieval (keyword overlap)
    query_words = set(req.query.lower().split())
    
    scored_chunks = []
    for i, p in enumerate(paragraphs):
        p_words = set(p.lower().split())
        score = len(query_words.intersection(p_words))
        scored_chunks.append({"chunk_id": f"chunk_{i}", "text": p, "score": score, "page": 1})
        
    # Sort by score and take top_k
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    top_chunks = scored_chunks[:req.top_k]
    
    # Generate Answer using llm.client directly
    from llm.client import chat_completion
    from generation.rag_generator import SYSTEM_PROMPT
    
    # Format Evidence
    evidence_text = ""
    for chunk in top_chunks:
        evidence_text += f"[CHUNK_ID: {chunk['chunk_id']}] (Page: {chunk['page']})\n{chunk['text']}\n---\n"
        
    user_prompt = f"Evidence:\n{evidence_text}\n\nTask:\n{req.query}"
    
    response = chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        mode="text",
        header_provider=x_llm_provider,
        header_api_key=x_llm_api_key,
        header_model=x_llm_model
    )
    
    generated_text = response.choices[0].message.content or ""
    
    return {
        "query": req.query,
        "generated_text": generated_text,
        "evidence_chunks": top_chunks
    }



@router.get("/chunk/{chunk_id}", response_model=EvidenceChunkResponse)
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
