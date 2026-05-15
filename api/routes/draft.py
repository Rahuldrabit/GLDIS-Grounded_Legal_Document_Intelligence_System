"""
API Routes: RAG Draft Generation
Retrieves evidence and generates grounded legal drafts.
"""
from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.schemas import DraftRequest, DraftResponse
from db.models import Document, Draft
from db.session import get_db

from retrieval.hybrid_retriever import HybridRetriever
from generation.rag_generator import RAGGenerator
from feedback.prompt_optimizer import PromptOptimizer

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/", response_model=DraftResponse)
async def generate_draft(
    request: DraftRequest,
    db: Session = Depends(get_db)
) -> DraftResponse:
    """Generate a grounded draft using hybrid retrieval and an LLM."""
    
    # 1. Validate doc
    doc = db.query(Document).filter(Document.id == request.document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.status != "ready":
        raise HTTPException(status_code=400, detail="Document is not fully processed yet.")
        
    # 2. Retrieve Evidence
    retriever = HybridRetriever()
    evidence = retriever.search(
        query=request.query,
        document_id=request.document_id,
        rerank_top_k=request.top_k
    )
    
    if not evidence:
        raise HTTPException(status_code=404, detail="No relevant evidence found for this document.")
        
    # 3. Get Few-Shot Examples (Learning Loop)
    optimizer = PromptOptimizer(db)
    examples = optimizer.get_best_examples(limit=2)
    
    # 4. Generate
    generator = RAGGenerator()
    try:
        draft_response = generator.generate_draft(
            document_id=request.document_id,
            query=request.query,
            evidence=evidence,
            few_shot_examples=examples
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(exc)}")
        
    # 5. Save Draft to DB
    db_draft = Draft(
        draft_id=draft_response.draft_id,
        document_id=request.document_id,
        query=request.query,
        generated_text=draft_response.generated_text,
        evidence_used=[e.chunk_id for e in evidence],
        citations=[c.model_dump() for c in draft_response.citations],
        grounding_score=draft_response.grounding_score,
        model_used=generator.model
    )
    db.add(db_draft)
    db.commit()
    
    return draft_response
