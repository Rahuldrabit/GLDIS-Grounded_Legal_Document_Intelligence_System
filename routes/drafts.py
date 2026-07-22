"""Routes: Draft generation and retrieval."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.schemas import DraftRequest, DraftResponse, DocumentStatus, EvidenceChunk
from core.rate_limit import rate_limit_guard
from db import models
from db.session import get_db
from feedback.improvement_loop import ImprovementLoop
from generation.generator import DraftGenerator
from ingestion.orchestrator import get_structured_extraction
from retrieval.hybrid_retriever import HybridRetriever

router = APIRouter(prefix="/api/drafts", tags=["drafts"])

_retriever = HybridRetriever()
_generator = DraftGenerator()
_improvement_loop = ImprovementLoop()


@router.post(
    "/generate",
    response_model=DraftResponse,
    dependencies=[rate_limit_guard(scope="llm_generate")],
)
def generate_draft(request: DraftRequest, db: Session = Depends(get_db)):
    """
    Generate a grounded Case Fact Summary + Internal Memo.
    Automatically injects learned few-shot examples and style rules.
    """
    # Validate document exists and is ready
    doc = db.query(models.Document).filter(models.Document.id == request.document_id).first()
    if not doc:
        raise HTTPException(404, f"Document {request.document_id} not found.")
    if doc.status != DocumentStatus.READY.value:
        raise HTTPException(
            409,
            f"Document is not ready (status: {doc.status}). "
            "Process it first with POST /api/documents/{id}/process/sync"
        )

    # Retrieve evidence
    evidence = _retriever.search(
        query=request.query or "Generate a case fact summary and internal memo.",
        top_k=request.top_k * 2,
        rerank_top_k=request.top_k,
        document_id=request.document_id,
    )

    # If hybrid retrieval returns nothing, fall back to the document's stored chunks.
    # This keeps draft generation usable even when the vector index is missing or weak.
    if not evidence:
        chunks = (
            db.query(models.Chunk)
            .filter(models.Chunk.document_id == request.document_id)
            .order_by(models.Chunk.chunk_index.asc(), models.Chunk.page.asc().nullslast())
            .limit(request.top_k)
            .all()
        )
        evidence = [
            EvidenceChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                page=chunk.page,
                section=chunk.section,
                score=0.0,
                retrieval_method="document_chunks",
            )
            for chunk in chunks
        ]

    if not evidence:
        raise HTTPException(
            422,
            "No evidence found for this document. "
            "Ensure the document was processed and indexed."
        )

    # Get structured extraction
    structured = get_structured_extraction(request.document_id, db)

    # Get improvement context from feedback loop
    improvement_ctx = _improvement_loop.get_generation_context(db)

    draft_type = getattr(request, "draft_type", "case_summary_memo") or "case_summary_memo"

    # Generate draft
    draft = _generator.generate(
        request=request,
        evidence_chunks=evidence,
        structured_data=structured,
        few_shot_examples=improvement_ctx.get("few_shot_examples"),
        style_rules=improvement_ctx.get("style_rules"),
        draft_type=draft_type,
    )

    # Persist draft to DB
    db_draft = models.Draft(
        draft_id=draft.draft_id,
        document_id=request.document_id,
        query=request.query,
        generated_text=draft.generated_text,
        evidence_used=[c.chunk_id for c in evidence],
        citations=[c.model_dump() for c in draft.citations],
        grounding_score=draft.grounding_score,
        draft_type=draft_type,
    )
    db.add(db_draft)
    db.commit()

    return draft

from pydantic import BaseModel
from fastapi import Header
from typing import Optional

class StatelessDraftRequest(BaseModel):
    document_text: str
    query: str = "Generate a case fact summary and internal memo."
    top_k: int = 5

@router.post("/stateless-generate")
def stateless_generate(
    req: StatelessDraftRequest,
    x_llm_provider: Optional[str] = Header(None, alias="X-LLM-Provider"),
    x_llm_api_key: Optional[str] = Header(None, alias="X-LLM-Api-Key"),
    x_llm_model: Optional[str] = Header(None, alias="X-LLM-Model"),
):
    """Stateless draft generation without relying on DB."""
    if not req.document_text.strip():
        raise HTTPException(400, "Document text is required.")

    # 1. Chunking
    paragraphs = [p.strip() for p in req.document_text.split('\n\n') if p.strip()]
    
    # 2. Simple retrieval
    query_words = set(req.query.lower().split())
    scored_chunks = []
    for i, p in enumerate(paragraphs):
        p_words = set(p.lower().split())
        score = len(query_words.intersection(p_words))
        scored_chunks.append(EvidenceChunk(
            chunk_id=f"chunk_{i}",
            document_id="stateless",
            text=p,
            page=1,
            section=None,
            score=score,
            retrieval_method="stateless"
        ))
        
    scored_chunks.sort(key=lambda x: x.score, reverse=True)
    top_chunks = scored_chunks[:req.top_k]

    # Generate Draft
    from generation.rag_generator import RAGGenerator
    
    # We need to temporarily set the config to use BYOK for RAGGenerator
    # But RAGGenerator reads from config in __init__. So we'll instantiate it 
    # and call it. But RAGGenerator calls chat_completion which reads resolve_llm_config().
    # So we should modify RAGGenerator to accept BYOK or just use chat_completion directly.
    # Let's just use chat_completion directly for stateless to be safe and avoid modifying RAGGenerator further.
    from llm.client import chat_completion
    from generation.rag_generator import SYSTEM_PROMPT
    
    evidence_text = ""
    for chunk in top_chunks:
        evidence_text += f"[CHUNK_ID: {chunk.chunk_id}] (Page: {chunk.page})\n{chunk.text}\n---\n"
        
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
    
    import uuid
    return {
        "draft_id": str(uuid.uuid4()),
        "document_id": "stateless",
        "generated_text": generated_text,
        "citations": [],
        "evidence_chunks": [c.model_dump() for c in top_chunks],
        "grounding_score": 1.0,
    }


@router.get("/{draft_id}", response_model=DraftResponse)
def get_draft(draft_id: str, db: Session = Depends(get_db)):
    """Retrieve a previously generated draft with its evidence."""
    draft = db.query(models.Draft).filter(models.Draft.draft_id == draft_id).first()
    if not draft:
        raise HTTPException(404, f"Draft {draft_id} not found.")

    # Reconstruct evidence chunks from stored chunk_ids
    from core.schemas import EvidenceChunk, Citation
    evidence = []
    for chunk_id in (draft.evidence_used or []):
        chunk = db.query(models.Chunk).filter(models.Chunk.chunk_id == chunk_id).first()
        if chunk:
            evidence.append(EvidenceChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                page=chunk.page,
                section=chunk.section,
                score=0.0,
            ))

    citations = [Citation(**c) for c in (draft.citations or [])]

    return DraftResponse(
        draft_id=draft.draft_id,
        document_id=draft.document_id,
        generated_text=draft.generated_text,
        citations=citations,
        evidence_chunks=evidence,
        grounding_score=draft.grounding_score or 0.0,
        created_at=draft.created_at,
    )


@router.get("/{draft_id}/evidence")
def get_draft_evidence(draft_id: str, db: Session = Depends(get_db)):
    """Retrieve the evidence passages used to generate a specific draft."""
    draft = db.query(models.Draft).filter(models.Draft.draft_id == draft_id).first()
    if not draft:
        raise HTTPException(404, f"Draft {draft_id} not found.")

    evidence = []
    for chunk_id in (draft.evidence_used or []):
        chunk = db.query(models.Chunk).filter(models.Chunk.chunk_id == chunk_id).first()
        if chunk:
            evidence.append({
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "page": chunk.page,
                "section": chunk.section,
            })

    return {
        "draft_id": draft_id,
        "grounding_score": draft.grounding_score,
        "evidence_count": len(evidence),
        "evidence": evidence,
    }


@router.get("")
def list_drafts(document_id: str = None, db: Session = Depends(get_db)):
    """List all drafts, optionally filtered by document_id."""
    query = db.query(models.Draft).order_by(models.Draft.created_at.desc())
    if document_id:
        query = query.filter(models.Draft.document_id == document_id)
    drafts = query.limit(50).all()
    return [
        {
            "draft_id": d.draft_id,
            "document_id": d.document_id,
            "grounding_score": d.grounding_score,
            "created_at": d.created_at,
        }
        for d in drafts
    ]
