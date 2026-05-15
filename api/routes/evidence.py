"""
API Routes: Evidence Retrieval
Allows reviewers to fetch the source evidence chunks associated with a draft.
"""
from __future__ import annotations

from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.schemas import EvidenceChunk
from db.models import Draft, Chunk
from db.session import get_db

router = APIRouter()

@router.get("/{draft_id}", response_model=List[EvidenceChunk])
async def get_evidence(
    draft_id: str,
    db: Session = Depends(get_db)
) -> List[EvidenceChunk]:
    """Retrieve the exact evidence chunks used to generate a specific draft."""
    draft = db.query(Draft).filter(Draft.draft_id == draft_id).first()
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
        
    chunk_ids = draft.evidence_used
    if not chunk_ids:
        return []
        
    db_chunks = db.query(Chunk).filter(Chunk.chunk_id.in_(chunk_ids)).all()
    
    evidence = []
    for c in db_chunks:
        evidence.append(EvidenceChunk(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            text=c.text,
            page=c.page,
            section=c.section,
            score=1.0, # Unknown at this point unless stored in DB
            retrieval_method="db_lookup"
        ))
        
    return evidence
