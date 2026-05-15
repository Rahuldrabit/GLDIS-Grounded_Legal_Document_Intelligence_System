"""Routes: Feedback submission and improvement history."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.schemas import FeedbackRequest, FeedbackRecord
from db import models
from db.session import get_db
from feedback.feedback_engine import FeedbackEngine
from feedback.improvement_loop import ImprovementLoop

router = APIRouter(prefix="/api/feedback", tags=["feedback"])

_engine = FeedbackEngine()
_loop = ImprovementLoop()


@router.post("", response_model=FeedbackRecord)
def submit_feedback(request: FeedbackRequest, db: Session = Depends(get_db)):
    """
    Submit an operator edit for a draft.
    The system will analyze the edit, extract patterns,
    and use them to improve future drafts.
    """
    # Validate draft exists
    draft = db.query(models.Draft).filter(models.Draft.draft_id == request.draft_id).first()
    if not draft:
        raise HTTPException(404, f"Draft {request.draft_id} not found.")

    # Reconstruct evidence for context
    from core.schemas import EvidenceChunk
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

    record = _engine.process_feedback(request, db, retrieved_evidence=evidence)
    return record


@router.get("/history")
def get_feedback_history(limit: int = 20, db: Session = Depends(get_db)):
    """View recent feedback submissions and their analysis."""
    edits = _engine.get_recent_edits(db, limit=limit)
    return [
        {
            "edit_id": e.edit_id,
            "draft_id": e.draft_id,
            "feedback_type": e.feedback_type,
            "reviewer_id": e.reviewer_id,
            "reviewer_comment": e.reviewer_comment,
            "diff_summary": e.edit_diff,
            "created_at": e.created_at,
        }
        for e in edits
    ]


@router.get("/improvements")
def get_improvements(db: Session = Depends(get_db)):
    """
    View the current state of the improvement loop:
    - Learned style rules
    - Available few-shot examples
    - Edit statistics and trends
    """
    stats = _engine.get_improvement_stats(db)
    trend = _loop.get_improvement_trend(db)
    ctx = _loop.get_generation_context(db)

    return {
        "statistics": stats,
        "active_style_rules": ctx["style_rules"],
        "improvement_trend": trend,
        "context_summary": ctx["improvement_summary"],
    }


@router.get("/examples")
def get_few_shot_examples(limit: int = 10, db: Session = Depends(get_db)):
    """View the current pool of few-shot examples used in generation."""
    examples = (
        db.query(models.FewShotExample)
        .order_by(
            models.FewShotExample.quality_score.desc(),
            models.FewShotExample.created_at.desc(),
        )
        .limit(limit)
        .all()
    )
    return [
        {
            "id": ex.id,
            "feedback_type": ex.feedback_type,
            "quality_score": ex.quality_score,
            "use_count": ex.use_count,
            "created_at": ex.created_at,
            "corrected_draft_preview": ex.corrected_draft[:300] + "...",
        }
        for ex in examples
    ]
