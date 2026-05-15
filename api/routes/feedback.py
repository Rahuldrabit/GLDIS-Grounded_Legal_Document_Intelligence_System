"""
API Routes: Feedback & Learning Loop
Receives reviewer edits, stores them, and occasionally triggers prompt optimization.
"""
from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.schemas import FeedbackRequest, FeedbackRecord
from db.session import get_db
from feedback.edit_capture import FeedbackCaptureSystem
from feedback.prompt_optimizer import PromptOptimizer

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/", response_model=FeedbackRecord)
async def submit_feedback(
    request: FeedbackRequest,
    db: Session = Depends(get_db)
) -> FeedbackRecord:
    """
    Submit reviewer edits for a generated draft.
    The system captures the diff and learns from it.
    """
    capture_sys = FeedbackCaptureSystem(db)
    
    try:
        record = capture_sys.process_feedback(request)
        
        # Trigger learning loop occasionally (or every time for prototype)
        optimizer = PromptOptimizer(db)
        optimizer.extract_few_shot_examples(limit=5)
        
        return record
    except Exception as exc:
        logger.error(f"Feedback processing failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
