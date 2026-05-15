"""
Stage 12 — Edit Capture and Diffing
Captures reviewer edits, computes word-level diffs, and stores them
in the database for future prompt learning.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from core.schemas import EvidenceChunk, FeedbackRecord, FeedbackRequest
from db.models import Draft, Edit

logger = logging.getLogger(__name__)

# Optional robust diffing using difflib
try:
    import difflib
    DIFFLIB_AVAILABLE = True
except ImportError:
    DIFFLIB_AVAILABLE = False


def _compute_diff(original: str, edited: str) -> str:
    """Compute a human-readable word-level diff."""
    if not DIFFLIB_AVAILABLE:
        return "Diff not available (difflib missing)."
    
    # Simple word-based diff
    orig_words = original.split()
    edit_words = edited.split()
    
    diff = difflib.ndiff(orig_words, edit_words)
    diff_text = []
    
    for word in diff:
        if word.startswith("- "):
            diff_text.append(f"[-{word[2:]}-]")
        elif word.startswith("+ "):
            diff_text.append(f"[+{word[2:]}+]")
        elif word.startswith("  "):
            diff_text.append(word[2:])
            
    return " ".join(diff_text)


class FeedbackCaptureSystem:
    """Captures edits and stores them for continuous improvement."""

    def __init__(self, db_session: Session):
        self.db = db_session

    def process_feedback(self, request: FeedbackRequest) -> FeedbackRecord:
        """
        Process incoming feedback, compute diffs, and store in DB.
        """
        # Get original draft to link retrieved evidence
        draft_record = self.db.query(Draft).filter(Draft.draft_id == request.draft_id).first()
        
        evidence_used = []
        if draft_record and draft_record.evidence_used:
            # Reconstruct dummy EvidenceChunks or query them if needed
            # For simplicity, we just store the JSON list directly in the Edit model
            pass
            
        diff_text = _compute_diff(request.original_draft, request.edited_draft)
        
        # Create DB record
        edit_db = Edit(
            draft_id=request.draft_id,
            original_text=request.original_draft,
            edited_text=request.edited_draft,
            edit_diff=diff_text,
            feedback_type=request.feedback_type.value,
            reviewer_comment=request.reviewer_comment,
            reviewer_id=request.reviewer_id,
            retrieved_evidence=draft_record.evidence_used if draft_record else []
        )
        
        self.db.add(edit_db)
        self.db.commit()
        self.db.refresh(edit_db)
        
        logger.info(f"Feedback captured for draft {request.draft_id} from {request.reviewer_id}")
        
        return FeedbackRecord(
            feedback_id=edit_db.edit_id,
            draft_id=edit_db.draft_id,
            original_draft=edit_db.original_text,
            edited_draft=edit_db.edited_text,
            edit_diff=edit_db.edit_diff,
            feedback_type=request.feedback_type,
            reviewer_comment=edit_db.reviewer_comment,
            reviewer_id=edit_db.reviewer_id,
            created_at=edit_db.created_at
        )

    def get_recent_edits(self, limit: int = 10) -> List[dict]:
        """Fetch recently captured edits for evaluation/learning."""
        edits = self.db.query(Edit).order_by(Edit.created_at.desc()).limit(limit).all()
        return [
            {
                "edit_id": e.edit_id,
                "draft_id": e.draft_id,
                "feedback_type": e.feedback_type,
                "diff": e.edit_diff
            }
            for e in edits
        ]
