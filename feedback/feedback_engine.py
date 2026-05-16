"""
Feedback Phase — Edit Capture Engine
Receives operator edits, runs diff analysis, stores results, and
extracts few-shot examples for prompt improvement.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from core.schemas import (
    EvidenceChunk, FeedbackRecord, FeedbackRequest, FeedbackType
)
from db import models
from feedback.diff_analyzer import DiffAnalyzer, EditAnalysis

logger = logging.getLogger(__name__)

_analyzer = DiffAnalyzer()

# Minimum quality bar for promoting an edit to a few-shot example
_MIN_QUALITY_SCORE = 0.65
# Minimum similarity — very high edits = trivial, very low = too different
_MIN_SIMILARITY = 0.3
_MAX_SIMILARITY = 0.97


def _quality_score(analysis: EditAnalysis, feedback_type: FeedbackType) -> float:
    """
    Heuristic quality score [0, 1] for deciding whether to promote
    an edit to a few-shot example.

    High score = well-targeted edit that will improve future generation.
    """
    score = 0.5  # Base

    # Reward meaningful but not wholesale rewrites
    if _MIN_SIMILARITY < analysis.similarity < _MAX_SIMILARITY:
        score += 0.2

    # Reward specific, actionable rule derivation
    score += min(len(analysis.inferred_rules) * 0.1, 0.2)

    # Reward important feedback types
    if feedback_type in (FeedbackType.HALLUCINATION, FeedbackType.MISSING_FACT):
        score += 0.1

    return min(round(score, 3), 1.0)


class FeedbackEngine:
    """
    Captures operator edits, stores structured analysis, and
    manages the few-shot example pool.
    """

    def process_feedback(
        self,
        request: FeedbackRequest,
        db: Session,
        retrieved_evidence: Optional[List[EvidenceChunk]] = None,
    ) -> FeedbackRecord:
        """
        Process a single operator edit submission:
        1. Compute structured diff
        2. Classify edit types
        3. Infer style rules
        4. Persist to DB
        5. Promote to few-shot example if high quality
        """
        analysis = _analyzer.analyze(request.original_draft, request.edited_draft)

        # Serialize diff for storage
        diff_summary = self._format_diff_summary(analysis)

        feedback_id = str(uuid.uuid4())
        evidence_list = [e.model_dump() for e in (retrieved_evidence or [])]

        # Persist Edit record
        edit_record = models.Edit(
            edit_id=feedback_id,
            draft_id=request.draft_id,
            original_text=request.original_draft,
            edited_text=request.edited_draft,
            edit_diff=diff_summary,
            feedback_type=request.feedback_type.value,
            reviewer_comment=request.reviewer_comment,
            reviewer_id=request.reviewer_id or "anonymous",
            retrieved_evidence=evidence_list,
        )
        db.add(edit_record)

        # Determine quality and optionally promote to few-shot example
        qs = _quality_score(analysis, request.feedback_type)
        if qs >= _MIN_QUALITY_SCORE:
            evidence_context = "\n\n".join(
                e.text for e in (retrieved_evidence or [])[:5]
            )
            example = models.FewShotExample(
                id=str(uuid.uuid4()),
                evidence_context=evidence_context,
                original_draft=request.original_draft[:3000],
                corrected_draft=request.edited_draft[:3000],
                feedback_type=request.feedback_type.value,
                quality_score=qs,
            )
            db.add(example)
            logger.info(
                f"Promoted edit {feedback_id} to few-shot example "
                f"(quality={qs:.2f}, type={request.feedback_type.value})"
            )

        db.commit()

        record = FeedbackRecord(
            feedback_id=feedback_id,
            draft_id=request.draft_id,
            original_draft=request.original_draft,
            edited_draft=request.edited_draft,
            edit_diff=diff_summary,
            feedback_type=request.feedback_type,
            reviewer_comment=request.reviewer_comment,
            reviewer_id=request.reviewer_id or "anonymous",
            retrieved_evidence=retrieved_evidence or [],
        )

        logger.info(
            f"Feedback {feedback_id} processed | "
            f"similarity={analysis.similarity:.2f} | "
            f"ops={len(analysis.operations)} | "
            f"quality={qs:.2f}"
        )
        return record

    def _format_diff_summary(self, analysis: EditAnalysis) -> str:
        """Format a human-readable diff summary for storage."""
        lines = [
            f"Similarity: {analysis.similarity:.2%}",
            f"Edit distance: {analysis.edit_distance}",
            f"Dominant type: {analysis.dominant_type}",
            f"Operations: {len(analysis.operations)}",
            "",
            "Inferred rules:",
        ]
        for rule in analysis.inferred_rules:
            lines.append(f"  - {rule}")

        if analysis.case_specific_fixes:
            lines.append("\nCase-specific fixes (not injected into prompts):")
            for fix in analysis.case_specific_fixes:
                lines.append(f"  [FIX] {fix}")

        lines.append("\nEdit operations (top 10):")
        for op in analysis.operations[:10]:
            lines.append(
                f"  [{op.edit_type.value.upper()}] ({op.category}) "
                f"'{op.original[:60]}' → '{op.replacement[:60]}'"
            )

        return "\n".join(lines)

    def get_recent_edits(self, db: Session, limit: int = 20) -> List[models.Edit]:
        return (
            db.query(models.Edit)
            .order_by(models.Edit.created_at.desc())
            .limit(limit)
            .all()
        )

    def get_improvement_stats(self, db: Session) -> dict:
        """Return aggregate statistics about feedback-driven improvements."""
        total_edits = db.query(models.Edit).count()
        total_examples = db.query(models.FewShotExample).count()

        # Count by feedback type
        from sqlalchemy import func
        type_counts = (
            db.query(models.Edit.feedback_type, func.count(models.Edit.edit_id))
            .group_by(models.Edit.feedback_type)
            .all()
        )

        # Average quality of promoted examples
        avg_quality = (
            db.query(func.avg(models.FewShotExample.quality_score))
            .scalar() or 0.0
        )

        return {
            "total_edits": total_edits,
            "few_shot_examples": total_examples,
            "edit_type_breakdown": {t: c for t, c in type_counts},
            "avg_example_quality": round(float(avg_quality), 3),
        }
