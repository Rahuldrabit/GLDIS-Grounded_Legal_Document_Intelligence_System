"""
Feedback Phase — Improvement Loop
Assembles learned few-shot examples, inferred style rules, and
aggregated preferences to improve future draft generation prompts.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import List, Optional

from sqlalchemy.orm import Session

from db import models

logger = logging.getLogger(__name__)

# Maximum few-shot examples to inject per generation
_MAX_FEW_SHOT = 3
# Maximum style rules to inject
_MAX_RULES = 5


class ImprovementLoop:
    """
    Manages the feedback→improvement cycle.
    
    Before generating a new draft:
    1. Retrieves high-quality few-shot examples from prior edits
    2. Aggregates style rules inferred from edit patterns
    3. Returns both for prompt injection
    
    This creates measurable improvement: each operator correction
    feeds directly into the next generation's prompt.
    """

    def get_generation_context(
        self,
        db: Session,
        feedback_type_filter: Optional[str] = None,
    ) -> dict:
        """
        Returns the improvement context to inject into generation:
        - few_shot_examples: List[FewShotExample] ORM objects
        - style_rules: List[str] aggregated from all edits
        - improvement_summary: dict for logging/display
        """
        few_shot = self._get_best_examples(db, feedback_type_filter)
        style_rules = self._aggregate_style_rules(db)

        context = {
            "few_shot_examples": few_shot,
            "style_rules": style_rules[:_MAX_RULES],
            "improvement_summary": {
                "examples_injected": len(few_shot),
                "rules_applied": len(style_rules[:_MAX_RULES]),
                "total_examples_available": db.query(models.FewShotExample).count(),
                "total_edits_learned_from": db.query(models.Edit).count(),
            },
        }

        logger.info(
            f"Improvement context: {len(few_shot)} examples, "
            f"{len(style_rules)} style rules"
        )
        return context

    def _get_best_examples(
        self,
        db: Session,
        feedback_type_filter: Optional[str] = None,
    ) -> List[models.FewShotExample]:
        """
        Retrieve the best few-shot examples, ordered by:
        1. Quality score (descending)
        2. Recency (more recent first)
        3. Diversity of feedback type
        """
        query = db.query(models.FewShotExample)

        if feedback_type_filter:
            query = query.filter(
                models.FewShotExample.feedback_type == feedback_type_filter
            )

        # Get more than we need, then select for diversity
        candidates = (
            query
            .order_by(
                models.FewShotExample.quality_score.desc(),
                models.FewShotExample.created_at.desc(),
            )
            .limit(_MAX_FEW_SHOT * 4)
            .all()
        )

        # Increment use_count on selected examples
        selected = self._select_diverse(candidates, _MAX_FEW_SHOT)
        for ex in selected:
            ex.use_count = (ex.use_count or 0) + 1
        if selected:
            db.commit()

        return selected

    def _select_diverse(
        self,
        candidates: List[models.FewShotExample],
        n: int,
    ) -> List[models.FewShotExample]:
        """Select n examples with maximum type diversity."""
        if len(candidates) <= n:
            return candidates

        # Group by feedback type
        by_type: dict = {}
        for ex in candidates:
            by_type.setdefault(ex.feedback_type, []).append(ex)

        # Round-robin selection across types
        selected = []
        types = list(by_type.keys())
        idx = 0
        while len(selected) < n and any(by_type[t] for t in types):
            t = types[idx % len(types)]
            if by_type[t]:
                selected.append(by_type[t].pop(0))
            idx += 1

        return selected

    def _aggregate_style_rules(self, db: Session) -> List[str]:
        """
        Parse all stored edit diffs and extract the most common style rules.
        Returns rules sorted by frequency.
        """
        edits = db.query(models.Edit).filter(
            models.Edit.edit_diff.isnot(None)
        ).all()

        rule_counter: Counter = Counter()
        for edit in edits:
            if not edit.edit_diff:
                continue
            # Parse rules section from stored diff summary
            lines = edit.edit_diff.splitlines()
            in_rules = False
            for line in lines:
                if line.strip() == "Inferred rules:":
                    in_rules = True
                    continue
                if in_rules and line.startswith("  - "):
                    rule = line[4:].strip()
                    if rule:
                        rule_counter[rule] += 1
                elif in_rules and not line.startswith("  "):
                    in_rules = False

        # Return rules that appeared more than once, or all if few edits
        threshold = 1
        rules = [
            rule for rule, count in rule_counter.most_common(_MAX_RULES * 2)
            if count >= threshold
        ]

        return rules[:_MAX_RULES]

    def get_improvement_trend(self, db: Session) -> List[dict]:
        """
        Return chronological list of edit similarity scores
        to show whether drafts are improving over time.
        """
        edits = (
            db.query(models.Edit)
            .order_by(models.Edit.created_at.asc())
            .all()
        )

        trend = []
        for edit in edits:
            # Parse similarity from diff summary
            sim = None
            if edit.edit_diff:
                for line in edit.edit_diff.splitlines():
                    if line.startswith("Similarity:"):
                        try:
                            sim = float(line.split(":")[1].strip().rstrip("%")) / 100
                        except Exception:
                            pass
                        break
            trend.append({
                "edit_id": edit.edit_id,
                "draft_id": edit.draft_id,
                "feedback_type": edit.feedback_type,
                "similarity": sim,
                "created_at": edit.created_at.isoformat() if edit.created_at else None,
            })

        return trend
