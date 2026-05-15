"""
Stage 12 (continued) — Prompt Optimization Pipeline
Extracts high-quality edits to be used as few-shot examples
for future draft generation prompts.
"""
from __future__ import annotations

import logging
from typing import List

from sqlalchemy.orm import Session

from db.models import Chunk, Edit, FewShotExample

logger = logging.getLogger(__name__)

class PromptOptimizer:
    """
    Learns from captured edits by converting the best ones
    into few-shot prompt examples.
    """

    def __init__(self, db_session: Session):
        self.db = db_session

    def extract_few_shot_examples(self, limit: int = 5) -> int:
        """
        Identify high-quality edits and promote them to FewShotExamples.
        Returns the number of new examples created.
        """
        # Heuristic for "high quality": substantial edit but not a complete rewrite
        # For this prototype, we'll just promote the latest edits that have reviewer comments
        
        candidates = self.db.query(Edit).filter(
            Edit.reviewer_comment != None,
            Edit.feedback_type.in_(["missing_fact", "poor_structure"])
        ).order_by(Edit.created_at.desc()).limit(limit).all()
        
        added = 0
        for edit in candidates:
            # Check if it's already promoted
            existing = self.db.query(FewShotExample).filter_by(
                original_draft=edit.original_text,
                corrected_draft=edit.edited_text
            ).first()
            
            if existing:
                continue
                
            # Reconstruct evidence context
            evidence_texts = []
            if edit.retrieved_evidence:
                # Need to fetch the chunk texts from the DB
                chunks = self.db.query(Chunk).filter(Chunk.chunk_id.in_(edit.retrieved_evidence)).all()
                for c in chunks:
                    evidence_texts.append(f"[CHUNK_ID: {c.chunk_id}]\n{c.text}")
                    
            context = "\n---\n".join(evidence_texts)
            if not context:
                context = "No evidence recorded."
                
            example = FewShotExample(
                evidence_context=context,
                original_draft=edit.original_text,
                corrected_draft=edit.edited_text,
                feedback_type=edit.feedback_type,
                quality_score=1.0
            )
            self.db.add(example)
            added += 1
            
        self.db.commit()
        logger.info(f"Promoted {added} edits to few-shot examples.")
        return added

    def get_best_examples(self, feedback_type: str = None, limit: int = 3) -> List[dict]:
        """Retrieve the best examples to inject into the RAG prompt."""
        query = self.db.query(FewShotExample).order_by(FewShotExample.quality_score.desc())
        
        if feedback_type:
            query = query.filter(FewShotExample.feedback_type == feedback_type)
            
        examples = query.limit(limit).all()
        
        # Update use counts
        for ex in examples:
            ex.use_count += 1
        self.db.commit()
        
        return [
            {
                "evidence": ex.evidence_context,
                "query": "Generate a concise case fact summary.", # Default task
                "corrected_draft": ex.corrected_draft
            }
            for ex in examples
        ]
