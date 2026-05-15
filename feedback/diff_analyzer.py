"""
Feedback Phase — Structured Diff Analyzer
Computes semantic and character-level diffs between original and edited drafts,
classifying changes into meaningful edit categories.
"""
from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple

logger = logging.getLogger(__name__)


class EditType(str, Enum):
    INSERTION   = "insertion"    # Operator added content
    DELETION    = "deletion"     # Operator removed content
    REPLACEMENT = "replacement"  # Operator changed content


@dataclass
class EditOperation:
    edit_type:    EditType
    original:     str
    replacement:  str
    category:     str   # missing_fact | hallucination | poor_structure | incorrect_fact | style


@dataclass
class EditAnalysis:
    original_draft:  str
    edited_draft:    str
    edit_distance:   int                         # Levenshtein approximation
    similarity:      float                       # 0–1, higher = more similar
    operations:      List[EditOperation] = field(default_factory=list)
    inferred_rules:  List[str]          = field(default_factory=list)
    dominant_type:   str = "other"


# ──────────────────────────────────────────────────────────────────────────────
# Edit category classifiers
# ──────────────────────────────────────────────────────────────────────────────

_HALLUCINATION_SIGNALS = re.compile(
    r"\b(inferr|assum|believe|likely|probably|appear|seem|suggest|speculate"
    r"|uncertain|unclear|may have|might have|could have)\b",
    re.IGNORECASE,
)

_STRUCTURE_SIGNALS = re.compile(
    r"^(#{1,3} |[-*•] |\d+\. )",  # Markdown headings / list markers
    re.MULTILINE,
)

_FACT_SIGNALS = re.compile(
    r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\$[\d,]+|USD|EUR|Case\s*No|Article\s+\d|Section\s+\d)\b",
    re.IGNORECASE,
)


def _classify_edit(original: str, replacement: str, edit_type: EditType) -> str:
    """Classify a single edit operation into a category."""
    if edit_type == EditType.DELETION:
        if _HALLUCINATION_SIGNALS.search(original):
            return "hallucination"
        if len(original.split()) < 5:
            return "style"
        return "hallucination"

    if edit_type == EditType.INSERTION:
        if _FACT_SIGNALS.search(replacement):
            return "missing_fact"
        if len(replacement.split()) > 10:
            return "missing_fact"
        return "style"

    if edit_type == EditType.REPLACEMENT:
        if _FACT_SIGNALS.search(original) or _FACT_SIGNALS.search(replacement):
            return "incorrect_fact"
        if _STRUCTURE_SIGNALS.search(original) or _STRUCTURE_SIGNALS.search(replacement):
            return "poor_structure"
        if len(original.split()) > len(replacement.split()) * 1.5:
            return "poor_structure"
        return "style"

    return "other"


# ──────────────────────────────────────────────────────────────────────────────
# Rule inference from edit patterns
# ──────────────────────────────────────────────────────────────────────────────

def _infer_rules(operations: List[EditOperation]) -> List[str]:
    """
    Extract reusable style/content rules from edit patterns.
    These are injected into future generation prompts.
    """
    rules = []
    cats = [op.category for op in operations]

    if cats.count("hallucination") >= 1:
        rules.append("Remove speculative language — only state facts directly supported by evidence.")

    if cats.count("missing_fact") >= 1:
        rules.append("Ensure all key facts from the evidence (dates, amounts, parties) are included.")

    if cats.count("poor_structure") >= 1:
        # Check if operator added or removed bullet points
        for op in operations:
            if op.category == "poor_structure":
                if _STRUCTURE_SIGNALS.search(op.replacement) and not _STRUCTURE_SIGNALS.search(op.original):
                    rules.append("Use bullet-point lists for key facts, not dense paragraphs.")
                    break
                if not _STRUCTURE_SIGNALS.search(op.replacement) and _STRUCTURE_SIGNALS.search(op.original):
                    rules.append("Use concise prose for summaries, not fragmented bullet lists.")
                    break

    if cats.count("incorrect_fact") >= 1:
        rules.append("Double-check all numerical values and dates against the evidence passages before including them.")

    if cats.count("style") >= 2:
        # Check for consistent length reduction
        deletions = [op for op in operations if op.edit_type == EditType.DELETION]
        if len(deletions) > len(operations) // 2:
            rules.append("Keep drafts concise — avoid redundant sentences that repeat evidence verbatim.")

    return rules


# ──────────────────────────────────────────────────────────────────────────────
# Main analyzer
# ──────────────────────────────────────────────────────────────────────────────

class DiffAnalyzer:
    """
    Computes structured diffs between original and operator-edited drafts.
    Produces actionable EditAnalysis objects for the improvement loop.
    """

    def analyze(self, original: str, edited: str) -> EditAnalysis:
        """
        Full diff analysis pipeline:
        1. Compute sequence matcher diff
        2. Classify each edit operation
        3. Infer reusable rules
        4. Determine dominant edit type
        """
        matcher = difflib.SequenceMatcher(None, original, edited, autojunk=False)
        similarity = matcher.ratio()
        edit_distance = int((1 - similarity) * max(len(original), len(edited)))

        operations: List[EditOperation] = []

        # Word-level diff for more semantic granularity
        orig_words = original.split()
        edit_words = edited.split()
        word_matcher = difflib.SequenceMatcher(None, orig_words, edit_words, autojunk=False)

        for tag, i1, i2, j1, j2 in word_matcher.get_opcodes():
            if tag == "equal":
                continue

            orig_span = " ".join(orig_words[i1:i2])
            edit_span = " ".join(edit_words[j1:j2])

            if tag == "insert":
                edit_type = EditType.INSERTION
                category = _classify_edit("", edit_span, edit_type)
            elif tag == "delete":
                edit_type = EditType.DELETION
                category = _classify_edit(orig_span, "", edit_type)
            else:  # replace
                edit_type = EditType.REPLACEMENT
                category = _classify_edit(orig_span, edit_span, edit_type)

            # Skip trivial whitespace changes
            if not orig_span.strip() and not edit_span.strip():
                continue

            operations.append(EditOperation(
                edit_type=edit_type,
                original=orig_span,
                replacement=edit_span,
                category=category,
            ))

        inferred_rules = _infer_rules(operations)

        # Determine dominant category
        cats = [op.category for op in operations]
        dominant_type = max(set(cats), key=cats.count) if cats else "other"

        analysis = EditAnalysis(
            original_draft=original,
            edited_draft=edited,
            edit_distance=edit_distance,
            similarity=round(similarity, 4),
            operations=operations,
            inferred_rules=inferred_rules,
            dominant_type=dominant_type,
        )

        logger.info(
            f"Diff analysis: similarity={similarity:.2f}, "
            f"ops={len(operations)}, dominant={dominant_type}, "
            f"rules={len(inferred_rules)}"
        )
        return analysis
