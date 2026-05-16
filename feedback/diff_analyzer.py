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


class EditClassification(str, Enum):
    FIX           = "fix"           # Fact correction for this specific doc only
    RULE          = "rule"          # Generalizable style/structural rule
    CASE_SPECIFIC = "case_specific" # Ambiguous; store but do not inject into prompts


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
    inferred_rules:     List[str] = field(default_factory=list)
    case_specific_fixes: List[str] = field(default_factory=list)
    dominant_type:       str = "other"


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
# Generalizability classifier
# ──────────────────────────────────────────────────────────────────────────────

def _classify_generalizability(op: EditOperation) -> EditClassification:
    """
    Decide whether a single edit should become a universal RULE injected into
    all future prompts, a one-off FIX for this document's facts, or a
    CASE_SPECIFIC correction that is stored but not injected.

    Priority order:
    1. Any edit touching _FACT_SIGNALS (dates, amounts, case IDs) is a FIX —
       it corrects this document's facts, not a reusable style pattern.
    2. Structural/formatting changes (poor_structure) are always RULE.
    3. Hallucination deletions are always RULE (universally applicable).
    4. Tiny replacements (≤3 word delta) are CASE_SPECIFIC — too ambiguous.
    5. Everything else defaults to RULE.
    """
    if _FACT_SIGNALS.search(op.original) or _FACT_SIGNALS.search(op.replacement):
        return EditClassification.FIX

    if op.category == "poor_structure":
        return EditClassification.RULE

    if op.category == "hallucination":
        return EditClassification.RULE

    if op.edit_type == EditType.REPLACEMENT:
        if abs(len(op.original.split()) - len(op.replacement.split())) <= 3:
            return EditClassification.CASE_SPECIFIC

    return EditClassification.RULE


# ──────────────────────────────────────────────────────────────────────────────
# Rule inference from edit patterns
# ──────────────────────────────────────────────────────────────────────────────

def _infer_rules(
    operations: List[EditOperation],
) -> tuple[List[str], List[str]]:
    """
    Partition operations by generalizability, then extract rules.

    Returns:
        (universal_rules, case_specific_fixes)
        universal_rules    — injected into future prompts for ALL documents.
        case_specific_fixes — stored for record-keeping only; not injected.
    """
    rule_ops = [op for op in operations
                if _classify_generalizability(op) == EditClassification.RULE]
    fix_ops  = [op for op in operations
                if _classify_generalizability(op) == EditClassification.FIX]

    rules: List[str] = []
    fixes: List[str] = []

    # ── Universal rules (from RULE-classified operations only) ────────────────
    rule_cats = [op.category for op in rule_ops]

    if rule_cats.count("hallucination") >= 1:
        rules.append("Remove speculative language — only state facts directly supported by evidence.")

    if rule_cats.count("missing_fact") >= 1:
        rules.append("Ensure all key facts from the evidence (dates, amounts, parties) are included.")

    if rule_cats.count("poor_structure") >= 1:
        for op in rule_ops:
            if op.category == "poor_structure":
                if _STRUCTURE_SIGNALS.search(op.replacement) and not _STRUCTURE_SIGNALS.search(op.original):
                    rules.append("Use bullet-point lists for key facts, not dense paragraphs.")
                    break
                if not _STRUCTURE_SIGNALS.search(op.replacement) and _STRUCTURE_SIGNALS.search(op.original):
                    rules.append("Use concise prose for summaries, not fragmented bullet lists.")
                    break

    if rule_cats.count("incorrect_fact") >= 1:
        rules.append("Double-check all numerical values and dates against the evidence passages before including them.")

    if rule_cats.count("style") >= 2:
        deletions = [op for op in rule_ops if op.edit_type == EditType.DELETION]
        if len(deletions) > len(rule_ops) // 2:
            rules.append("Keep drafts concise — avoid redundant sentences that repeat evidence verbatim.")

    # ── Case-specific fixes (stored, never injected) ──────────────────────────
    for op in fix_ops:
        snippet = (op.original[:60] + "…") if len(op.original) > 60 else op.original
        rep     = (op.replacement[:60] + "…") if len(op.replacement) > 60 else op.replacement
        fixes.append(
            f"[{op.edit_type.value.upper()}] fact correction: '{snippet}' → '{rep}'"
        )

    return rules, fixes


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

        inferred_rules, case_specific_fixes = _infer_rules(operations)

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
            case_specific_fixes=case_specific_fixes,
            dominant_type=dominant_type,
        )

        logger.info(
            f"Diff analysis: similarity={similarity:.2f}, "
            f"ops={len(operations)}, dominant={dominant_type}, "
            f"rules={len(inferred_rules)}, fixes={len(case_specific_fixes)}"
        )
        return analysis
