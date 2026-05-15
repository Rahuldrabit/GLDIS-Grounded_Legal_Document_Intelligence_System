"""
Prompt templates for grounded legal draft generation.
Supports dynamic injection of few-shot examples from the feedback loop.
"""
from __future__ import annotations

from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# System prompts
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a legal document analyst at Pearson Specter Litt.
Your task is to generate a grounded Case Fact Summary and Internal Memo based SOLELY on the provided evidence passages.

STRICT RULES:
1. ONLY state facts that are directly supported by the provided evidence passages.
2. Every factual claim MUST include an inline citation in the format [Source: chunk_id, p.X] where X is the page number.
3. If evidence is insufficient to make a determination, explicitly state: "Insufficient evidence to determine [topic]."
4. Do NOT infer, speculate, or add information not present in the evidence.
5. If information appears uncertain or partially legible, mark it with [UNCERTAIN].
6. Structure the output with clear headings and bullet points.

OUTPUT FORMAT:
---
## CASE FACT SUMMARY

### Parties Involved
- [List parties with citations]

### Key Facts
- [Numbered list of facts with citations]

### Key Dates & Deadlines
- [List dates with citations]

### Financial Details
- [List monetary amounts with citations]

### Obligations & Requirements
- [List obligations with citations]

---

## INTERNAL MEMO

**RE:** [Brief subject line]
**DATE:** [Current date]
**FROM:** AI Legal Assistant
**TO:** Reviewing Attorney

### Summary
[1-2 paragraph executive summary grounded in evidence]

### Key Issues
[Numbered list of issues identified from the documents]

### Recommended Actions
[Bulleted list of next steps based on document content]

### Evidence Gaps
[List any areas where the provided documents are insufficient]
---"""


# ──────────────────────────────────────────────────────────────────────────────
# Evidence formatting
# ──────────────────────────────────────────────────────────────────────────────

def format_evidence_block(evidence_chunks: list) -> str:
    """Format evidence chunks into a structured block for the prompt."""
    if not evidence_chunks:
        return "No evidence passages available."

    parts = []
    for i, chunk in enumerate(evidence_chunks, 1):
        page_info = f", Page {chunk.page}" if chunk.page else ""
        section_info = f", Section: {chunk.section}" if chunk.section else ""
        header = f"[Evidence {i} | ID: {chunk.chunk_id}{page_info}{section_info} | Score: {chunk.score:.3f}]"
        parts.append(f"{header}\n{chunk.text}")

    return "\n\n---\n\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Few-shot example formatting
# ──────────────────────────────────────────────────────────────────────────────

def format_few_shot_examples(examples: list) -> str:
    """Format few-shot examples from feedback loop into the prompt."""
    if not examples:
        return ""

    parts = ["## EXAMPLES OF HIGH-QUALITY DRAFTS (learn from these patterns):\n"]
    for i, ex in enumerate(examples, 1):
        parts.append(f"### Example {i}")
        parts.append(f"**Evidence context:**\n{ex.evidence_context[:500]}...")
        parts.append(f"**Good draft:**\n{ex.corrected_draft[:800]}...")
        parts.append("")

    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Style guide from feedback
# ──────────────────────────────────────────────────────────────────────────────

def format_style_guide(rules: List[str]) -> str:
    """Format aggregated style preferences from operator edits."""
    if not rules:
        return ""

    parts = ["\n## STYLE GUIDELINES (learned from operator corrections):\n"]
    for rule in rules:
        parts.append(f"- {rule}")

    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Full prompt assembly
# ──────────────────────────────────────────────────────────────────────────────

def build_generation_prompt(
    query: str,
    evidence_chunks: list,
    few_shot_examples: Optional[list] = None,
    style_rules: Optional[List[str]] = None,
    structured_fields: Optional[dict] = None,
) -> list:
    """
    Build the complete message list for the LLM API call.
    Returns a list of dicts with 'role' and 'content' keys.
    """
    # System message
    system_content = SYSTEM_PROMPT

    # Inject style guide if available
    if style_rules:
        system_content += "\n" + format_style_guide(style_rules)

    # User message
    user_parts = []

    # Add few-shot examples if available
    if few_shot_examples:
        user_parts.append(format_few_shot_examples(few_shot_examples))

    # Add structured extraction data if available
    if structured_fields:
        user_parts.append("## PRE-EXTRACTED STRUCTURED DATA:")
        for field, value in structured_fields.items():
            if value:
                if isinstance(value, list):
                    user_parts.append(f"- **{field}**: {', '.join(str(v) for v in value)}")
                else:
                    user_parts.append(f"- **{field}**: {value}")
        user_parts.append("")

    # Add evidence
    user_parts.append("## EVIDENCE PASSAGES:\n")
    user_parts.append(format_evidence_block(evidence_chunks))

    # Add query/instruction
    user_parts.append(f"\n\n## TASK:\n{query}")

    user_parts.append(
        "\n\nRemember: Ground every fact in the evidence above. "
        "Use [Source: chunk_id, p.X] citations. "
        "If evidence is insufficient, say so explicitly."
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "\n".join(user_parts)},
    ]
