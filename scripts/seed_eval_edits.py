"""
Seed the database with two synthetic edit records so that
evaluate_improvement_trend() has >= 2 data points.

The diff summaries are formatted to match exactly what
ImprovementLoop._aggregate_style_rules() and get_improvement_trend() parse.

Safe to run multiple times — skips if >= 2 edits already exist.

Usage:
    python scripts/seed_eval_edits.py
"""
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from db.session import create_tables, get_session_factory
from db import models

create_tables()
SessionLocal = get_session_factory()
db = SessionLocal()

try:
    existing_count = db.query(models.Edit).count()
    if existing_count >= 2:
        print(f"Already have {existing_count} edits; skipping seed.")
        sys.exit(0)

    drafts = (
        db.query(models.Draft)
        .order_by(models.Draft.created_at.asc())
        .limit(2)
        .all()
    )

    if not drafts:
        print(
            "ERROR: No drafts in the database. "
            "Generate at least one draft via POST /api/drafts/generate first."
        )
        sys.exit(1)

    draft_a = drafts[0]
    draft_b = drafts[1] if len(drafts) > 1 else drafts[0]

    # ── Edit A: hallucination removal (earlier in time = lower similarity baseline) ──
    ORIGINAL_A = (
        "The lease term shall commence on February 1, 2024. "
        "Monthly rent is $12,500 due on the first of each month. "
        "It is inferred that both parties likely agreed to standard terms."
    )
    EDITED_A = (
        "The lease term shall commence on February 1, 2024. "
        "Monthly rent is $12,500 due on the first of each month."
    )
    DIFF_A = (
        "Similarity: 84.50%\n"
        "Edit distance: 72\n"
        "Dominant type: hallucination\n"
        "Operations: 1\n"
        "\n"
        "Inferred rules:\n"
        "  - Remove speculative language — only state facts directly supported by evidence.\n"
        "\n"
        "Edit operations (top 10):\n"
        "  [DELETION] (hallucination) "
        "'It is inferred that both parties likely agreed to standard terms.' → ''"
    )

    # ── Edit B: structural improvement (later in time = higher similarity) ──
    ORIGINAL_B = (
        "Parties: Acme Corp and Beta Ltd. "
        "Key obligation: payment of USD 45,000 per month. "
        "Termination clause applies after 30 days written notice."
    )
    EDITED_B = (
        "**Parties:** Acme Corp and Beta Ltd.\n"
        "- Key obligation: payment of USD 45,000 per month.\n"
        "- Termination clause: 30 days written notice required."
    )
    DIFF_B = (
        "Similarity: 91.20%\n"
        "Edit distance: 48\n"
        "Dominant type: poor_structure\n"
        "Operations: 2\n"
        "\n"
        "Inferred rules:\n"
        "  - Use bullet-point lists for key facts, not dense paragraphs.\n"
        "\n"
        "Edit operations (top 10):\n"
        "  [REPLACEMENT] (poor_structure) 'Parties:' → '**Parties:**'\n"
        "  [REPLACEMENT] (poor_structure) 'Key obligation:' → '- Key obligation:'"
    )

    now = datetime.utcnow()

    edit_a = models.Edit(
        edit_id=str(uuid.uuid4()),
        draft_id=draft_a.draft_id,
        original_text=ORIGINAL_A,
        edited_text=EDITED_A,
        edit_diff=DIFF_A,
        feedback_type="hallucination",
        reviewer_comment="Removed speculative inference statement",
        reviewer_id="eval_seed",
        retrieved_evidence=[],
        created_at=now - timedelta(hours=2),
    )

    edit_b = models.Edit(
        edit_id=str(uuid.uuid4()),
        draft_id=draft_b.draft_id,
        original_text=ORIGINAL_B,
        edited_text=EDITED_B,
        edit_diff=DIFF_B,
        feedback_type="poor_structure",
        reviewer_comment="Reformatted dense paragraph to bullet points",
        reviewer_id="eval_seed",
        retrieved_evidence=[],
        created_at=now - timedelta(hours=1),
    )

    db.add(edit_a)
    db.add(edit_b)
    db.commit()

    print(f"Seeded 2 eval edits:")
    print(f"  Edit A (hallucination removal): {edit_a.edit_id}")
    print(f"  Edit B (structure improvement):  {edit_b.edit_id}")
    print(f"Total edits in DB: {db.query(models.Edit).count()}")

finally:
    db.close()
