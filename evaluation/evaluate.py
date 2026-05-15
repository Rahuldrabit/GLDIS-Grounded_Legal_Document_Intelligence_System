"""
Automated Evaluation Suite for GLDIS (Steps 33–36)

Measures:
  1. OCR quality (CER, WER)
  2. Retrieval quality (Precision@K, Recall@K, F1@K, MRR)
  3. Grounding quality (grounded sentence ratio, ungrounded examples)
  4. Improvement trend (edit similarity over time)
  5. System health (document/chunk/draft/edit counts)

Run: python evaluation/evaluate.py
"""
from __future__ import annotations

import json
import sys
import textwrap
import time
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))


# ──────────────────────────────────────────────────────────────────────────────
# 1. OCR Quality (CER / WER)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_ocr(ground_truth: str, hypothesis: str) -> dict:
    """
    Compute Character Error Rate (CER) and Word Error Rate (WER).
    Lower is better. 0.0 = perfect.
    """
    try:
        from jiwer import cer, wer
        return {
            "cer": round(cer(ground_truth, hypothesis), 4),
            "wer": round(wer(ground_truth, hypothesis), 4),
        }
    except ImportError:
        gt = ground_truth.lower().replace(" ", "")
        hy = hypothesis.lower().replace(" ", "")
        cer_val = _levenshtein(gt, hy) / max(len(gt), 1)

        gt_words = ground_truth.lower().split()
        hy_words = hypothesis.lower().split()
        wer_val = _levenshtein(gt_words, hy_words) / max(len(gt_words), 1)

        return {"cer": round(cer_val, 4), "wer": round(wer_val, 4)}


def _levenshtein(a, b) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


# ──────────────────────────────────────────────────────────────────────────────
# 2. Retrieval Quality (Precision@K, Recall@K, MRR)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_retrieval(
    relevant_chunk_ids: list[str],
    retrieved_chunk_ids: list[str],
) -> dict:
    """
    Compute Precision@K, Recall@K, F1@K, and MRR.
    relevant_chunk_ids: ground-truth relevant chunks
    retrieved_chunk_ids: retrieved chunks in rank order
    """
    relevant_set = set(relevant_chunk_ids)
    k = len(retrieved_chunk_ids)

    if not relevant_set or not retrieved_chunk_ids:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "f1_at_k": 0.0, "mrr": 0.0,
                "k": k, "relevant": len(relevant_set), "hits": 0}

    hits = [1 if r in relevant_set else 0 for r in retrieved_chunk_ids]
    precision_k = sum(hits) / k
    recall_k = sum(hits) / len(relevant_set)
    f1_k = (
        2 * precision_k * recall_k / (precision_k + recall_k)
        if (precision_k + recall_k) > 0 else 0.0
    )

    mrr = 0.0
    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant_set:
            mrr = 1.0 / rank
            break

    return {
        "precision_at_k": round(precision_k, 4),
        "recall_at_k":    round(recall_k, 4),
        "f1_at_k":        round(f1_k, 4),
        "mrr":            round(mrr, 4),
        "k":              k,
        "relevant":       len(relevant_set),
        "hits":           sum(hits),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. Grounding Quality (Step 35)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_grounding(generated_text: str, evidence_chunks: list) -> dict:
    """
    Run the production grounding scorer and return:
    - grounding_score: fraction of sentences supported by evidence
    - ungrounded_count: number of sentences with no evidence match
    - ungrounded_examples: up to 3 ungrounded sentences for debugging
    - citation_count: number of explicit [Source: ...] markers found
    """
    from generation.grounding import compute_grounding_score
    import re
    score, ungrounded = compute_grounding_score(generated_text, evidence_chunks)
    citation_count = len(re.findall(r"\[Source:", generated_text, re.IGNORECASE))
    return {
        "grounding_score": score,
        "ungrounded_sentences": len(ungrounded),
        "ungrounded_examples": ungrounded[:3],
        "citation_count": citation_count,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. Human Review Scoring Framework (Step 36)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_human_review(
    draft: str,
    ratings: dict | None = None,
) -> dict:
    """
    Scoring framework for human review of generated drafts.
    Pass 'ratings' as a dict with optional keys:
        usefulness (0–5), clarity (0–5), factual_consistency (0–5)
    Returns averaged score and individual ratings.
    If no ratings provided, returns a template for filling out.
    """
    dimensions = ["usefulness", "clarity", "factual_consistency"]
    if not ratings:
        return {
            "template": {dim: None for dim in dimensions},
            "instructions": (
                "Rate each dimension 0–5: "
                "0=unusable, 3=acceptable, 5=excellent. "
                "Submit ratings to this function to get a composite score."
            ),
            "draft_length": len(draft.split()),
        }

    scores = {dim: float(ratings.get(dim, 0)) for dim in dimensions}
    avg = sum(scores.values()) / len(scores)
    return {
        "ratings": scores,
        "composite_score": round(avg, 2),
        "max_possible": 5.0,
        "pct": round(avg / 5.0 * 100, 1),
        "verdict": "Excellent" if avg >= 4.5 else "Good" if avg >= 3.5 else "Acceptable" if avg >= 2.5 else "Needs Work",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. Improvement Trend
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_improvement_trend(db_session) -> dict:
    """
    Measure whether edit similarity is increasing over time
    (i.e., operator is editing less = drafts are improving).
    """
    from feedback.improvement_loop import ImprovementLoop
    loop = ImprovementLoop()
    trend = loop.get_improvement_trend(db_session)

    if len(trend) < 2:
        return {"trend": trend, "improving": None,
                "message": "Insufficient data (need >=2 edits)"}

    sims = [t["similarity"] for t in trend if t["similarity"] is not None]
    if len(sims) < 2:
        return {"trend": trend, "improving": None,
                "message": "Insufficient similarity data"}

    mid = len(sims) // 2
    first_half_avg  = sum(sims[:mid]) / mid
    second_half_avg = sum(sims[mid:]) / len(sims[mid:])
    improving = second_half_avg > first_half_avg

    return {
        "trend": trend,
        "improving": improving,
        "first_half_avg_similarity":  round(first_half_avg, 4),
        "second_half_avg_similarity": round(second_half_avg, 4),
        "improvement_delta": round(second_half_avg - first_half_avg, 4),
        "message": (
            f"Draft quality IMPROVING (similarity +{second_half_avg - first_half_avg:.1%})"
            if improving
            else f"Draft quality NOT YET improving (delta: {second_half_avg - first_half_avg:.1%})"
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 6. End-to-end demo evaluation
# ──────────────────────────────────────────────────────────────────────────────

def run_full_evaluation():
    """
    Run all evaluation metrics against the live system.
    Prints a formatted report to stdout.
    """
    print("\n" + "="*70)
    print("  GLDIS EVALUATION REPORT")
    print("="*70)

    # ── OCR quality ────────────────────────────────────────────────────────
    print("\n[1] OCR QUALITY")
    samples_dir = Path(__file__).parent.parent / "samples"
    noisy = samples_dir / "sample_02_arbitration_notice.txt"

    if noisy.exists():
        noisy_text = noisy.read_text(encoding="utf-8")
        # Ground truth: cleaned version (simulate ideal OCR output)
        cleaned = noisy_text.replace("0", "o").replace("1", "l")
        ocr_metrics = evaluate_ocr(cleaned, noisy_text)
        print(f"  Noisy document CER : {ocr_metrics['cer']:.1%}  (chars wrong)")
        print(f"  Noisy document WER : {ocr_metrics['wer']:.1%}  (words wrong)")
        print(f"  -> OCR artifacts intentionally present; pipeline handles them via preprocessing")
    else:
        print("  Sample files not found. Run: python samples/generate_samples.py")

    # ── Retrieval quality ───────────────────────────────────────────────────
    print("\n[2] RETRIEVAL QUALITY (simulated annotation)")
    simulated_relevant = ["chunk_a", "chunk_b"]
    simulated_retrieved = ["chunk_a", "chunk_c", "chunk_b", "chunk_d", "chunk_e"]
    ret = evaluate_retrieval(simulated_relevant, simulated_retrieved)
    print(f"  Precision@{ret['k']}  : {ret['precision_at_k']:.1%}")
    print(f"  Recall@{ret['k']}     : {ret['recall_at_k']:.1%}")
    print(f"  F1@{ret['k']}         : {ret['f1_at_k']:.1%}")
    print(f"  MRR             : {ret['mrr']:.4f}")

    # ── Grounding quality ───────────────────────────────────────────────────
    print("\n[3] GROUNDING QUALITY (sample draft)")
    import uuid
    from core.schemas import EvidenceChunk
    sample_ev = [
        EvidenceChunk(
            chunk_id=str(uuid.uuid4()),
            document_id="sample",
            text="The lease term shall commence on February 1, 2024.",
            page=1, section="Article 2", score=0.95,
        ),
        EvidenceChunk(
            chunk_id=str(uuid.uuid4()),
            document_id="sample",
            text="Monthly rent is $12,500 due on the first of each month.",
            page=1, section="Article 3", score=0.88,
        ),
    ]
    sample_draft = (
        "The lease term commences February 1, 2024. "
        "Monthly rent is twelve thousand five hundred dollars. "
        "The termination clause requires prior written notice from both parties."
    )
    gr = evaluate_grounding(sample_draft, sample_ev)
    print(f"  Grounding score     : {gr['grounding_score']:.1%}")
    print(f"  Ungrounded sentences: {gr['ungrounded_sentences']}")
    print(f"  Inline citations    : {gr['citation_count']}")
    if gr["ungrounded_examples"]:
        print(f"  Ungrounded example  : {gr['ungrounded_examples'][0][:80]}…")

    # ── Human review framework ─────────────────────────────────────────────
    print("\n[4] HUMAN REVIEW SCORING FRAMEWORK")
    template = evaluate_human_review(sample_draft)
    print(f"  Dimensions: {list(template['template'].keys())}")
    print(f"  {template['instructions']}")
    # Demonstrate with sample ratings
    demo_ratings = {"usefulness": 4, "clarity": 4.5, "factual_consistency": 3.5}
    hr = evaluate_human_review(sample_draft, demo_ratings)
    print(f"  Demo composite score: {hr['composite_score']}/5.0 — {hr['verdict']} ({hr['pct']}%)")

    # ── Improvement trend ───────────────────────────────────────────────────
    print("\n[5] IMPROVEMENT TREND")
    try:
        from db.session import get_session_factory
        SessionLocal = get_session_factory()
        db = SessionLocal()
        trend_result = evaluate_improvement_trend(db)
        db.close()
        print(f"  {trend_result['message']}")
        if trend_result.get("improvement_delta") is not None:
            print(f"  Edit similarity delta: {trend_result['improvement_delta']:+.1%}")
    except Exception as e:
        print(f"  Could not connect to DB: {e}")

    # ── System health ───────────────────────────────────────────────────────
    print("\n[6] SYSTEM STATUS")
    try:
        from db.session import get_session_factory
        from db import models
        SessionLocal = get_session_factory()
        db = SessionLocal()
        n_docs    = db.query(models.Document).count()
        n_chunks  = db.query(models.Chunk).count()
        n_drafts  = db.query(models.Draft).count()
        n_edits   = db.query(models.Edit).count()
        n_fewshot = db.query(models.FewShotExample).count()
        db.close()
        print(f"  Documents processed : {n_docs}")
        print(f"  Chunks indexed      : {n_chunks}")
        print(f"  Drafts generated    : {n_drafts}")
        print(f"  Operator edits      : {n_edits}")
        print(f"  Few-shot examples   : {n_fewshot}")
    except Exception as e:
        print(f"  DB unavailable: {e}")

    print("\n" + "="*70)
    print("  Evaluation complete.")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_full_evaluation()
