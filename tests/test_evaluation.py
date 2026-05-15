"""Tests for evaluation metrics (Steps 33–36)."""
from __future__ import annotations
import pytest


def test_evaluate_ocr_perfect():
    from evaluation.evaluate import evaluate_ocr
    text = "The quick brown fox jumps over the lazy dog."
    result = evaluate_ocr(text, text)
    assert result["cer"] == 0.0
    assert result["wer"] == 0.0


def test_evaluate_ocr_with_errors():
    from evaluation.evaluate import evaluate_ocr
    ground_truth = "The tenant shall pay rent."
    hypothesis  = "The tenant shall pay rnt."   # one char error
    result = evaluate_ocr(ground_truth, hypothesis)
    assert result["cer"] > 0.0


def test_evaluate_retrieval_perfect():
    from evaluation.evaluate import evaluate_retrieval
    relevant = ["c1", "c2", "c3"]
    retrieved = ["c1", "c2", "c3"]
    r = evaluate_retrieval(relevant, retrieved)
    assert r["precision_at_k"] == 1.0
    assert r["recall_at_k"] == 1.0
    assert r["mrr"] == 1.0


def test_evaluate_retrieval_partial():
    from evaluation.evaluate import evaluate_retrieval
    relevant = ["c1", "c2"]
    retrieved = ["c1", "c3", "c4", "c2"]
    r = evaluate_retrieval(relevant, retrieved)
    assert 0 < r["precision_at_k"] < 1.0
    assert r["recall_at_k"] == 1.0
    assert r["mrr"] == 1.0   # c1 is at rank 1


def test_evaluate_retrieval_no_hits():
    from evaluation.evaluate import evaluate_retrieval
    r = evaluate_retrieval(["c1", "c2"], ["c3", "c4"])
    assert r["precision_at_k"] == 0.0
    assert r["recall_at_k"] == 0.0
    assert r["mrr"] == 0.0


def test_evaluate_retrieval_empty():
    from evaluation.evaluate import evaluate_retrieval
    r = evaluate_retrieval([], [])
    assert r["precision_at_k"] == 0.0


def test_evaluate_grounding(sample_evidence):
    from evaluation.evaluate import evaluate_grounding
    cid = sample_evidence[0].chunk_id
    text = f"The lease term commences February 1, 2024. [Source: {cid}, p.1]"
    result = evaluate_grounding(text, sample_evidence)
    assert "grounding_score" in result
    assert result["grounding_score"] > 0


def test_levenshtein_distance():
    from evaluation.evaluate import _levenshtein
    assert _levenshtein("abc", "abc") == 0
    assert _levenshtein("abc", "axc") == 1
    assert _levenshtein("", "abc") == 3
