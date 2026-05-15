"""Tests for feedback loop: diff analyzer, feedback engine, improvement loop (Steps 29–32)."""
from __future__ import annotations
import pytest


def test_diff_analyzer_identical():
    from feedback.diff_analyzer import DiffAnalyzer
    analyzer = DiffAnalyzer()
    result = analyzer.analyze("Hello world.", "Hello world.")
    assert result.similarity == 1.0
    assert result.edit_distance == 0
    assert len(result.operations) == 0


def test_diff_analyzer_insertion():
    from feedback.diff_analyzer import DiffAnalyzer
    analyzer = DiffAnalyzer()
    result = analyzer.analyze(
        "The tenant shall pay rent.",
        "The tenant shall pay rent on January 1, 2024.",
    )
    assert result.similarity < 1.0
    assert len(result.operations) >= 1
    insertion_ops = [op for op in result.operations if op.edit_type.value == "insertion"]
    assert len(insertion_ops) >= 1


def test_diff_analyzer_deletion():
    from feedback.diff_analyzer import DiffAnalyzer
    analyzer = DiffAnalyzer()
    result = analyzer.analyze(
        "The tenant shall likely pay rent sometime soon maybe.",
        "The tenant shall pay rent.",
    )
    deletion_ops = [op for op in result.operations if op.edit_type.value == "deletion"]
    assert len(deletion_ops) >= 1


def test_diff_analyzer_hallucination_detection():
    from feedback.diff_analyzer import DiffAnalyzer
    analyzer = DiffAnalyzer()
    result = analyzer.analyze(
        "The contract was probably signed on an unknown date.",
        "The contract was signed on January 15, 2024.",
    )
    categories = [op.category for op in result.operations]
    assert "hallucination" in categories or "incorrect_fact" in categories


def test_diff_analyzer_infers_rules():
    from feedback.diff_analyzer import DiffAnalyzer
    analyzer = DiffAnalyzer()
    result = analyzer.analyze(
        "The amount is likely around 10,000 dollars approximately.",
        "The amount is $10,500.00 [Source: chunk-1, p.1]",
    )
    assert isinstance(result.inferred_rules, list)


def test_improvement_loop_returns_context(db_session):
    from feedback.improvement_loop import ImprovementLoop
    loop = ImprovementLoop()
    ctx = loop.get_generation_context(db_session)
    assert "few_shot_examples" in ctx
    assert "style_rules" in ctx
    assert "improvement_summary" in ctx


def test_improvement_trend_empty(db_session):
    from feedback.improvement_loop import ImprovementLoop
    loop = ImprovementLoop()
    trend = loop.get_improvement_trend(db_session)
    assert isinstance(trend, list)
