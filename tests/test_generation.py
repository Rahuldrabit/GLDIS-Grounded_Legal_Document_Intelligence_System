"""Tests for grounded draft generation (Steps 24–26)."""
from __future__ import annotations
import pytest


def test_grounding_score_all_grounded(sample_evidence):
    from generation.grounding import compute_grounding_score
    text = (
        f"The lease term commences February 1, 2024. [Source: {sample_evidence[0].chunk_id}, p.1]\n"
        f"Rent is $12,500 per month. [Source: {sample_evidence[1].chunk_id}, p.1]"
    )
    score, ungrounded = compute_grounding_score(text, sample_evidence)
    assert score > 0.5
    assert len(ungrounded) == 0


def test_grounding_score_ungrounded():
    from generation.grounding import compute_grounding_score
    from core.schemas import EvidenceChunk
    import uuid
    ev = [EvidenceChunk(
        chunk_id=str(uuid.uuid4()), document_id="d", text="Rent is one thousand dollars.",
        page=1, section="Rent", score=0.9
    )]
    text = "The moon is made of cheese and the sky is green on Tuesdays in Antarctica."
    score, ungrounded = compute_grounding_score(text, ev)
    assert len(ungrounded) >= 1


def test_extract_citations(sample_evidence):
    from generation.grounding import extract_citations
    cid = sample_evidence[0].chunk_id
    text = f"The term starts February 1, 2024. [Source: {cid}, p.1]"
    citations = extract_citations(text, sample_evidence)
    assert len(citations) >= 1
    assert citations[0].chunk_id == cid


def test_extract_citations_unknown_id(sample_evidence):
    from generation.grounding import extract_citations
    import uuid
    fake_id = str(uuid.uuid4())
    text = f"Some claim. [Source: {fake_id}, p.2]"
    citations = extract_citations(text, sample_evidence)
    assert len(citations) == 1
    assert citations[0].document_id == "unknown"


def test_format_evidence_block(sample_evidence):
    from generation.prompts import format_evidence_block
    block = format_evidence_block(sample_evidence)
    assert "Evidence 1" in block
    # chunk_id first 8 chars should appear in the block header
    assert sample_evidence[0].chunk_id[:8] in block


def test_build_generation_prompt(sample_evidence):
    from generation.prompts import build_generation_prompt
    messages = build_generation_prompt(
        query="Summarize the key facts.",
        evidence_chunks=sample_evidence,
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Evidence" in messages[1]["content"]


def test_mock_generator_produces_output(sample_evidence):
    from generation.generator import _call_mock
    result = _call_mock([], sample_evidence)
    assert "CASE FACT SUMMARY" in result
    assert "INTERNAL MEMO" in result
