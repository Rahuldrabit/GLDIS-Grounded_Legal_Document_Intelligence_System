"""Tests for entity extraction modules (Steps 11–13)."""
from __future__ import annotations

import uuid
import pytest
from core.schemas import Chunk


def _make_chunk(text: str) -> Chunk:
    return Chunk(
        chunk_id=str(uuid.uuid4()),
        document_id="test-doc",
        text=text,
        page=1,
        section="Test Section",
        token_count=len(text.split()),
    )


# ── Rule-Based Extractor ─────────────────────────────────────────────────────

def test_rule_extracts_dates():
    from extraction.rule_based import RuleBasedExtractor
    extractor = RuleBasedExtractor()
    chunk = _make_chunk("The agreement was signed on January 15, 2024.")
    fields = extractor.extract(chunk.text, chunk_id=chunk.chunk_id)
    date_fields = [f for f in fields if f.field == "date"]
    assert len(date_fields) >= 1
    assert "2024" in date_fields[0].value


def test_rule_extracts_monetary_value():
    from extraction.rule_based import RuleBasedExtractor
    extractor = RuleBasedExtractor()
    chunk = _make_chunk("Tenant shall pay $12,500.00 per month.")
    fields = extractor.extract(chunk.text, chunk_id=chunk.chunk_id)
    money_fields = [f for f in fields if f.field == "monetary_value"]
    assert len(money_fields) >= 1


def test_rule_extracts_case_number():
    from extraction.rule_based import RuleBasedExtractor
    extractor = RuleBasedExtractor()
    chunk = _make_chunk("Case No. CV-2024-00123 is pending before the court.")
    fields = extractor.extract(chunk.text, chunk_id=chunk.chunk_id)
    case_fields = [f for f in fields if f.field == "case_number"]
    assert len(case_fields) >= 1


def test_rule_extracts_deadline():
    from extraction.rule_based import RuleBasedExtractor
    extractor = RuleBasedExtractor()
    chunk = _make_chunk("Payment is due by March 31, 2024.")
    fields = extractor.extract(chunk.text, chunk_id=chunk.chunk_id)
    deadline_fields = [f for f in fields if f.field == "deadline"]
    assert len(deadline_fields) >= 1


def test_rule_extract_all_from_chunks(sample_chunks):
    from extraction.rule_based import RuleBasedExtractor
    extractor = RuleBasedExtractor()
    fields = extractor.extract_all(sample_chunks)
    assert len(fields) > 0
    field_names = {f.field for f in fields}
    assert "date" in field_names or "monetary_value" in field_names


# ── NER Extractor ────────────────────────────────────────────────────────────

def test_ner_extractor_loads():
    from extraction.ner_extractor import NERExtractor
    extractor = NERExtractor()
    # If spaCy is not installed, extractor.nlp will be None — that's OK
    assert hasattr(extractor, "nlp")


def test_ner_extracts_from_chunk_if_spacy_available():
    from extraction.ner_extractor import NERExtractor
    extractor = NERExtractor()
    if extractor.nlp is None:
        pytest.skip("spaCy not installed or model not downloaded")
    chunk = _make_chunk("John Smith signed the contract on behalf of Acme Corp in New York.")
    fields = extractor.extract(chunk.text, chunk_id=chunk.chunk_id)
    field_names = {f.field for f in fields}
    assert "person" in field_names or "organisation" in field_names


def test_ner_obligation_detection():
    from extraction.ner_extractor import NERExtractor
    extractor = NERExtractor()
    if extractor.nlp is None:
        pytest.skip("spaCy not installed")
    chunk = _make_chunk("Tenant shall pay rent on the first of each month and must maintain the premises.")
    fields = extractor.extract(chunk.text, chunk_id=chunk.chunk_id)
    obligation_fields = [f for f in fields if f.field == "obligation"]
    assert len(obligation_fields) >= 1
