"""
Stage 6 (continued) — NER-Based Entity Extractor
Uses spaCy for named-entity recognition to extract parties, persons,
organisations, and semantic legal roles.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from core.schemas import ExtractedField

logger = logging.getLogger(__name__)

_NLP = None

def _load_nlp():
    global _NLP
    if _NLP is not None:
        return _NLP
    try:
        import spacy
        for model in ("en_core_web_lg", "en_core_web_sm"):
            try:
                _NLP = spacy.load(model)
                logger.info(f"spaCy model loaded: {model}")
                return _NLP
            except OSError:
                continue
        logger.warning("No spaCy model found. Run: python -m spacy download en_core_web_sm")
    except ImportError:
        logger.warning("spaCy not installed. NER extraction disabled.")
    return None


_LABEL_MAP = {
    "PERSON":  "person",
    "ORG":     "organisation",
    "GPE":     "jurisdiction",
    "LOC":     "location",
    "DATE":    "date",
    "MONEY":   "monetary_value",
    "LAW":     "legal_reference",
}

_OBLIGATION_TRIGGERS = {
    "shall", "must", "agrees to", "obligated to", "required to",
    "undertakes to", "is responsible for", "will provide", "will not",
}


class NERExtractor:
    """spaCy-based NER extractor for legal entities and obligation detection."""

    def __init__(self):
        self.nlp = _load_nlp()

    def extract(self, text: str, chunk_id: Optional[str] = None) -> List[ExtractedField]:
        if self.nlp is None:
            return []

        fields: List[ExtractedField] = []
        seen: set = set()

        try:
            doc = self.nlp(text[:100_000])
        except Exception as exc:
            logger.warning(f"NER processing failed: {exc}")
            return []

        for ent in doc.ents:
            field_name = _LABEL_MAP.get(ent.label_)
            if not field_name:
                continue
            value = ent.text.strip()
            if not value or len(value) < 2:
                continue
            key = (field_name, value.lower())
            if key in seen:
                continue
            seen.add(key)
            fields.append(ExtractedField(
                field=field_name, value=value,
                confidence=0.78, source_chunk_id=chunk_id,
            ))

        for sent in doc.sents:
            sent_lower = sent.text.lower()
            if any(t in sent_lower for t in _OBLIGATION_TRIGGERS):
                txt = sent.text.strip()
                if len(txt) > 10:
                    fields.append(ExtractedField(
                        field="obligation", value=txt,
                        confidence=0.72, source_chunk_id=chunk_id,
                    ))

        return fields

    def extract_all(self, chunks: list) -> List[ExtractedField]:
        all_fields: List[ExtractedField] = []
        for chunk in chunks:
            all_fields.extend(self.extract(chunk.text, chunk_id=chunk.chunk_id))
        return all_fields
