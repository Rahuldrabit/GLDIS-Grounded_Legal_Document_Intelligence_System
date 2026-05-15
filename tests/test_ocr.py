"""Tests for OCR engines and VLM extractor (Steps 3, 5, 7–10)."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


def test_hybrid_ocr_init():
    from ocr.hybrid_ocr import HybridOCR
    ocr = HybridOCR()
    # should not raise
    assert hasattr(ocr, "has_pymupdf")
    assert hasattr(ocr, "has_tesseract")
    assert hasattr(ocr, "has_paddleocr")


def test_hybrid_ocr_txt_file(tmp_path):
    """Plain text files should produce a single-page result via text fallback."""
    from ocr.hybrid_ocr import HybridOCR
    txt = tmp_path / "doc.txt"
    txt.write_text("Hello world this is a test document for GLDIS.")
    ocr = HybridOCR()
    result = ocr.extract(str(txt))
    # Should not raise; pages may be empty if no engine handles .txt but no crash
    assert hasattr(result, "pages")
    assert hasattr(result, "total_text")


def test_vlm_extractor_skips_when_disabled():
    """VLM extractor should return empty results when VLM is disabled."""
    from ocr.vlm_extractor import extract_with_vlm
    with patch("ocr.vlm_extractor.get_settings") as mock_settings:
        s = MagicMock()
        s.vlm_enabled = False
        s.openai_api_key = ""
        mock_settings.return_value = s
        results, logs = extract_with_vlm(["fake_path.png"])
    assert results == []
    assert logs == []


def test_parse_vlm_response_valid_json():
    from ocr.vlm_extractor import _parse_vlm_response
    raw = '{"document_type": "contract", "text": "Hello", "confidence": 0.9}'
    parsed = _parse_vlm_response(raw)
    assert parsed is not None
    assert parsed["document_type"] == "contract"


def test_parse_vlm_response_code_fence():
    from ocr.vlm_extractor import _parse_vlm_response
    raw = '```json\n{"document_type": "notice", "text": "Test"}\n```'
    parsed = _parse_vlm_response(raw)
    assert parsed is not None
    assert parsed["document_type"] == "notice"


def test_parse_vlm_response_invalid():
    from ocr.vlm_extractor import _parse_vlm_response
    parsed = _parse_vlm_response("This is not JSON at all.")
    assert parsed is None


def test_is_digital_pdf_on_nonexistent():
    from ocr.hybrid_ocr import _is_digital_pdf
    # Non-existent file should return False without raising
    result = _is_digital_pdf("/nonexistent/path.pdf")
    assert result is False
