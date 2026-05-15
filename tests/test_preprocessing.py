"""Tests for image preprocessing pipeline (Steps 5–6)."""
from __future__ import annotations

import numpy as np
import pytest


def test_approx_tokens():
    from preprocessing.chunker import _approx_tokens
    assert _approx_tokens("hello world") > 0
    assert _approx_tokens("a" * 400) == 100


def test_split_sentences_basic():
    from preprocessing.chunker import _split_into_sentences
    text = "The Tenant shall pay rent. Landlord agrees to maintain. Case No. 1 is active."
    parts = _split_into_sentences(text)
    assert len(parts) >= 2


def test_deskew_no_crash():
    """deskew should return an array without crashing on a blank image."""
    from preprocessing.pipeline import deskew, CV2_AVAILABLE
    if not CV2_AVAILABLE:
        pytest.skip("OpenCV not available")
    import cv2
    blank = np.ones((100, 100, 3), dtype=np.uint8) * 255
    result = deskew(blank)
    assert result.shape == blank.shape


def test_enhance_contrast_no_crash():
    from preprocessing.pipeline import enhance_contrast, CV2_AVAILABLE
    if not CV2_AVAILABLE:
        pytest.skip("OpenCV not available")
    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    result = enhance_contrast(img)
    assert result.shape == img.shape


def test_preprocess_image_returns_pil():
    from preprocessing.pipeline import preprocess_image, PIL_AVAILABLE, CV2_AVAILABLE
    if not (PIL_AVAILABLE and CV2_AVAILABLE):
        pytest.skip("Pillow or OpenCV not available")
    from PIL import Image
    pil_img = Image.new("RGB", (200, 200), color=(200, 200, 200))
    result = preprocess_image(pil_img, apply_threshold=False)
    assert isinstance(result, Image.Image)
    assert result.size == (200, 200)
