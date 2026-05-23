"""
Stage 3 — Hybrid OCR Engine (Steps 3, 5, 10)

Routing order:
  1. VLM (Qwen2.5-VL / GPT-4.1 Vision) — if vlm_enabled=True and images available
     Fallback: VLM confidence < threshold → continue to OCR
  2. Digital PDF → PyMuPDF (fast, lossless)
  3. Scanned images → Tesseract (clean pages)
     Fallback: avg confidence < 0.6 → PaddleOCR
  4. Complex layouts / handwriting → PaddleOCR
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from core.config import get_settings
from core.schemas import DocumentOCRResult, OCREngine, PageOCRResult, TextBlock

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Engine availability checks
# ──────────────────────────────────────────────────────────────────────────────

def _check_pymupdf() -> bool:
    try:
        import fitz  # noqa: F401
        return True
    except ImportError:
        return False


def _check_tesseract() -> bool:
    try:
        import pytesseract
        settings = get_settings()
        if settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _check_paddleocr() -> bool:
    try:
        from paddleocr import PaddleOCR  # noqa: F401
        return True
    except ImportError:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Individual engine implementations
# ──────────────────────────────────────────────────────────────────────────────

def _ocr_pymupdf(pdf_path: str) -> List[PageOCRResult]:
    """Extract text directly from digital PDFs using PyMuPDF (fastest, lossless)."""
    import fitz

    results: List[PageOCRResult] = []
    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc, start=1):
        blocks_raw = page.get_text("dict")["blocks"]
        blocks: List[TextBlock] = []
        full_text_parts: List[str] = []

        for block in blocks_raw:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    txt = span.get("text", "").strip()
                    if txt:
                        bbox = span.get("bbox", [0, 0, 0, 0])
                        blocks.append(TextBlock(text=txt, bbox=list(bbox), confidence=1.0))
                        full_text_parts.append(txt)

        full_text = " ".join(full_text_parts)
        results.append(PageOCRResult(
            page=page_num,
            text=full_text,
            confidence=1.0,
            engine=OCREngine.PYMUPDF,
            blocks=blocks,
        ))

    doc.close()
    logger.info(f"PyMuPDF: {len(results)} pages from {pdf_path}")
    return results


def _ocr_tesseract(image_paths: List[str]) -> List[PageOCRResult]:
    """Run Tesseract OCR on pre-rendered page images."""
    import pytesseract
    from PIL import Image

    settings = get_settings()
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    results: List[PageOCRResult] = []
    for page_num, img_path in enumerate(image_paths, start=1):
        try:
            img = Image.open(img_path)
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

            blocks: List[TextBlock] = []
            full_text_parts: List[str] = []
            confidences: List[float] = []

            for i, word in enumerate(data["text"]):
                word = word.strip()
                if not word:
                    continue
                conf = float(data["conf"][i])
                if conf < 0:
                    continue
                conf_norm = conf / 100.0
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                blocks.append(TextBlock(
                    text=word, bbox=[x, y, x + w, y + h], confidence=conf_norm,
                ))
                full_text_parts.append(word)
                confidences.append(conf_norm)

            avg_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0
            results.append(PageOCRResult(
                page=page_num,
                text=" ".join(full_text_parts),
                confidence=avg_conf,
                engine=OCREngine.TESSERACT,
                blocks=blocks,
            ))
        except Exception as exc:
            logger.warning(f"Tesseract failed on page {page_num}: {exc}")
            results.append(PageOCRResult(
                page=page_num, text="", confidence=0.0,
                engine=OCREngine.TESSERACT, blocks=[],
            ))
    return results


def _ocr_paddleocr(image_paths: List[str]) -> List[PageOCRResult]:
    """Run PaddleOCR for complex layout / mixed-language documents."""
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    results: List[PageOCRResult] = []

    for page_num, img_path in enumerate(image_paths, start=1):
        try:
            paddle_result = ocr.ocr(img_path, cls=True)
            blocks: List[TextBlock] = []
            text_parts: List[str] = []
            confs: List[float] = []

            if paddle_result and paddle_result[0]:
                for line in paddle_result[0]:
                    bbox_pts, (txt, conf) = line
                    xs = [p[0] for p in bbox_pts]
                    ys = [p[1] for p in bbox_pts]
                    flat_bbox = [min(xs), min(ys), max(xs), max(ys)]
                    blocks.append(TextBlock(text=txt, bbox=flat_bbox, confidence=float(conf)))
                    text_parts.append(txt)
                    confs.append(float(conf))

            avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
            results.append(PageOCRResult(
                page=page_num,
                text=" ".join(text_parts),
                confidence=avg_conf,
                engine=OCREngine.PADDLEOCR,
                blocks=blocks,
            ))
        except Exception as exc:
            logger.warning(f"PaddleOCR failed on page {page_num}: {exc}")
            results.append(PageOCRResult(
                page=page_num, text="", confidence=0.0,
                engine=OCREngine.PADDLEOCR, blocks=[],
            ))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _is_digital_pdf(pdf_path: str) -> bool:
    """Heuristic: PDF has embedded text (>100 chars) → treat as digital."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        total_chars = sum(len(p.get_text()) for p in doc)
        doc.close()
        return total_chars > 100
    except Exception:
        return False


def _avg_confidence(pages: List[PageOCRResult]) -> float:
    if not pages:
        return 0.0
    return sum(p.confidence for p in pages) / len(pages)


# ──────────────────────────────────────────────────────────────────────────────
# HybridOCR Dispatcher
# ──────────────────────────────────────────────────────────────────────────────

class HybridOCR:
    """
    Routes each document to the optimal understanding engine.

    VLM-first routing (Step 7):
        Image pages → VLM (Qwen2.5-VL / GPT-4.1 Vision)
            → if confidence < threshold: Tesseract → PaddleOCR

    Digital PDF routing:
        Digital PDF → PyMuPDF (text layer, lossless)

    Legacy OCR routing (fallback):
        Scanned pages → Tesseract
            → if avg_conf < 0.6: PaddleOCR
    """

    def __init__(self):
        settings = get_settings()
        if settings.tesseract_cmd:
            try:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
            except Exception:
                pass
        self.has_pymupdf = _check_pymupdf()
        self.has_tesseract = _check_tesseract()
        self.has_paddleocr = _check_paddleocr()
        logger.info(
            f"HybridOCR engines — PyMuPDF:{self.has_pymupdf} "
            f"Tesseract:{self.has_tesseract} PaddleOCR:{self.has_paddleocr}"
        )

    def extract(
        self,
        file_path: str,
        image_paths: Optional[List[str]] = None,
        force_engine: Optional[OCREngine] = None,
    ) -> DocumentOCRResult:
        """
        Main entry point. Returns DocumentOCRResult with all pages.

        Routing logic (unless force_engine is set):
          1. For image-based docs: VLM-first if enabled, then OCR fallback
          2. For digital PDFs: PyMuPDF
          3. For scanned PDFs (no text layer): images → VLM → Tesseract → PaddleOCR
        """
        settings = get_settings()
        document_id = Path(file_path).stem
        pages: List[PageOCRResult] = []
        vlm_logs: List[dict] = []

        ext = Path(file_path).suffix.lower()
        if ext == ".txt":
            try:
                text = Path(file_path).read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = Path(file_path).read_text(encoding="utf-8", errors="ignore")

            pages = [
                PageOCRResult(
                    page=1,
                    text=text,
                    confidence=1.0,
                    engine=OCREngine.PYMUPDF,
                    blocks=[],
                )
            ]
            return DocumentOCRResult(document_id=document_id, pages=pages, vlm_logs=[])
        is_pdf = ext == ".pdf"

        # ── Forced engine (API override) ─────────────────────────────────────
        if force_engine == OCREngine.PYMUPDF:
            pages = self._run_pymupdf(file_path, is_pdf)
        elif force_engine == OCREngine.PADDLEOCR:
            pages = self._run_paddleocr(file_path, image_paths)
        elif force_engine == OCREngine.TESSERACT:
            pages = self._run_tesseract(file_path, image_paths)
        elif force_engine == OCREngine.VLM:
            pages, vlm_logs = self._run_vlm(file_path, image_paths)

        # ── Auto routing ─────────────────────────────────────────────────────
        elif is_pdf and self.has_pymupdf and _is_digital_pdf(file_path):
            # Digital PDF → PyMuPDF (no images needed)
            logger.info("Route: digital PDF → PyMuPDF")
            pages = self._run_pymupdf(file_path, is_pdf)

        else:
            # Image-based doc or scanned PDF
            img_paths = image_paths or ([] if is_pdf else [file_path])

            if img_paths and settings.vlm_enabled:
                # ── Step 7: VLM-first with MinerU block guidance ─────────────
                logger.info(f"Route: image pages → VLM ({settings.vlm_model})")
                # Try block-guided extraction first (MinerU-guided)
                pages, vlm_logs = self._run_vlm_blocks(file_path, img_paths)
                avg_conf = _avg_confidence(pages)

                # If block-guided produced nothing or low confidence, fall back to standard VLM
                if not pages or avg_conf < settings.vlm_confidence_threshold:
                    logger.info(
                        f"Block-guided VLM missing/low-confidence ({avg_conf:.2f}) -> standard VLM"
                    )
                    pages, vlm_logs = self._run_vlm(file_path, img_paths)
                    avg_conf = _avg_confidence(pages)

                if avg_conf < settings.vlm_confidence_threshold:
                    logger.info(
                        f"VLM confidence {avg_conf:.2f} < {settings.vlm_confidence_threshold} "
                        f"→ falling back to OCR"
                    )
                    pages = self._ocr_fallback(img_paths)
            else:
                # ── OCR fallback (VLM disabled or no images) ─────────────────
                pages = self._ocr_fallback(img_paths)

        return DocumentOCRResult(document_id=document_id, pages=pages, vlm_logs=vlm_logs)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _run_pymupdf(self, file_path: str, is_pdf: bool) -> List[PageOCRResult]:
        if not is_pdf or not self.has_pymupdf:
            return []
        return _ocr_pymupdf(file_path)

    def _run_tesseract(self, file_path: str, image_paths: Optional[List[str]]) -> List[PageOCRResult]:
        if not self.has_tesseract:
            return []
        return _ocr_tesseract(image_paths or [file_path])

    def _run_paddleocr(self, file_path: str, image_paths: Optional[List[str]]) -> List[PageOCRResult]:
        if not self.has_paddleocr:
            return []
        return _ocr_paddleocr(image_paths or [file_path])

    def _run_vlm(
        self, file_path: str, image_paths: Optional[List[str]]
    ) -> Tuple[List[PageOCRResult], List[dict]]:
        from ocr.vlm_extractor import extract_with_vlm
        paths = image_paths or [file_path]
        return extract_with_vlm(paths)

    def _run_vlm_blocks(
        self, file_path: str, image_paths: Optional[List[str]]
    ) -> Tuple[List[PageOCRResult], List[dict]]:
        """
        Attempt MinerU-guided block extraction via the adapter.
        Falls back to empty list if adapter is unavailable.
        """
        try:
            from ocr.block_extraction_adapter import extract_vlm_with_blocks
            if not image_paths:
                return [], []
            result, logs = extract_vlm_with_blocks(file_path, image_paths)
            return result.pages, logs
        except Exception:
            return [], []

    def _ocr_fallback(self, img_paths: List[str]) -> List[PageOCRResult]:
        """Tesseract → PaddleOCR fallback for scanned/image pages."""
        if not img_paths:
            logger.error("No images available for OCR fallback.")
            return []

        if self.has_tesseract:
            logger.info("Route: scanned → Tesseract")
            pages = _ocr_tesseract(img_paths)
            avg_conf = _avg_confidence(pages)
            if avg_conf < 0.6 and self.has_paddleocr:
                logger.info(f"Tesseract confidence {avg_conf:.2f} < 0.6 → PaddleOCR")
                return _ocr_paddleocr(img_paths)
            return pages

        if self.has_paddleocr:
            logger.info("Route: complex layout → PaddleOCR")
            return _ocr_paddleocr(img_paths)

        logger.error("No OCR engine available. Install Tesseract or PaddleOCR.")
        return []
