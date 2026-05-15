"""
Stage 2 — Image Preprocessing Pipeline
Prepares raw document images for OCR with deskewing, denoising,
contrast enhancement, and adaptive thresholding.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Optional heavy imports — gracefully degrade if unavailable
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("opencv-python-headless not installed; preprocessing will be skipped.")

try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("Pillow not installed; preprocessing will be skipped.")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logger.warning("pdf2image not installed; PDF rendering unavailable.")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import subprocess
    import json
    MINERU_AVAILABLE = True
except ImportError:
    MINERU_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# MinerU Layout Bridge (Optional — Stage 1 enhancement)
# ──────────────────────────────────────────────────────────────────────────────

class MinerULayoutBlock:
    """Represents a layout block extracted by MinerU at 300 DPI."""
    def __init__(self, page: int, block_type: str, text: str, bbox: List[float], confidence: float = 1.0):
        self.page = page
        self.block_type = block_type  # "text", "header", "footer", "table", etc.
        self.text = text
        self.bbox = bbox  # [x1, y1, x2, y2] in 300DPI pixels
        self.confidence = confidence


def extract_layout_blocks_mineru(pdf_path: str) -> Optional[List[List[MinerULayoutBlock]]]:
    """
    Call MinerU CLI to extract layout blocks from a PDF.
    Returns a list of page blocks (page 0 → blocks[], page 1 → blocks[], etc.)
    Returns None if MinerU is unavailable or fails.
    """
    from core.config import get_settings
    
    settings = get_settings()
    if not settings.mineru_enabled:
        return None

    try:
        import subprocess
        import json
        from pathlib import Path

        mineru_path = settings.mineru_cli_path or "magic-pdf"
        output_dir = Path(pdf_path).parent / ".mineru_out"
        output_dir.mkdir(exist_ok=True)

        # Run MinerU CLI: magic-pdf pdf-parse --pdf <pdf> --output-dir <dir>
        cmd = [mineru_path, "pdf-parse", "--pdf", pdf_path, "--output-dir", str(output_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            logger.warning(f"MinerU failed with code {result.returncode}: {result.stderr}")
            return None

        # Parse MinerU JSON output (expected: <output_dir>/<pdf_name>.json)
        pdf_name = Path(pdf_path).stem
        json_path = output_dir / f"{pdf_name}.json"

        if not json_path.exists():
            logger.warning(f"MinerU output not found at {json_path}")
            return None

        with open(json_path, "r") as f:
            mineru_data = json.load(f)

        # Convert MinerU structure to MinerULayoutBlock list per page
        pages_blocks: List[List[MinerULayoutBlock]] = []
        for page_data in mineru_data.get("pages", []):
            page_num = page_data.get("page_idx", 0)
            page_blocks = []
            for block in page_data.get("blocks", []):
                block_type = block.get("type", "text")  # MinerU block type
                text = block.get("text", "")
                bbox = block.get("bbox", [0, 0, 0, 0])  # [x1, y1, x2, y2]
                confidence = block.get("confidence", 1.0)
                page_blocks.append(MinerULayoutBlock(page_num, block_type, text, bbox, confidence))
            pages_blocks.append(page_blocks)

        logger.info(f"MinerU extracted {len(pages_blocks)} pages with blocks from {pdf_path}")
        return pages_blocks

    except FileNotFoundError:
        logger.warning(f"MinerU CLI not found at '{settings.mineru_cli_path}'. Falling back to pdf2image.")
        return None
    except Exception as exc:
        logger.warning(f"MinerU extraction failed: {exc}. Falling back to pdf2image.")
        return None


def pdf_to_images(pdf_path: str, dpi: int = 300) -> List["Image.Image"]:
    """
    Convert each PDF page to a high-resolution PIL Image.
    Falls back to PyMuPDF if pdf2image is unavailable.
    """
    if not PDF2IMAGE_AVAILABLE:
        if not PYMUPDF_AVAILABLE:
            logger.error("pdf2image and PyMuPDF not available. Cannot render PDF pages.")
            return []

        try:
            document = fitz.open(pdf_path)
            images = []
            zoom = dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            for page in document:
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                mode = "RGB" if pix.n < 4 else "RGBA"
                from PIL import Image as PILImage

                image = PILImage.frombytes(mode, [pix.width, pix.height], pix.samples)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                images.append(image)
            logger.info(f"Rendered {len(images)} pages from {pdf_path} with PyMuPDF fallback at {dpi} DPI.")
            return images
        except Exception as exc:
            logger.error(f"PyMuPDF PDF rendering failed for {pdf_path}: {exc}")
            return []

    try:
        images = convert_from_path(pdf_path, dpi=dpi)
        logger.info(f"Rendered {len(images)} pages from {pdf_path} at {dpi} DPI.")
        return images
    except Exception as exc:
        logger.error(f"PDF rendering failed for {pdf_path}: {exc}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Individual Preprocessing Steps
# ──────────────────────────────────────────────────────────────────────────────

def deskew(image_np: "np.ndarray") -> "np.ndarray":
    """
    Detect and correct page rotation using Hough line transform.
    Operates on grayscale images.
    """
    if not CV2_AVAILABLE:
        return image_np

    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) if len(image_np.shape) == 3 else image_np
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is None:
        return image_np

    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta) - 90
        if -45 < angle < 45:
            angles.append(angle)

    if not angles:
        return image_np

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return image_np  # Skip negligible skew

    h, w = image_np.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    deskewed = cv2.warpAffine(
        image_np, rotation_matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    logger.debug(f"Deskewed page by {median_angle:.2f}°")
    return deskewed


def denoise(image_np: "np.ndarray") -> "np.ndarray":
    """Apply Non-Local Means denoising to remove scanner artifacts."""
    if not CV2_AVAILABLE:
        return image_np
    return cv2.fastNlMeansDenoisingColored(image_np, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)


def enhance_contrast(image_np: "np.ndarray") -> "np.ndarray":
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve OCR visibility."""
    if not CV2_AVAILABLE:
        return image_np

    if len(image_np.shape) == 3:
        lab = cv2.cvtColor(image_np, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge((l_channel, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(image_np)


def adaptive_threshold(image_np: "np.ndarray") -> "np.ndarray":
    """
    Apply adaptive thresholding to improve text/background separation.
    Returns a binary image suitable for OCR on scanned docs.
    """
    if not CV2_AVAILABLE:
        return image_np

    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) if len(image_np.shape) == 3 else image_np
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )
    # Convert back to BGR so it can be fed to colour-expecting OCR engines
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def fix_orientation(image_np: "np.ndarray") -> "np.ndarray":
    """
    Detect and fix 180° flipped pages using Tesseract OSD.
    Silently skips if pytesseract is not installed.
    """
    try:
        import pytesseract
        osd = pytesseract.image_to_osd(image_np, output_type=pytesseract.Output.DICT)
        rotation = osd.get("rotate", 0)
        if rotation:
            logger.debug(f"Fixing orientation: rotating {rotation}°")
            if rotation == 90:
                return cv2.rotate(image_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif rotation == 180:
                return cv2.rotate(image_np, cv2.ROTATE_180)
            elif rotation == 270:
                return cv2.rotate(image_np, cv2.ROTATE_90_CLOCKWISE)
    except Exception:
        pass
    return image_np


# ──────────────────────────────────────────────────────────────────────────────
# Full Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_image(
    pil_image: "Image.Image",
    apply_deskew: bool = True,
    apply_denoise: bool = True,
    apply_contrast: bool = True,
    apply_threshold: bool = False,   # Only for very noisy scans
    apply_orientation: bool = True,
) -> "Image.Image":
    """
    Full preprocessing pipeline for a single PIL Image.
    Returns a preprocessed PIL Image ready for OCR.
    """
    if not CV2_AVAILABLE or not PIL_AVAILABLE:
        return pil_image

    # PIL → NumPy (BGR for OpenCV)
    img_np = np.array(pil_image.convert("RGB"))
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    if apply_orientation:
        img_np = fix_orientation(img_np)
    if apply_deskew:
        img_np = deskew(img_np)
    if apply_denoise:
        img_np = denoise(img_np)
    if apply_contrast:
        img_np = enhance_contrast(img_np)
    if apply_threshold:
        img_np = adaptive_threshold(img_np)

    # NumPy → PIL
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def preprocess_pdf_pages(
    pdf_path: str,
    dpi: int = 300,
    **kwargs,
) -> List[Optional["Image.Image"]]:
    """
    Render PDF pages and run the full preprocessing pipeline on each page.
    """
    raw_images = pdf_to_images(pdf_path, dpi=dpi)
    if not raw_images:
        return []

    processed = []
    for i, img in enumerate(raw_images):
        try:
            processed.append(preprocess_image(img, **kwargs))
            logger.debug(f"Preprocessed page {i + 1}/{len(raw_images)}")
        except Exception as exc:
            logger.warning(f"Preprocessing failed for page {i + 1}: {exc}; using raw image.")
            processed.append(img)

    return processed
