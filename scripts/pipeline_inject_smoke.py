"""
Local full-pipeline smoke runner.

Creates a temporary PDF and PNG, uploads both through the FastAPI app,
runs synchronous processing, and generates drafts for each document.

Intended for local validation with LM Studio + Docker running.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure a shared in-memory test DB and local model defaults before importing app.
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("VLM_ENABLED", "true")
os.environ.setdefault("VLM_API_BASE", "http://localhost:1234/v1")
os.environ.setdefault("VLM_MODEL", "qwen/qwen2.5-vl-7b-instruct")
os.environ.setdefault("LLM_PROVIDER", "lmstudio")
os.environ.setdefault("LLM_BASE_URL", "http://localhost:1234/v1")
os.environ.setdefault("LLM_MODEL", "gemma-3")
os.environ.setdefault("REASONING_API_BASE", "http://localhost:1234/v1")
os.environ.setdefault("REASONING_MODEL", "gemma-3")
os.environ.setdefault("VERIFIER_ENABLED", "false")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
os.environ.setdefault("UPLOAD_DIR", tempfile.mkdtemp())
os.environ.setdefault("FAISS_INDEX_PATH", tempfile.mkdtemp())
os.environ.setdefault("BM25_INDEX_PATH", tempfile.mkdtemp())
os.environ.setdefault("FEEDBACK_STORE_PATH", tempfile.mkdtemp())

from db.session import create_tables  # noqa: E402
from main import app  # noqa: E402


SAMPLE_TEXT = """COMMERCIAL LEASE AGREEMENT

This Commercial Lease Agreement is entered into as of January 15, 2024,
by and between Acme Properties LLC (Landlord) and Beta Corp (Tenant).

Case No: CV-2024-00123

ARTICLE 1 - PREMISES
Landlord leases to Tenant the property located at 500 Main Street, New York, NY 10001.

ARTICLE 2 - TERM
The lease term commences on February 1, 2024 and expires on January 31, 2026.

ARTICLE 3 - RENT
Tenant shall pay monthly rent of $12,500.00 due on the first of each month.

ARTICLE 4 - OBLIGATIONS
Tenant must maintain the premises in good condition.
Tenant shall not sublet without written consent of Landlord.
"""


def _load_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 34)
    except Exception:
        return ImageFont.load_default()


def _build_test_image(output_dir: Path) -> Path:
    image = Image.new("RGB", (1700, 2200), "white")
    draw = ImageDraw.Draw(image)
    font = _load_font()
    x = 80
    y = 80
    for line in SAMPLE_TEXT.splitlines():
        draw.text((x, y), line, fill="black", font=font)
        y += 56 if line.strip() else 28
    image_path = output_dir / "smoke_doc.png"
    image.save(image_path)
    return image_path


def _build_test_pdf(output_dir: Path) -> Path:
    image = Image.new("RGB", (1700, 2200), "white")
    draw = ImageDraw.Draw(image)
    font = _load_font()
    x = 80
    y = 80
    for line in SAMPLE_TEXT.splitlines():
        draw.text((x, y), line, fill="black", font=font)
        y += 56 if line.strip() else 28
    pdf_path = output_dir / "smoke_doc.pdf"
    image.save(pdf_path, "PDF", resolution=300.0)
    return pdf_path


def _upload_and_process(client: TestClient, file_path: Path) -> Dict[str, Any]:
    with file_path.open("rb") as handle:
        response = client.post(
            "/api/documents/upload",
            files={"file": (file_path.name, handle, "application/octet-stream")},
        )
    response.raise_for_status()
    upload_data = response.json()
    document_id = upload_data["document_id"]

    process_response = client.post(f"/api/documents/{document_id}/process/sync")
    process_response.raise_for_status()
    process_data = process_response.json()

    document_response = client.get(f"/api/documents/{document_id}")
    document_response.raise_for_status()
    document_data = document_response.json()

    draft_response = client.post(
        "/api/drafts/generate",
        json={"document_id": document_id, "query": "Generate a case fact summary and internal memo.", "top_k": 5},
    )
    draft_response.raise_for_status()
    draft_data = draft_response.json()

    evidence_response = client.get(f"/api/drafts/{draft_data['draft_id']}/evidence")
    evidence_response.raise_for_status()

    return {
        "document_id": document_id,
        "upload": upload_data,
        "process": process_data,
        "document": document_data,
        "draft": draft_data,
        "evidence": evidence_response.json(),
    }


def main() -> None:
    create_tables()

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        pdf_path = _build_test_pdf(temp_path)
        image_path = _build_test_image(temp_path)

        with TestClient(app) as client:
            pdf_result = _upload_and_process(client, pdf_path)
            image_result = _upload_and_process(client, image_path)

        print("PDF result:\n" + json.dumps(pdf_result, indent=2, default=str))
        print("\nPNG result:\n" + json.dumps(image_result, indent=2, default=str))


if __name__ == "__main__":
    main()
