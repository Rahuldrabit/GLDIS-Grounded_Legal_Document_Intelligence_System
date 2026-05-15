"""Integration tests for FastAPI endpoints (Step 37)."""
from __future__ import annotations
import io
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from main import app
    from db.session import create_tables

    create_tables()
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_api_status(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    data = r.json()
    assert "indexed_vectors" in data
    assert "openai_configured" in data


def test_upload_invalid_extension(client):
    r = client.post(
        "/api/documents/upload",
        files={"file": ("test.exe", b"fake", "application/octet-stream")},
    )
    assert r.status_code == 400


def test_upload_txt_file(client):
    content = b"LEASE AGREEMENT\nCase No: CV-2024-001\nDate: January 1, 2024"
    r = client.post(
        "/api/documents/upload",
        files={"file": ("test_lease.txt", content, "text/plain")},
    )
    assert r.status_code == 200
    data = r.json()
    assert "document_id" in data
    assert data["status"] == "uploaded"
    return data["document_id"]


def test_list_documents(client):
    r = client.get("/api/documents")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_get_nonexistent_document(client):
    r = client.get("/api/documents/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404


def test_search_empty_query(client):
    r = client.post("/api/search", params={"query": "  "})
    assert r.status_code == 400


def test_search_returns_results_structure(client):
    r = client.post("/api/search", params={"query": "lease agreement", "top_k": 3})
    assert r.status_code == 200
    data = r.json()
    assert "query" in data
    assert "results" in data


def test_feedback_history_empty(client):
    r = client.get("/api/feedback/history")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_improvements_endpoint(client):
    r = client.get("/api/feedback/improvements")
    assert r.status_code == 200
    data = r.json()
    assert "statistics" in data
    assert "active_style_rules" in data
