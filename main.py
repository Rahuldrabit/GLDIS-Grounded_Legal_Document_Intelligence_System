"""
GLDIS — Grounded Legal Document Intelligence System
FastAPI application entrypoint.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from core.config import get_settings
from llm.client import resolve_llm_config
from db.session import create_tables
from routes import documents, drafts, feedback, retrieval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create directories and initialise the database."""
    settings = get_settings()
    settings.ensure_dirs()
    create_tables()
    logger.info("GLDIS started — database and directories ready.")
    yield
    logger.info("GLDIS shutting down.")


app = FastAPI(
    title="Grounded Legal Document Intelligence System",
    description=(
        "Ingests messy legal documents, extracts structured information, "
        "retrieves grounded evidence, and generates citable legal drafts. "
        "Learns from operator edits to improve over time."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routers
app.include_router(documents.router)
app.include_router(drafts.router)
app.include_router(feedback.router)
app.include_router(retrieval.router)




@app.get("/health", tags=["system"])
def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "GLDIS"}


@app.get("/api/status", tags=["system"])
def system_status():
    """Return system configuration and index status."""
    from retrieval.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever()
    settings = get_settings()
    llm_config = resolve_llm_config(mode="text")
    return {
        "model": llm_config.model,
        "llm_provider": llm_config.provider,
        "llm_base_url": llm_config.base_url,
        "embedding_model": settings.embedding_model,
        "indexed_vectors": retriever.total_indexed,
        "openai_configured": bool(
            settings.openai_api_key and settings.openai_api_key != "sk-your-key-here"
        ),
        "openai_model": settings.openai_model,
        "vlm_enabled": settings.vlm_enabled,
        "vlm_model": settings.vlm_model,
        "upload_dir": settings.upload_dir,
    }


# Serve React UI in production
ui_dist_dir = Path(__file__).parent / "ui" / "dist"
if ui_dist_dir.exists():
    app.mount("/", StaticFiles(directory=str(ui_dist_dir), html=True), name="ui")
else:
    # Fallback to legacy static UI if React build isn't found
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        @app.get("/", include_in_schema=False)
        def serve_ui():
            return FileResponse(str(static_dir / "index.html"))


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
