"""
FastAPI Main Application
Mounts all routes and initializes the database.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import get_settings
from db.session import create_tables
from api.routes import upload, process, draft, feedback, evidence

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("gldis")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events: Startup and Shutdown."""
    logger.info("Initializing GLDIS...")
    settings = get_settings()
    settings.ensure_dirs()
    create_tables()
    logger.info("Database initialized.")
    yield
    logger.info("Shutting down GLDIS.")

app = FastAPI(
    title="Grounded Legal Document Intelligence System (GLDIS)",
    description="AI system for processing, retrieving, and grounding legal document generation.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(upload.router, prefix="/upload", tags=["Ingestion"])
app.include_router(process.router, prefix="/process", tags=["Processing"])
app.include_router(draft.router, prefix="/draft", tags=["Generation"])
app.include_router(feedback.router, prefix="/feedback", tags=["Learning Loop"])
app.include_router(evidence.router, prefix="/evidence", tags=["Review Interface"])

@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "service": "GLDIS"}
