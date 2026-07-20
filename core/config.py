"""
Configuration management using pydantic-settings.
All settings are loaded from environment variables or .env file.
"""
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "sqlite:///./data/gldis.db"

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1"
    openai_base_url: str = "https://api.openai.com/v1"

    # ── LLM Provider ────────────────────────────────────────────────────────
    # Supported values: lmstudio, ollama, openai, openrouter
    llm_provider: str = ""
    llm_model: str = "qwen/qwen3-vl-4b"
    llm_base_url: str = "http://localhost:1234/v1"
    llm_api_key: str = "lm-studio"
    llm_max_tokens: int = 1200

    # ── OpenRouter (OpenAI-compatible provider) ────────────────────────────────
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openai/gpt-4o-mini"

    # ── API Rate Limiting (LLM-heavy endpoints) ────────────────────────────────
    llm_rate_limit_enabled: bool = True
    llm_rate_limit_requests: int = 30
    llm_rate_limit_window_seconds: int = 60

    # ── VLM (Qwen2.5-VL via LM Studio — primary document understanding) ───────
    # Set vlm_enabled=true and point vlm_api_base at a running LM Studio server.
    # When enabled, VLM is tried first for image-based pages; OCR is the fallback.
    vlm_enabled: bool = False
    vlm_model: str = "qwen2.5-vl-7b-instruct"
    vlm_api_base: str = "http://localhost:1234/v1"
    vlm_confidence_threshold: float = 0.6   # Below this → fall back to OCR
    vlm_max_tokens: int = 4000

    # ── OCR ───────────────────────────────────────────────────────────────────
    tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_batch_size: int = 32

    # ── Storage paths ─────────────────────────────────────────────────────────
    upload_dir: str = "./data/uploads"
    faiss_index_path: str = "./data/faiss_index"
    bm25_index_path: str = "./data/bm25_index"
    feedback_store_path: str = "./data/feedback"

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_target_tokens: int = 400
    chunk_overlap_tokens: int = 75

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = 10
    rerank_top_k: int = 5
    graphrag_enabled: bool = False  # Stage 2 GraphRAG enhancement
    graph_expand_hops: int = 1  # Neo4j neighborhood traversal depth

    # ── Graph (Stage 2 — Neo4j GraphRAG) ──────────────────────────────────────
    neo4j_enabled: bool = False
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "gldis_secret"

    # ── Generation ────────────────────────────────────────────────────────────
    max_evidence_tokens: int = 3000
    generation_temperature: float = 0.1

    # ── Reasoning (Stage 3 — DeepSeek-R1 + Llama-4 Verifier) ──────────────────
    reasoning_model: str = "deepseek-r1-distill-llama-70b"  # Reasoner model ID
    reasoning_api_base: str = "http://localhost:8001/v1"  # Separate model server
    verifier_model: str = "llama-4-8b"  # Verification model ID
    verifier_api_base: str = "http://localhost:8002/v1"  # Separate model server
    verifier_enabled: bool = True
    max_correction_iterations: int = 2

    # ── Stage 1: MinerU (optional layout bridge) ──────────────────────────────
    mineru_enabled: bool = False
    mineru_cli_path: str = "magic-pdf"  # Path to MinerU CLI; can be full path on Windows

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    def ensure_dirs(self) -> None:
        """Create all required data directories."""
        for path_str in [
            self.upload_dir,
            self.faiss_index_path,
            self.bm25_index_path,
            self.feedback_store_path,
            "./data",
        ]:
            Path(path_str).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
