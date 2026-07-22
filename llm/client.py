"""Shared helpers for OpenAI-compatible LLM providers.

This module centralizes provider selection so the rest of the codebase can
switch between LM Studio, Ollama, OpenAI, and OpenRouter through configuration
only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from core.config import get_settings


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    base_url: str
    api_key: str
    model: str
    max_tokens: int
    supports_json_mode: bool


def _normalize_provider(provider: str) -> str:
    provider = (provider or "").strip().lower()
    if provider in {"lmstudio", "lm-studio", "local", "local-openai"}:
        return "lmstudio"
    if provider in {"ollama"}:
        return "ollama"
    if provider in {"openai"}:
        return "openai"
    if provider in {"openrouter"}:
        return "openrouter"
    return "lmstudio"


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def resolve_llm_config(
    mode: str = "text",
    header_provider: Optional[str] = None,
    header_api_key: Optional[str] = None,
    header_model: Optional[str] = None,
    header_base_url: Optional[str] = None,
) -> LLMConfig:
    settings = get_settings()
    provider_override = _as_text(getattr(settings, "llm_provider", "")).strip().lower()
    openai_api_key = _as_text(getattr(settings, "openai_api_key", ""))
    openrouter_api_key = _as_text(getattr(settings, "openrouter_api_key", ""))
    llm_api_key = _as_text(getattr(settings, "llm_api_key", ""))
    llm_base_url = _as_text(getattr(settings, "llm_base_url", ""))
    vlm_api_base = _as_text(getattr(settings, "vlm_api_base", ""))
    llm_model = _as_text(getattr(settings, "llm_model", ""))
    vlm_model = _as_text(getattr(settings, "vlm_model", ""))

    # BYOK handling
    if header_provider:
        provider = _normalize_provider(header_provider)
        if header_provider.strip().lower() == "google":
            provider = "google"
    elif provider_override:
        provider = _normalize_provider(provider_override)
    elif mode == "vision" and bool(getattr(settings, "vlm_enabled", False)):
        provider = "lmstudio"
    elif openrouter_api_key:
        provider = "openrouter"
    elif openai_api_key and openai_api_key != "sk-your-key-here":
        provider = "openai"
    else:
        provider = "lmstudio"

    if provider == "google":
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        api_key = header_api_key or _as_text(getattr(settings, "google_api_key", ""))
        model = header_model or "gemini-2.5-flash"
        supports_json_mode = True
    elif provider == "openai":
        base_url = _as_text(getattr(settings, "openai_base_url", "https://api.openai.com/v1")) or "https://api.openai.com/v1"
        api_key = header_api_key or openai_api_key
        model = header_model or _as_text(getattr(settings, "openai_model", "gpt-4o-mini")) or "gpt-4o-mini"
        supports_json_mode = True
    elif provider == "openrouter":
        base_url = _as_text(getattr(settings, "openrouter_base_url", "https://openrouter.ai/api/v1")) or "https://openrouter.ai/api/v1"
        api_key = header_api_key or openrouter_api_key
        model = header_model or _as_text(getattr(settings, "openrouter_model", "google/gemma-4-31b-it:free")) or "google/gemma-4-31b-it:free"
        supports_json_mode = True
    else:
        base_url = header_base_url or llm_base_url or vlm_api_base
        api_key = header_api_key or llm_api_key or "lm-studio"
        model = header_model or llm_model or vlm_model
        supports_json_mode = False if provider in {"lmstudio", "ollama"} else True

    if mode == "vision" and provider not in {"openai", "google", "openrouter"}:
        supports_json_mode = False

    max_tokens = int(getattr(settings, "llm_max_tokens", 0) or 0) if mode == "text" else int(getattr(settings, "vlm_max_tokens", 0) or 0)
    if max_tokens <= 0:
        max_tokens = 1200 if mode == "text" else 4000

    return LLMConfig(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        supports_json_mode=supports_json_mode,
    )


def create_openai_client(
    mode: str = "text",
    header_provider: Optional[str] = None,
    header_api_key: Optional[str] = None,
    header_model: Optional[str] = None,
    header_base_url: Optional[str] = None,
):
    """Create an OpenAI client configured for the active provider."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package not installed") from exc

    config = resolve_llm_config(
        mode=mode, 
        header_provider=header_provider, 
        header_api_key=header_api_key, 
        header_model=header_model,
        header_base_url=header_base_url
    )
    import httpx
    return OpenAI(
        api_key=config.api_key, 
        base_url=config.base_url,
        timeout=httpx.Timeout(60.0, connect=5.0)
    )


def chat_completion(
    messages: list,
    *,
    mode: str = "text",
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
    response_format: Optional[dict[str, Any]] = None,
    header_provider: Optional[str] = None,
    header_api_key: Optional[str] = None,
    header_model: Optional[str] = None,
    header_base_url: Optional[str] = None,
):
    """Call the active OpenAI-compatible chat completion endpoint."""
    import openai
    
    client = create_openai_client(
        mode=mode, 
        header_provider=header_provider, 
        header_api_key=header_api_key, 
        header_model=header_model,
        header_base_url=header_base_url
    )
    config = resolve_llm_config(
        mode=mode,
        header_provider=header_provider, 
        header_api_key=header_api_key, 
        header_model=header_model,
        header_base_url=header_base_url
    )
    request_kwargs: dict[str, Any] = {
        "model": model or config.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens or config.max_tokens,
    }
    if response_format is not None and config.supports_json_mode:
        request_kwargs["response_format"] = response_format
        
    try:
        return client.chat.completions.create(**request_kwargs)
    except openai.NotFoundError as e:
        raise ValueError(f"Model '{request_kwargs['model']}' not found on provider '{config.provider}'. Please check your model name.") from e
    except openai.APIConnectionError as e:
        raise ValueError(f"Could not connect to {config.provider} at {config.base_url}. Is the service running?") from e
