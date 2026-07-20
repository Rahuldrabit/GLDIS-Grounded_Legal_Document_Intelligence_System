from __future__ import annotations


def test_resolve_openrouter_when_explicit_provider():
    from core.config import get_settings
    from llm.client import resolve_llm_config

    settings = get_settings()
    old_provider = settings.llm_provider
    old_key = settings.openrouter_api_key
    old_model = settings.openrouter_model
    old_base = settings.openrouter_base_url

    settings.llm_provider = "openrouter"
    settings.openrouter_api_key = "or-test-key"
    settings.openrouter_model = "google/gemma-4-31b-it:free"
    settings.openrouter_base_url = "https://openrouter.ai/api/v1"

    try:
        cfg = resolve_llm_config(mode="text")
        assert cfg.provider == "openrouter"
        assert cfg.api_key == "or-test-key"
        assert cfg.model == "google/gemma-4-31b-it:free"
        assert cfg.base_url == "https://openrouter.ai/api/v1"
    finally:
        settings.llm_provider = old_provider
        settings.openrouter_api_key = old_key
        settings.openrouter_model = old_model
        settings.openrouter_base_url = old_base


def test_auto_selects_openrouter_when_key_present():
    from core.config import get_settings
    from llm.client import resolve_llm_config

    settings = get_settings()
    old_provider = settings.llm_provider
    old_key = settings.openrouter_api_key
    old_openai_key = settings.openai_api_key

    settings.llm_provider = ""
    settings.openrouter_api_key = "or-test-key"
    settings.openai_api_key = ""

    try:
        cfg = resolve_llm_config(mode="text")
        assert cfg.provider == "openrouter"
    finally:
        settings.llm_provider = old_provider
        settings.openrouter_api_key = old_key
        settings.openai_api_key = old_openai_key
