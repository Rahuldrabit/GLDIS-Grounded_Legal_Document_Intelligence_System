"""In-memory rate limiter for selected API endpoints."""
from __future__ import annotations

from collections import defaultdict, deque
from math import ceil
from threading import Lock
from time import monotonic
from typing import Callable

from fastapi import Depends, HTTPException, Request

from core.config import get_settings

_requests_by_key: dict[str, deque[float]] = defaultdict(deque)
_rate_limit_lock = Lock()


def _client_key(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    client = request.client.host if request.client else "unknown"
    return client or "unknown"


def _enforce_limit(scope: str, request: Request) -> None:
    settings = get_settings()
    if not bool(getattr(settings, "llm_rate_limit_enabled", True)):
        return

    max_requests = int(getattr(settings, "llm_rate_limit_requests", 30) or 30)
    window_seconds = int(getattr(settings, "llm_rate_limit_window_seconds", 60) or 60)
    if max_requests <= 0 or window_seconds <= 0:
        return

    now = monotonic()
    threshold = now - window_seconds
    key = f"{scope}:{_client_key(request)}"

    with _rate_limit_lock:
        history = _requests_by_key[key]
        while history and history[0] <= threshold:
            history.popleft()

        if len(history) >= max_requests:
            retry_after = max(1, ceil(window_seconds - (now - history[0])))
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please retry later.",
                headers={"Retry-After": str(retry_after)},
            )

        history.append(now)


def rate_limit_guard(scope: str = "llm") -> Callable[[Request], None]:
    def _guard(request: Request) -> None:
        _enforce_limit(scope=scope, request=request)

    return Depends(_guard)


def reset_rate_limiter_state() -> None:
    with _rate_limit_lock:
        _requests_by_key.clear()
