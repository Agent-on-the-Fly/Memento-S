"""LLM API client functions for OpenRouter and Anthropic-compatible endpoints."""

from contextvars import ContextVar
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from core.config import (
    LLM_API,
    LLM_ENFORCE_CALL_BUDGET,
    LLM_MAX_CALLS_PER_TURN,
    MODEL,
    OPENROUTER_ALLOW_FALLBACKS,
    OPENROUTER_API_KEY,
    OPENROUTER_APP_NAME,
    OPENROUTER_BASE_URL,
    OPENROUTER_MAX_TOKENS,
    OPENROUTER_PROVIDER,
    OPENROUTER_PROVIDER_ORDER,
    OPENROUTER_RETRIES,
    OPENROUTER_RETRY_BACKOFF,
    OPENROUTER_SITE_URL,
    OPENROUTER_TIMEOUT,
)
from core.utils.logging_utils import log_event

_LLM_CALL_BUDGET: ContextVar[int | None] = ContextVar("llm_call_budget", default=None)


def _runtime_str(name: str, fallback: str | None = None) -> str:
    raw = os.getenv(name)
    if raw is None:
        return str(fallback or "").strip()
    return str(raw).strip()


def _runtime_int(name: str, fallback: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(fallback)
    try:
        return int(str(raw).strip())
    except Exception:
        return int(fallback)


def _runtime_float(name: str, fallback: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(fallback)
    try:
        return float(str(raw).strip())
    except Exception:
        return float(fallback)


def _runtime_flag(name: str, fallback: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(fallback)
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def reset_llm_call_budget(limit: int | None = None) -> None:
    """Reset per-turn LLM call budget."""
    if not _runtime_flag("LLM_ENFORCE_CALL_BUDGET", LLM_ENFORCE_CALL_BUDGET):
        _LLM_CALL_BUDGET.set(None)
        return
    if limit is None:
        limit = max(1, _runtime_int("LLM_MAX_CALLS_PER_TURN", LLM_MAX_CALLS_PER_TURN))
    _LLM_CALL_BUDGET.set(max(0, int(limit)))


def get_llm_call_budget() -> int | None:
    return _LLM_CALL_BUDGET.get()


def _consume_llm_call_budget() -> None:
    if not _runtime_flag("LLM_ENFORCE_CALL_BUDGET", LLM_ENFORCE_CALL_BUDGET):
        return
    remaining = _LLM_CALL_BUDGET.get()
    if remaining is None:
        remaining = max(1, _runtime_int("LLM_MAX_CALLS_PER_TURN", LLM_MAX_CALLS_PER_TURN))
    if remaining <= 0:
        raise RuntimeError("LLM call budget exceeded for current turn")
    _LLM_CALL_BUDGET.set(remaining - 1)


# ---------------------------------------------------------------------------
# Shared HTTP retry helper
# ---------------------------------------------------------------------------

def _http_request_with_retry(
    url: str,
    data: bytes,
    headers: dict[str, str],
    *,
    method: str = "POST",
    retries: int = OPENROUTER_RETRIES,
    backoff: float = OPENROUTER_RETRY_BACKOFF,
    timeout: int = OPENROUTER_TIMEOUT,
    provider_label: str = "API",
) -> str:
    """Send an HTTP request with retry logic for rate limits and transient errors.

    Rebuilds the ``urllib.request.Request`` on each attempt because the
    request body is consumed when an ``HTTPError`` is read.

    Returns the raw response body as a string.
    """
    last_exc: Exception | None = None
    raw: str = ""
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
            last_exc = None
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8") if exc.fp else ""
            if exc.code in (429, 500, 502, 503, 529) and attempt < retries:
                wait_time = backoff * attempt * 2
                log_event(
                    "llm_retry_wait",
                    provider=provider_label,
                    http_status=exc.code,
                    attempt=attempt,
                    retries=retries,
                    wait_time_sec=wait_time,
                )
                time.sleep(wait_time)
                continue
            raise RuntimeError(f"{provider_label} error {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError, ssl.SSLError) as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(backoff * attempt)
                continue
            raise RuntimeError(
                f"{provider_label} request failed: {type(exc).__name__}: {exc}"
            ) from exc
    if last_exc is not None:
        raise RuntimeError(f"{provider_label} request failed") from last_exc
    return raw


def _normalize_openrouter_base(url: str) -> str:
    """Normalize an OpenRouter base URL to end with /api/v1."""
    base = (url or "").strip().rstrip("/")
    if not base:
        return "https://openrouter.ai/api/v1"
    if base.endswith("/api"):
        return base + "/v1"
    if base.endswith("/api/v1"):
        return base
    if base.endswith("openrouter.ai"):
        return base + "/api/v1"
    return base


def _openrouter_chat_completions(system: str, messages: list[dict]) -> str:
    """Send a chat completion request via the OpenRouter API."""
    model = _runtime_str("OPENROUTER_MODEL", MODEL)
    api_key = _runtime_str("OPENROUTER_API_KEY", OPENROUTER_API_KEY or "")
    base_url = _runtime_str("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL)
    max_tokens = _runtime_int("OPENROUTER_MAX_TOKENS", OPENROUTER_MAX_TOKENS)
    provider = _runtime_str("OPENROUTER_PROVIDER", OPENROUTER_PROVIDER)
    provider_order_raw = _runtime_str("OPENROUTER_PROVIDER_ORDER", OPENROUTER_PROVIDER_ORDER)
    allow_fallbacks = _runtime_flag("OPENROUTER_ALLOW_FALLBACKS", OPENROUTER_ALLOW_FALLBACKS)
    site_url = _runtime_str("OPENROUTER_SITE_URL", OPENROUTER_SITE_URL)
    app_name = _runtime_str("OPENROUTER_APP_NAME", OPENROUTER_APP_NAME)
    retries = _runtime_int("OPENROUTER_RETRIES", OPENROUTER_RETRIES)
    backoff = _runtime_float("OPENROUTER_RETRY_BACKOFF", OPENROUTER_RETRY_BACKOFF)
    timeout = _runtime_int("OPENROUTER_TIMEOUT", OPENROUTER_TIMEOUT)

    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY in environment")
    base = _normalize_openrouter_base(base_url)
    url = f"{base}/chat/completions"

    oai_messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, (str, bytes)):
            text = content.decode("utf-8") if isinstance(content, bytes) else content
        else:
            text = json.dumps(content, ensure_ascii=False) if content is not None else ""
        oai_messages.append({"role": role, "content": text})

    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": oai_messages,
    }
    provider_order: list[str] = []
    if provider_order_raw:
        provider_order = [p.strip() for p in provider_order_raw.split(",") if p.strip()]
    elif provider:
        provider_order = [provider]
    if provider_order:
        payload["provider"] = {
            "order": provider_order,
            "allow_fallbacks": allow_fallbacks,
        }

    data = json.dumps(payload).encode("utf-8")
    headers: dict[str, str] = {
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}",
    }
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name

    raw = _http_request_with_retry(
        url,
        data,
        headers,
        retries=retries,
        backoff=backoff,
        timeout=timeout,
        provider_label="OpenRouter API",
    )

    out = json.loads(raw or "{}")
    choices = out.get("choices") or []
    if not choices:
        return ""
    msg = (choices[0] or {}).get("message") or {}
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                t = part.get("text")
                if isinstance(t, str):
                    parts.append(t)
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content)


def openrouter_messages(system: str, messages: list[dict]) -> str:
    """Send messages to the configured LLM provider.

    This is the primary LLM entry point used throughout the agent.
    It dispatches to the OpenRouter-style chat completions API or
    Anthropic Messages API depending
    on the ``LLM_API`` config value.
    """
    _consume_llm_call_budget()
    provider = _runtime_str("LLM_API", LLM_API).lower() or "openrouter"
    model = _runtime_str("OPENROUTER_MODEL", MODEL)
    log_event(
        "llm_request",
        provider=provider or "openrouter",
        model=model,
        system=system,
        messages=messages,
        llm_budget_remaining=get_llm_call_budget(),
    )
    if provider in {"openrouter", "openai"}:
        out = _openrouter_chat_completions(system, messages)
        log_event("llm_response", provider="openrouter", model=model, output=out)
        return out
    api_key = _runtime_str("OPENROUTER_API_KEY", OPENROUTER_API_KEY or "")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY in environment")
    base = _runtime_str("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL).rstrip("/")
    max_tokens = _runtime_int("OPENROUTER_MAX_TOKENS", OPENROUTER_MAX_TOKENS)
    retries = _runtime_int("OPENROUTER_RETRIES", OPENROUTER_RETRIES)
    backoff = _runtime_float("OPENROUTER_RETRY_BACKOFF", OPENROUTER_RETRY_BACKOFF)
    timeout = _runtime_int("OPENROUTER_TIMEOUT", OPENROUTER_TIMEOUT)
    url = f"{base}/v1/messages"
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "content-type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    raw = _http_request_with_retry(
        url,
        data,
        headers,
        retries=retries,
        backoff=backoff,
        timeout=timeout,
        provider_label="Anthropic API",
    )

    out = json.loads(raw or "{}")
    parts = out.get("content", [])
    text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
    log_event("llm_response", provider="anthropic", model=model, output=text)
    return text
