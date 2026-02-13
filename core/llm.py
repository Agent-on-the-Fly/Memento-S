"""LLM API client functions for OpenRouter and Anthropic-compatible endpoints."""

import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from core.config import (
    LLM_API,
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
                print(
                    f"[Rate limit {exc.code}] Retrying in {wait_time}s... "
                    f"(attempt {attempt}/{retries})"
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
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Missing OPENROUTER_API_KEY in environment")
    base = _normalize_openrouter_base(OPENROUTER_BASE_URL)
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
        "model": MODEL,
        "max_tokens": OPENROUTER_MAX_TOKENS,
        "messages": oai_messages,
    }
    provider_order: list[str] = []
    if OPENROUTER_PROVIDER_ORDER:
        provider_order = [p.strip() for p in OPENROUTER_PROVIDER_ORDER.split(",") if p.strip()]
    elif OPENROUTER_PROVIDER:
        provider_order = [OPENROUTER_PROVIDER]
    if provider_order:
        payload["provider"] = {
            "order": provider_order,
            "allow_fallbacks": OPENROUTER_ALLOW_FALLBACKS,
        }

    data = json.dumps(payload).encode("utf-8")
    headers: dict[str, str] = {
        "content-type": "application/json",
        "authorization": f"Bearer {OPENROUTER_API_KEY}",
    }
    if OPENROUTER_SITE_URL:
        headers["HTTP-Referer"] = OPENROUTER_SITE_URL
    if OPENROUTER_APP_NAME:
        headers["X-Title"] = OPENROUTER_APP_NAME

    raw = _http_request_with_retry(url, data, headers, provider_label="OpenRouter API")

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
    provider = (LLM_API or "").strip().lower()
    log_event(
        "llm_request",
        provider=provider or "openrouter",
        model=MODEL,
        system=system,
        messages=messages,
    )
    if provider in {"openrouter", "openai"}:
        out = _openrouter_chat_completions(system, messages)
        log_event("llm_response", provider="openrouter", model=MODEL, output=out)
        return out
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Missing OPENROUTER_API_KEY in environment")
    base = OPENROUTER_BASE_URL.rstrip("/")
    url = f"{base}/v1/messages"
    payload = {
        "model": MODEL,
        "max_tokens": OPENROUTER_MAX_TOKENS,
        "system": system,
        "messages": messages,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "content-type": "application/json",
        "x-api-key": OPENROUTER_API_KEY,
        "anthropic-version": "2023-06-01",
    }

    raw = _http_request_with_retry(url, data, headers, provider_label="Anthropic API")

    out = json.loads(raw or "{}")
    parts = out.get("content", [])
    text = "".join(p.get("text", "") for p in parts if p.get("type") == "text")
    log_event("llm_response", provider="anthropic", model=MODEL, output=text)
    return text
