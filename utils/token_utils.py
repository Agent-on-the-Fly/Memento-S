# SPDX-License-Identifier: Apache-2.0
"""Token counting — delegates to ``litellm.token_counter``.

litellm selects the correct tokenizer per model provider (OpenAI, Anthropic,
Moonshot/KIMI, …) automatically.  A character-based fallback covers the rare
case where litellm itself is unavailable.

Performance: count_tokens uses LRU cache keyed on (text_hash, model) to avoid
redundant tokenizer calls on the same content in a single pipeline pass.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from litellm import token_counter

from utils.logger import get_logger

logger = get_logger(__name__)

_CHAR_PER_TOKEN_APPROX = 4
"""Rough chars-per-token ratio for O(1) coarse estimation."""


# ── Public API ─────────────────────────────────────────────────────


def count_tokens(text: str, model: str = "") -> int:
    """Count tokens for a plain text string (LRU cached)."""
    if not text:
        return 0
    return _count_tokens_cached(text, model)


def estimate_tokens_fast(text: str) -> int:
    """O(1) coarse token estimate — no tokenizer call.

    Use for pre-checks to skip expensive precise counting on short text.
    """
    if not text:
        return 0
    return len(text) // _CHAR_PER_TOKEN_APPROX + 1


def count_tokens_messages(
    messages: list[dict[str, Any]],
    model: str = "",
    tools: list[dict[str, Any]] | None = None,
) -> int:
    """Count tokens for a message list (OpenAI format).

    Handles role overhead, content, tool_calls, and tools schema.
    Sanitizes tool_calls (converts ToolCall dataclass → dict) before
    passing to litellm to avoid "not iterable" errors from litellm itself.
    """
    if not messages:
        return 0
    try:
        # litellm.token_counter does not accept ToolCall dataclass objects.
        # Serialize them to dicts so litellm can process the messages.
        sanitized_messages = _sanitize_messages(messages)
        kwargs: dict[str, Any] = {"model": model, "messages": sanitized_messages}
        if tools:
            kwargs["tools"] = tools
        return token_counter(**kwargs)
    except Exception as e:
        logger.warning("litellm token_counter failed ({}), using fallback", e)
        return _estimate_messages_fallback(messages)


def _sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert ToolCall dataclass objects in tool_calls to plain dicts."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            result.append(msg)
            continue
        sanitized = dict(msg)
        tc_list = sanitized.get("tool_calls")
        if tc_list and isinstance(tc_list, list):
            sanitized_tc: list[dict[str, Any]] = []
            for tc in tc_list:
                if isinstance(tc, dict):
                    sanitized_tc.append(tc)
                else:
                    # Convert ToolCall (or similar) dataclass to dict
                    as_dict = {
                        "id": getattr(tc, "id", ""),
                        "type": getattr(tc, "type", "function"),
                        "function": dict(getattr(tc, "function", {}))
                        if hasattr(tc, "function")
                        else {},
                    }
                    sanitized_tc.append(as_dict)
            sanitized["tool_calls"] = sanitized_tc
        result.append(sanitized)
    return result


# ── Cached implementation ─────────────────────────────────────────


@lru_cache(maxsize=512)
def _count_tokens_cached(text: str, model: str) -> int:
    """Cached core — keyed on (text, model).

    lru_cache hashes the text string; for very long strings the hash
    itself is O(n) but only once per unique content, and Python's
    str.__hash__ is fast C code.
    """
    try:
        return token_counter(model=model, text=text)
    except Exception:
        return _estimate_fallback(text)


def clear_token_cache() -> None:
    """Clear the LRU cache. Call between sessions if memory is a concern."""
    _count_tokens_cached.cache_clear()


# ── Fallback estimation ────────────────────────────────────────────


def _estimate_fallback(text: str) -> int:
    """Character-based estimation for when litellm is unavailable."""
    if not text:
        return 0
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    cjk_chars = sum(
        1 for c in text if "\u4e00" <= c <= "\u9fff" or "\u3000" <= c <= "\u303f"
    )
    other_chars = len(text) - ascii_chars - cjk_chars
    return int(ascii_chars / 3.5 + cjk_chars * 1.8 + other_chars / 1.5)


def _estimate_messages_fallback(messages: list[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += _estimate_fallback(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += _estimate_fallback(part.get("text", ""))
        tc_list = msg.get("tool_calls")
        if tc_list:
            for tc in tc_list:
                func = (
                    tc.get("function")
                    if isinstance(tc, dict)
                    else getattr(tc, "function", None)
                )
                if func:
                    name = (
                        func.get("name", "")
                        if isinstance(func, dict)
                        else getattr(func, "name", "")
                    )
                    args = (
                        func.get("arguments", "")
                        if isinstance(func, dict)
                        else getattr(func, "arguments", "")
                    )
                    total += (
                        _estimate_fallback(str(name))
                        + _estimate_fallback(str(args))
                        + 3
                    )
        total += 4
    return total
