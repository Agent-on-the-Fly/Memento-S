"""Token counting — delegates to ``litellm.token_counter``.

litellm selects the correct tokenizer per model provider (OpenAI, Anthropic,
Moonshot/KIMI, …) automatically.  A character-based fallback covers the rare
case where litellm itself is unavailable.
"""

from __future__ import annotations

from typing import Any

from litellm import token_counter

from utils.logger import get_logger

logger = get_logger(__name__)


# ── Public API ─────────────────────────────────────────────────────


def count_tokens(text: str, model: str = "") -> int:
    """Count tokens for a plain text string."""
    if not text:
        return 0
    try:
        return token_counter(model=model, text=text)
    except Exception:
        return _estimate_fallback(text)


def count_tokens_messages(
    messages: list[dict[str, Any]],
    model: str = "",
    tools: list[dict[str, Any]] | None = None,
) -> int:
    """Count tokens for a message list (OpenAI format).

    Handles role overhead, content, tool_calls, and tools schema.
    """
    if not messages:
        return 0
    try:
        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        return token_counter(**kwargs)
    except Exception as e:
        logger.warning("litellm token_counter failed ({}), using fallback", e)
        return _estimate_messages_fallback(messages)


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
