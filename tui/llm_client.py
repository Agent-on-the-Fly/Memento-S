
from __future__ import annotations

import asyncio
import logging
from typing import Any

from core.llm import LLM
from core.agent.prompts.templates import SUMMARIZE_CONVERSATION_PROMPT

logger = logging.getLogger(__name__)

_llm = LLM()


def chat_completions(system: str, messages: list[dict[str, Any]]) -> str:

    async def _call() -> str:
        resp = await _llm.chat(messages=messages, system=system)
        return resp.content or ""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, _call())
            return future.result()
    return asyncio.run(_call())


async def chat_completions_async(system: str, messages: list[dict[str, Any]]) -> str:
    resp = await _llm.chat(messages=messages, system=system)
    return resp.content or ""


def generate_title(messages: list[dict[str, Any]], max_length: int = 20) -> str:
    if not messages:
        return "New Conversation"

    recent = messages[-10:]
    snippet_parts: list[str] = []
    for msg in recent:
        role = msg.get("role", "unknown")
        text = _content_text(msg.get("content", ""))
        if text:
            snippet_parts.append(f"[{role}]: {text[:200]}")

    if not snippet_parts:
        return _fallback_title(messages, max_length)

    snippet = "\n".join(snippet_parts)
    prompt = (
        f"{max_length}"
        f"\n\n{snippet}"
    )

    try:
        title = chat_completions(
            "You are a concise title generator. Return only the title text.",
            [{"role": "user", "content": prompt}],
        ).strip().strip('"\'')

        if not title:
            return _fallback_title(messages, max_length)
        if len(title) > max_length:
            title = title[:max_length - 1] + "…"
        return title
    except Exception:
        logger.warning("generate_title failed, using fallback", exc_info=True)
        return _fallback_title(messages, max_length)


def _fallback_title(messages: list[dict[str, Any]], max_length: int = 20) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            text = _content_text(msg.get("content", ""))
            if text:
                text = text.replace("\n", " ").strip()
                if len(text) > max_length:
                    text = text[:max_length - 1] + "…"
                return text
    return "New Conversation"


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return " ".join(parts).strip()
    return ""


def summarize_context(messages: list[dict[str, Any]], max_tokens: int = 2000) -> str:
    context_parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        if role == "tool":
            context_parts.append(f"[TOOL_RESULT]: {content}")
        else:
            context_parts.append(f"[{role}]: {content}")

    full_context = "\n".join(context_parts)
    prompt = SUMMARIZE_CONVERSATION_PROMPT.format(
        max_tokens=max_tokens, context=full_context,
    )

    try:
        summary = chat_completions(
            "You are a precise summarizer. Return only the essential information. "
            "CRITICAL: Preserve [TOOL_RESULT] content (skill execution outputs) as completely as possible — "
            "these contain data the agent needs for subsequent steps.",
            [{"role": "user", "content": prompt}],
        )
        return summary.strip()
    except Exception:
        logger.warning("summarize_context failed, falling back to truncation", exc_info=True)
        max_chars = max_tokens * 6  #  fallback 
        if len(full_context) > max_chars:
            return full_context[:max_chars] + "...[truncated]"
        return full_context
