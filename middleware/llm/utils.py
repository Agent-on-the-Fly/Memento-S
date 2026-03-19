"""LLM 工具函数。

提供 LLM 调用封装和 LLM 输出检测。
"""
from __future__ import annotations

import asyncio
from typing import Any

from utils.logger import get_logger
from middleware.llm import LLMClient

logger = get_logger(__name__)

_llm: LLMClient | None = None


def _get_llm() -> LLMClient:
    """获取 LLM 实例（延迟初始化）"""
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm


def chat_completions(system: str, messages: list[dict[str, Any]]) -> str:
    """同步 LLM chat completion，在无 event loop 时使用 asyncio.run，否则用线程池。"""

    async def _call() -> str:
        resp = await _get_llm().async_chat(messages=messages, system=system)
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


async def chat_completions_async(
    system: str,
    messages: list[dict[str, Any]],
    max_tokens: int | None = None,
) -> str:
    """异步 LLM chat completion。"""
    kwargs: dict[str, Any] = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    resp = await _get_llm().async_chat(messages=messages, system=system, **kwargs)
    return resp.content or ""


def looks_like_tool_call_text(content: str) -> bool:
    """检测 LLM 文本输出是否像未成功解析的 tool call。

    只匹配模型特有的控制 token，避免对正常 JSON 误判。
    """
    if not content:
        return False
    _CONTROL_TOKENS = (
        "<|tool_calls_section_begin|>",
        "<|tool_call_begin|>",
        "<|plugin|>",
        "<function=",
    )
    return any(token in content for token in _CONTROL_TOKENS)
