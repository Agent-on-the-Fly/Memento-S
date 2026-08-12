# SPDX-License-Identifier: Apache-2.0
"""ArtifactProviderImpl — 文件系统存储的 Tool Result 持久化。

迁移自 core/context/artifacts.py 的 ArtifactStore。
无 core/ 依赖，LLM 客户端通过参数注入。
_smart_extract_content 委托 infra/shared/extract.smart_extract_content 处理（唯一真实来源）。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Awaitable, Callable

from utils.logger import get_logger
from utils.token_utils import count_tokens, estimate_tokens_fast

from infra.shared.extract import (
    _extract_structured,
    build_digest,
    extract_key_content,
    smart_extract_content as _smart_extract_content_impl,
)

logger = get_logger(__name__)

_FLOOR_TOKENS = 200

# LLM client type alias
LLMClient = Callable[..., Awaitable[str]]


def _default_llm_client(
    messages: list[dict[str, Any]],
    *,
    system: str = "",
    max_tokens: int = 800,
) -> Awaitable[str]:
    from middleware.llm.llm_client import chat_completions_async

    return chat_completions_async(
        system=system,
        messages=messages,
        max_tokens=max_tokens,
    )


async def _smart_extract_content(
    content: str,
    budget_tokens: int,
    *,
    model: str = "",
    tool_name: str = "",
    llm_client: LLMClient | None = None,
) -> str:
    """分层内容提取。委托 shared/extract.py 处理。"""
    return await _smart_extract_content_impl(
        content,
        budget_tokens,
        model=model,
        tool_name=tool_name,
        llm_client=llm_client,
    )


class ArtifactProviderImpl:
    """Session-level artifact store backed by filesystem.

    迁移自 core/context/artifacts.py 的 ArtifactStore。
    实现了 ArtifactProvider 接口。

    所有阈值 = remaining_budget_tokens * ratio，无固定 token 数字。
    """

    def __init__(
        self,
        session_dir: Path,
        *,
        persist_ratio: float = 0.15,
        extract_ratio: float = 0.05,
        model: str = "",
        llm_client: LLMClient | None = None,
    ) -> None:
        self._dir = session_dir / "artifacts"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._persist_ratio = persist_ratio
        self._extract_ratio = extract_ratio
        self._model = model
        self._llm_client = llm_client or _default_llm_client

    async def process_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        content: str,
        *,
        remaining_budget_tokens: int = 0,
    ) -> dict[str, Any]:
        """Dynamically persist large results to file, return extracted preview."""
        base: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
        }

        content_tokens = count_tokens(content, model=self._model)

        if tool_name == "execute_skill":
            persist_threshold = max(
                _FLOOR_TOKENS, int(remaining_budget_tokens * 0.4)
            )
        else:
            persist_threshold = max(
                _FLOOR_TOKENS, int(remaining_budget_tokens * self._persist_ratio)
            )

        if content_tokens <= persist_threshold:
            return {**base, "content": content, "_persisted": False}

        path = self._dir / f"{tool_call_id}.txt"
        try:
            await asyncio.to_thread(path.write_text, content, encoding="utf-8")
        except OSError:
            logger.opt(exception=True).warning(
                "Failed to persist artifact {}, inlining full content",
                tool_call_id,
            )
            return {**base, "content": content, "_persisted": False}

        if tool_name == "execute_skill":
            extract_budget = max(_FLOOR_TOKENS, int(remaining_budget_tokens * 0.35))
        else:
            extract_budget = max(
                _FLOOR_TOKENS, int(remaining_budget_tokens * self._extract_ratio)
            )
        preview = await _smart_extract_content(
            content, extract_budget, model=self._model, tool_name=tool_name,
            llm_client=self._llm_client,
        )

        replaced = (
            f"[Output persisted: {len(content)} chars / {content_tokens} tokens]\n"
            f"Preview:\n{preview}\n"
            f"(Use recall_context('{tool_call_id}') to retrieve full content.)"
        )
        logger.info(
            "Persisted artifact {} ({} tokens) for tool {}",
            tool_call_id, content_tokens, tool_name,
        )
        return {**base, "content": replaced, "_persisted": True}

    async def read_artifact(self, tool_call_id: str) -> str | None:
        """Read full artifact content from file (non-blocking)."""
        path = self._dir / f"{tool_call_id}.txt"
        if not path.exists():
            return None
        try:
            return await asyncio.to_thread(path.read_text, encoding="utf-8")
        except OSError:
            logger.opt(exception=True).warning("Failed to read artifact {}", tool_call_id)
            return None

    async def has_artifact(self, tool_call_id: str) -> bool:
        """Check if artifact exists on disk."""
        return await asyncio.to_thread(lambda: (self._dir / f"{tool_call_id}.txt").exists())
