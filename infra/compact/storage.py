# SPDX-License-Identifier: Apache-2.0
"""存储抽象层 — 压缩模块的内容落盘接口。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from utils.logger import get_logger
from utils.token_utils import count_tokens

from .extract import smart_extract_content
from .utils import extract_key_content, estimate_tokens_fast
from infra.compact.constants import DEFAULT_FLOOR_TOKENS

logger = get_logger(__name__)

_FLOOR_TOKENS = DEFAULT_FLOOR_TOKENS

_FLOOR_TOKENS = 200


class FileStorageBackend:
    """文件系统存储后端 — 基于 session_dir/artifacts/。

    artifacts 存放在 session_dir/artifacts/{tool_call_id}.txt，
    天然随 session 目录生命周期管理，删 session 目录即清理。
    """

    def __init__(
        self,
        session_dir: Path,
        *,
        persist_ratio: float = 0.15,
        extract_ratio: float = 0.05,
        model: str = "",
    ) -> None:
        self._dir = session_dir / "artifacts"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._persist_ratio = persist_ratio
        self._extract_ratio = extract_ratio
        self._model = model

    async def persist(self, content: str, metadata: dict[str, Any]) -> str:
        """持久化内容到文件。"""
        tool_call_id = metadata.get("tool_call_id", "")
        if not tool_call_id:
            raise ValueError("tool_call_id required for persist")

        path = self._dir / f"{tool_call_id}.txt"
        try:
            await asyncio.to_thread(path.write_text, content, encoding="utf-8")
            logger.info(
                "Persisted artifact {} ({} chars)",
                tool_call_id,
                len(content),
            )
            return tool_call_id
        except OSError:
            logger.opt(exception=True).warning(
                "Failed to persist artifact {}", tool_call_id
            )
            raise

    async def recall(self, ref_id: str) -> str | None:
        """读取完整内容。"""
        path = self._dir / f"{ref_id}.txt"
        if not path.exists():
            return None
        try:
            return await asyncio.to_thread(path.read_text, encoding="utf-8")
        except OSError:
            logger.opt(exception=True).warning("Failed to read artifact {}", ref_id)
            return None

    async def has(self, ref_id: str) -> bool:
        """检查 artifact 是否存在。"""
        return (self._dir / f"{ref_id}.txt").exists()

    async def process_and_extract(
        self,
        tool_call_id: str,
        tool_name: str,
        content: str,
        *,
        remaining_budget_tokens: int = 0,
        model: str = "",
    ) -> tuple[str, bool]:
        """处理 tool result: 必要时落盘并返回提取后的内容。

        Returns:
            (提取后的内容, 是否已落盘)
        """
        content_tokens = count_tokens(content, model=model or self._model)

        # 计算落盘阈值
        if tool_name == "execute_skill":
            persist_threshold = max(
                _FLOOR_TOKENS, int(remaining_budget_tokens * 0.4)
            )
        else:
            persist_threshold = max(
                _FLOOR_TOKENS, int(remaining_budget_tokens * self._persist_ratio)
            )

        if content_tokens <= persist_threshold:
            return content, False

        # 落盘
        try:
            await self.persist(content, {"tool_call_id": tool_call_id})
        except Exception:
            return content, False

        # 提取 preview
        if tool_name == "execute_skill":
            extract_budget = max(
                _FLOOR_TOKENS, int(remaining_budget_tokens * 0.35)
            )
        else:
            extract_budget = max(
                _FLOOR_TOKENS, int(remaining_budget_tokens * self._extract_ratio)
            )

        preview = extract_key_content(content, extract_budget, model=model or self._model)
        replaced = (
            f"[Output persisted: {len(content)} chars / {content_tokens} tokens]\n"
            f"Preview:\n{preview}\n"
            f"(Use recall_context('{tool_call_id}') to retrieve full content.)"
        )
        return replaced, True


# ---------------------------------------------------------------------------
# ArtifactStore — 业务封装，兼容旧接口
# ---------------------------------------------------------------------------


class ArtifactStore:
    """Session-level artifact store backed by filesystem.

    兼容旧 core/context/artifacts.py 的接口。
    """

    def __init__(
        self,
        session_dir: Path,
        *,
        persist_ratio: float = 0.15,
        extract_ratio: float = 0.05,
        model: str = "",
    ) -> None:
        self._backend = FileStorageBackend(
            session_dir=session_dir,
            persist_ratio=persist_ratio,
            extract_ratio=extract_ratio,
            model=model,
        )
        self._model = model

    async def process_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        content: str,
        *,
        remaining_budget_tokens: int = 0,
    ) -> dict[str, Any]:
        """处理 tool result: 必要时落盘并返回提取后的 message。

        Returns:
            {"role": "tool", "content": ..., "tool_call_id": ...,
             "name": ..., "_persisted": bool}
        """
        base: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
        }

        replaced, persisted = await self._backend.process_and_extract(
            tool_call_id,
            tool_name,
            content,
            remaining_budget_tokens=remaining_budget_tokens,
            model=self._model,
        )

        return {**base, "content": replaced, "_persisted": persisted}

    async def read_artifact(self, tool_call_id: str) -> str | None:
        """读取完整 artifact 内容。"""
        return await self._backend.recall(tool_call_id)

    async def has_artifact(self, tool_call_id: str) -> bool:
        """检查 artifact 是否存在。"""
        return await self._backend.has(tool_call_id)

