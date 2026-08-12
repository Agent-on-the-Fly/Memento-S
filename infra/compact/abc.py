# SPDX-License-Identifier: Apache-2.0
"""核心抽象接口定义 — 依赖注入协议。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable, TypeVar

if TYPE_CHECKING:
    from infra.compact.models import CompactBudget, CompactResult, CompactTrigger


@runtime_checkable
class LLMClient(Protocol):
    """LLM 调用协议 — 支持任意实现 (OpenAI, Anthropic, 本地模型等)。"""

    async def chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        model: str = "",
    ) -> str:
        """调用 LLM 返回文本内容。

        Args:
            system: system prompt
            messages: 对话消息列表
            max_tokens: 最大返回 token 数
            model: 模型名称 (可选)

        Returns:
            LLM 生成的文本

        Raises:
            Exception: LLM 调用失败时抛出异常
        """
        ...


class TokenCounter(Protocol):
    """Token 计数协议 — 支持任意 tokenizer 实现。"""

    def count_text(self, text: str, *, model: str = "") -> int:
        """计算纯文本的 token 数。"""
        ...

    def count_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str = "",
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        """计算消息列表的 token 数。"""
        ...

    def estimate_fast(self, text: str) -> int:
        """O(1) 快速估算 (无需 tokenizer)。"""
        ...


@runtime_checkable
class StorageBackend(Protocol):
    """存储后端协议 — 支持任意存储实现 (文件系统、S3、内存等)。"""

    async def persist(self, content: str, metadata: dict[str, Any]) -> str:
        """持久化内容，返回引用 ID。

        Args:
            content: 要持久化的内容
            metadata: 元数据 (tool_call_id, tool_name 等)

        Returns:
            引用 ID (用于后续 recall)
        """
        ...

    async def recall(self, ref_id: str) -> str | None:
        """根据引用 ID 召回完整内容。"""
        ...


@runtime_checkable
class SummaryReader(Protocol):
    """Summary 读取协议 — 用于 SM Compact 读取外部摘要。"""

    def get_summary_content(self) -> str:
        """获取摘要内容 (如 summary.md)。"""
        ...

    def is_empty(self) -> bool:
        """摘要是否为空。"""
        ...

    def get_last_summarized_seq(self) -> int:
        """获取最后摘要的消息序列号。"""
        ...

    def truncate_for_compact(self) -> tuple[str, bool]:
        """获取截断后的摘要内容。

        Returns:
            (摘要内容, 是否成功)
        """
        ...


class CompactEngine(ABC):
    """压缩引擎抽象基类 — 子类可覆盖默认策略组合。"""

    @abstractmethod
    async def compact(
        self,
        messages: list[dict[str, Any]],
        budget: "CompactBudget",
        config: Any,
    ) -> "CompactResult":
        """执行压缩。

        Args:
            messages: 原始消息列表
            budget: 压缩预算
            config: 配置对象

        Returns:
            CompactResult 包含压缩后的消息
        """
        ...

    @abstractmethod
    async def should_compact(
        self,
        messages: list[dict[str, Any]],
        budget: "CompactBudget",
    ) -> "CompactTrigger | None":
        """判断是否需要压缩。

        Returns:
            CompactTrigger 如果需要压缩，否则 None
        """
        ...


class CompactObserver(ABC):
    """压缩回调 — 用于日志、监控、事件追踪等。"""

    @abstractmethod
    async def on_before_compact(
        self,
        messages: list[dict[str, Any]],
        budget: "CompactBudget",
        trigger: "CompactTrigger",
    ) -> None:
        """压缩前回调。"""
        ...

    @abstractmethod
    async def on_after_compact(
        self,
        result: "CompactResult",
        tokens_before: int,
        tokens_after: int,
    ) -> None:
        """压缩后回调。"""
        ...
