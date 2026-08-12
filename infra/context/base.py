# SPDX-License-Identifier: Apache-2.0
"""ContextProvider — Context 唯一抽象接口。

所有 Context 实现必须实现此接口，使 Context 管理与 Agent 解耦。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ContextProvider(ABC):
    """Context 唯一抽象接口。

    提供统一的上下文管理能力，包括消息组装、System Prompt 构建、
    Pre-API Pipeline、History 管理等。
    """

    # ── 消息初始化 ────────────────────────────────────────────────────

    @abstractmethod
    async def load_and_assemble(
        self,
        current_message: str,
        *,
        history: list[dict[str, Any]] | None = None,
        media: list[str] | None = None,
        matched_skills_context: str = "",
        agent_profile: Any = None,
        session_context: Any = None,
        mode: str = "agentic",
        intent_shifted: bool = False,
        effective_context_window: int | None = None,
        memory_query: str = "",
    ) -> list[dict[str, Any]]:
        """统一消息初始化。

        Args:
            current_message: 当前用户消息
            history: 可选的历史消息（默认从 DB 加载）
            media: 可选的媒体附件
            matched_skills_context: 匹配技能上下文
            agent_profile: Agent 配置
            session_context: Session 级别上下文
            mode: 运行模式（agentic / direct / interrupt）
            intent_shifted: Intent 是否转移
            effective_context_window: 有效上下文窗口大小
            memory_query: 用于 L2 topic 关键词检索的查询词（可选，默认用 current_message）

        Returns:
            完整的消息列表 [system, ...history, user]
        """
        ...

    # ── 消息追加 ─────────────────────────────────────────────────────

    @abstractmethod
    async def append(
        self,
        messages: list[dict[str, Any]],
        new_msgs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """追加消息到当前上下文。

        Args:
            messages: 当前消息列表
            new_msgs: 待追加的新消息

        Returns:
            追加后的消息列表
        """
        ...

    # ── Pre-API Pipeline ─────────────────────────────────────────────

    @abstractmethod
    async def prepare_for_api(
        self,
        messages: list[dict[str, Any]],
        *,
        state_messages: list[dict[str, Any]] | None = None,
        force_compact: bool = False,
    ) -> list[dict[str, Any]]:
        """Pre-API Pipeline 入口。

        在每次 LLM 调用前执行：
        - Microcompact：精简旧 tool result
        - Budget Guard：token 超限时触发压缩
        - SM Compact：Session Memory 截断

        Args:
            messages: 当前消息列表
            state_messages: 状态消息（可选）
            force_compact: 是否强制压缩

        Returns:
            处理后的消息列表
        """
        ...

    # ── System Prompt ─────────────────────────────────────────────────

    @abstractmethod
    async def assemble_system_prompt(
        self,
        *,
        mode: str = "agentic",
        intent_shifted: bool = False,
        matched_skills_context: str = "",
        agent_profile: Any = None,
        session_context: Any = None,
        memory_query: str = "",
    ) -> str:
        """构造 System Prompt。

        包含：Identity、Behavior、Protocol、Skills、L1 Session Memory、
        L2 Memory 等各节。

        Args:
            mode: 运行模式
            intent_shifted: Intent 是否转移
            matched_skills_context: 匹配技能上下文
            agent_profile: Agent 配置
            session_context: Session 级别上下文
            memory_query: 用于 L2 topic 关键词检索的查询词（可选）

        Returns:
            完整的 system prompt 字符串
        """
        ...

    # ── History ───────────────────────────────────────────────────────

    @abstractmethod
    async def load_history(self) -> list[dict[str, Any]]:
        """从 DB 加载历史消息。"""
        ...

    @abstractmethod
    def build_history_summary(
        self,
        history: list[dict[str, Any]] | None,
        max_rounds: int = 3,
        max_tokens: int = 800,
    ) -> str:
        """构建简短历史摘要。

        用于 Intent 识别等场景。

        Args:
            history: 历史消息列表
            max_rounds: 最大保留轮数
            max_tokens: 最大 token 数

        Returns:
            摘要字符串
        """
        ...

    # ── Token Budget ────────────────────────────────────────────────

    @abstractmethod
    def init_budget(self, context_max_tokens: int) -> None:
        """初始化 Token 预算。

        所有 token 阈值从 context_max_tokens * ratio 动态计算。

        Args:
            context_max_tokens: 上下文最大 token 数
        """
        ...

    @abstractmethod
    def sync_tokens(self, messages: list[dict[str, Any]] | None = None) -> None:
        """同步 Token 计数。

        Args:
            messages: 可选的消息列表（用于重新计算）
        """
        ...

    @property
    @abstractmethod
    def total_tokens(self) -> int:
        """当前总 token 数。"""
        ...

    # ── Tool Result Persistence ────────────────────────────────────

    @abstractmethod
    async def persist_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> dict[str, Any]:
        """持久化 Tool Result。

        委托 ArtifactStore 处理大结果。

        Args:
            tool_call_id: Tool Call ID
            tool_name: Tool 名称
            result: Tool 执行结果

        Returns:
            处理后的 tool result 消息
        """
        ...

    # ── Session Memory (L1) ─────────────────────────────────────────

    @property
    @abstractmethod
    def session_memory(self) -> Any:
        """获取 L1 Session Memory 实例。"""
        ...

    # ── L2 Memory ────────────────────────────────────────────────────

    @property
    @abstractmethod
    def context_memory(self) -> Any:
        """获取 L2 Context Memory 实例（已迁移至 InfraService 层，此处恒为 None）。"""
        ...

    # ── Compactor ─────────────────────────────────────────────────────

    @property
    @abstractmethod
    def compactor(self) -> Any:
        """获取预压缩流水线 (CompactorProvider)。"""
        ...

    # ── Artifact Provider ─────────────────────────────────────────

    @property
    @abstractmethod
    def artifact_provider(self) -> Any:
        """获取 Tool Result 持久化 Provider (ArtifactProvider)。"""
        ...

    # ── Force Compact ──────────────────────────────────────────────

    @abstractmethod
    async def force_compact_now(
        self,
        history_loader: Any,
    ) -> tuple[int, int, str]:
        """立即压缩历史上下文（Agent 外部触发）。

        Args:
            history_loader: 可调用的历史加载器

        Returns:
            (压缩前 token 数, 压缩后 token 数, preview 摘要)
        """
        ...

    # ── Stats / 可观测性 ─────────────────────────────────────────────

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """获取统计信息（用于监控和调试）。

        Returns:
            统计信息字典
        """
        ...
