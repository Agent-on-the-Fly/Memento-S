# SPDX-License-Identifier: Apache-2.0
"""FileContextProvider — 基于文件的 ContextProvider 实现。

使用 infra/ 内部实现（SessionMemory, LongTermMemory,
TokenBudgetCompactor, ArtifactProviderImpl），无 core/ 依赖。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from infra.context.base import ContextProvider
from infra.context.providers.artifact_impl import ArtifactProviderImpl
from infra.context.providers.compactor_impl import TokenBudgetCompactor
from infra.memory.impl.session_memory import SessionMemory
from infra.memory.impl.long_term_memory import LongTermMemory
from utils.logger import get_logger

logger = get_logger(__name__)


class FileContextProvider(ContextProvider):
    """基于文件的 ContextProvider。

    直接组合 infra/ 内部实现，无 core/ 依赖：
    - SessionMemory — L1 Session Memory
    - LongTermMemory — L2 长期记忆（随 FileContextProvider 构造）
    - TokenBudgetCompactor — Pre-API Pipeline
    - ArtifactProviderImpl — Tool Result 持久化
    """

    def __init__(
        self,
        session_dir: Path | str,
        data_dir: Path | str,
        model: str = "",
        *,
        context_dir: Path | str | None = None,
        context_max_tokens: int = 0,
        history_load_limit: int = 20,
        recent_rounds_keep: int = 3,
        history_budget_ratio: float = 0.5,
        summary_ratio: float = 0.15,
        persist_ratio: float = 0.15,
        extract_ratio: float = 0.05,
        preview_ratio: float = 0.005,
        slim_ratio: float = 0.003,
        microcompact_keep_recent: int = 5,
        microcompact_compactable_tools: list[str] | None = None,
        emergency_keep_tail: int = 6,
        max_compact_failures: int = 3,
        sm_compact_min_ratio: float = 0.02,
        sm_compact_max_ratio: float = 0.08,
        breaker_cooldown_s: float = 60.0,
        sm_llm_update_interval: int = 5,
        embedding_enabled: bool = False,
        history_loader: Any | None = None,
        skill_gateway: Any | None = None,
    ) -> None:
        if isinstance(session_dir, str):
            session_dir = Path(session_dir)
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        if context_dir is None:
            context_dir = session_dir.parent.parent  # session_dir = context_dir/sessions/{id}
        elif isinstance(context_dir, str):
            context_dir = Path(context_dir)

        self._session_dir = session_dir
        self._data_dir = data_dir
        self._context_dir = context_dir
        self._model = model
        self._context_max_tokens = context_max_tokens
        self._history_load_limit = history_load_limit
        self._recent_rounds_keep = recent_rounds_keep
        self._history_budget_ratio = history_budget_ratio
        self._summary_ratio = summary_ratio
        self._embedding_enabled = embedding_enabled
        self._history_loader = history_loader
        self._skill_gateway = skill_gateway

        # L1 Session Memory
        self._session_memory = SessionMemory(
            session_dir=session_dir,
            model=model,
            llm_update_interval=sm_llm_update_interval,
        )

        # Artifact Store
        self._artifact_provider = ArtifactProviderImpl(
            session_dir=session_dir,
            persist_ratio=persist_ratio,
            extract_ratio=extract_ratio,
            model=model,
        )

        # Compactor (without session_memory injected yet — will set after init)
        compactable_tools = set(microcompact_compactable_tools or [
            "execute_skill", "search_skill", "read_file", "bash",
        ])

        budget = context_max_tokens
        sm_min = int(budget * sm_compact_min_ratio) if budget else 800
        sm_max = int(budget * sm_compact_max_ratio) if budget else 3000

        self._compactor = TokenBudgetCompactor(
            model=model,
            context_budget=budget,
            microcompact_keep_recent=microcompact_keep_recent,
            microcompact_compactable_tools=compactable_tools,
            emergency_keep_tail=emergency_keep_tail,
            max_compact_failures=max_compact_failures,
            sm_compact_min_tokens=sm_min,
            sm_compact_max_tokens=sm_max,
            artifact_provider=self._artifact_provider,
            session_memory=self._session_memory,
            breaker_cooldown_s=breaker_cooldown_s,
            pipeline_preview_budget=int(budget * preview_ratio) if budget else 500,
        )

        # L2 Long Term Memory（跨 session）
        long_memory_dir = context_dir / "memory"
        long_memory_dir.mkdir(parents=True, exist_ok=True)
        self._context_memory = LongTermMemory(long_memory_dir, model=model)

        # Token tracking
        self._total_tokens: int = 0

    # ── Compactor ─────────────────────────────────────────────────────

    @property
    def compactor(self) -> Any:
        return self._compactor

    # ── Artifact Provider ─────────────────────────────────────────

    @property
    def artifact_provider(self) -> Any:
        return self._artifact_provider

    # ── Force Compact ──────────────────────────────────────────────

    async def force_compact_now(
        self,
        history_loader: Any = None,
    ) -> tuple[int, int, str]:
        """立即压缩历史上下文（Agent 外部触发）。"""
        return await self._compactor.force_compact_now(
            history_loader=history_loader,
            model=self._model,
            summary_tokens=int(self._context_max_tokens * self._summary_ratio)
            if self._context_max_tokens else 3000,
        )

    # ── 消息初始化 ────────────────────────────────────────────────────

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
        """统一消息初始化。"""
        # 1. 加载历史
        if history is None:
            history = await self.load_history()

        # 2. Token 同步
        self.sync_tokens(history)

        # 3. System prompt
        system_prompt = await self.assemble_system_prompt(
            mode=mode,
            intent_shifted=intent_shifted,
            matched_skills_context=matched_skills_context,
            agent_profile=agent_profile,
            session_context=session_context,
            memory_query=memory_query or current_message,
        )

        # 4. Budget-aware 历史裁剪
        if self._context_max_tokens > 0:
            budget = self._history_budget_ratio * self._context_max_tokens
            history = self._budget_aware_trim(history, int(budget))

        # 5. 组装消息
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": current_message},
        ]

        return messages

    def _budget_aware_trim(
        self,
        messages: list[dict[str, Any]],
        token_budget: int,
    ) -> list[dict[str, Any]]:
        """Token 预算感知的消息裁剪。"""
        if not messages or token_budget <= 0:
            return messages

        from utils.token_utils import count_tokens_messages

        total = count_tokens_messages(messages, model=self._model)
        if total <= token_budget:
            return messages

        # 优先保留最近轮
        from infra.context.providers.shared_utils import group_aware_trim

        # 保留最近 3 轮
        return group_aware_trim(messages, self._recent_rounds_keep)

    # ── 消息追加 ─────────────────────────────────────────────────────

    async def append(
        self,
        messages: list[dict[str, Any]],
        new_msgs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """追加消息到当前上下文。"""
        self.sync_tokens(messages)
        return messages + new_msgs

    # ── Pre-API Pipeline ─────────────────────────────────────────────

    async def prepare_for_api(
        self,
        messages: list[dict[str, Any]],
        *,
        state_messages: list[dict[str, Any]] | None = None,
        force_compact: bool = False,
    ) -> list[dict[str, Any]]:
        """Pre-API Pipeline 入口。"""
        working, was_compacted, trigger, tokens_before, tokens_after = (
            await self._compactor.prepare_for_api(
                messages,
                state_messages_ref=state_messages,
                force_compact=force_compact,
            )
        )
        self._total_tokens = tokens_after
        return working

    # ── System Prompt ─────────────────────────────────────────────────

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
        """构造 System Prompt。"""
        from utils.token_utils import count_tokens

        parts: list[str] = []

        # AgentProfile — capabilities and identity (priority 30, before matched_skills)
        if agent_profile is not None:
            profile_section = agent_profile.to_prompt_section()
            if profile_section:
                parts.append(profile_section)

        # Matched Skills (priority 40, before SM so skills are visible early)
        if matched_skills_context:
            parts.append(matched_skills_context)

        # L1 Session Memory (priority 50)
        sm_section = self._session_memory.to_prompt_section()
        if sm_section:
            parts.append(sm_section)
            logger.debug("SM prompt injected: {} chars", len(sm_section))

        # L2 Memory — MEMORY.md 索引（始终注入，作为总览）
        if self._context_memory:
            memory_section = self._context_memory.load_memory_prompt()
            if memory_section:
                parts.append(memory_section)

        # L2 Memory — 关键词检索注入（memory_query 非空时触发）
        if self._context_memory and memory_query:
            topic_section = self._context_memory.recall_topics_prompt(memory_query, k=3)
            if topic_section:
                parts.append(topic_section)

        # SessionContext (priority 60)
        if session_context:
            sc_section = getattr(session_context, "to_prompt_section", lambda: "")()
            if sc_section:
                parts.append(sc_section)

        result = "\n\n".join(parts)
        logger.debug(
            "assemble_system_prompt: {} parts, {} tokens",
            len(parts),
            count_tokens(result, model=self._model) if result else 0,
        )
        return result

    # ── History ───────────────────────────────────────────────────────

    async def load_history(self) -> list[dict[str, Any]]:
        """从 DB 加载历史消息（两层窗口 + token-aware 截止）。"""
        if self._history_loader:
            raw = await self._history_loader()
        else:
            return []

        # 应用历史条数限制
        if len(raw) > self._history_load_limit:
            raw = raw[-self._history_load_limit:]

        return raw

    def build_history_summary(
        self,
        history: list[dict[str, Any]] | None,
        max_rounds: int = 3,
        max_tokens: int = 800,
    ) -> str:
        """构建简短历史摘要。"""
        if not history:
            return ""

        from utils.token_utils import count_tokens_messages

        # 只取最近 max_rounds 轮
        rounds: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []

        for msg in history:
            if msg.get("role") == "assistant" and current:
                rounds.append(current)
                current = [msg]
            else:
                current.append(msg)

        if current:
            rounds.append(current)

        recent = rounds[-max_rounds:] if rounds else []
        recent_msgs: list[dict[str, Any]] = []
        for r in recent:
            recent_msgs.extend(r)

        total = count_tokens_messages(recent_msgs, model=self._model)
        if total <= max_tokens:
            return "\n".join(
                f"[{m.get('role','?')}] {m.get('content','')[:100]}"
                for m in recent_msgs if m.get("content")
            )

        # 截断
        chars = max_tokens * 3
        summary = "\n".join(
            f"[{m.get('role','?')}] {m.get('content','')[:80]}"
            for m in recent_msgs if m.get("content")
        )
        return summary[:chars]

    # ── Token Budget ────────────────────────────────────────────────

    def init_budget(self, context_max_tokens: int) -> None:
        """初始化 Token 预算。"""
        self._context_max_tokens = context_max_tokens
        self._compactor.update_budget(context_max_tokens)

    def sync_tokens(self, messages: list[dict[str, Any]] | None = None) -> None:
        """同步 Token 计数。"""
        if messages is not None:
            from utils.token_utils import count_tokens_messages
            self._total_tokens = count_tokens_messages(messages, model=self._model)

    @property
    def total_tokens(self) -> int:
        """当前总 token 数。"""
        return self._total_tokens

    # ── Tool Result Persistence ────────────────────────────────────

    async def persist_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> dict[str, Any]:
        """持久化 Tool Result。"""
        remaining = (
            self._context_max_tokens - self._total_tokens
            if self._context_max_tokens > 0
            else 10000
        )
        return await self._artifact_provider.process_tool_result(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            content=result,
            remaining_budget_tokens=remaining,
        )

    # ── Session Memory (L1) ─────────────────────────────────────────

    @property
    def session_memory(self) -> Any:
        """获取 L1 Session Memory 实例。"""
        return self._session_memory

    # ── L2 Memory ────────────────────────────────────────────────────

    @property
    def context_memory(self) -> LongTermMemory:
        """获取 L2 Long Term Memory 实例。"""
        return self._context_memory

    # ── Stats / 可观测性 ─────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """获取统计信息。"""
        compact_stats = self._compactor.get_stats()
        return {
            "provider": "file",
            "total_tokens": self._total_tokens,
            "context_max_tokens": self._context_max_tokens,
            "embedding_enabled": self._embedding_enabled,
            "compact": {
                "total_calls": compact_stats.total_calls,
                "total_compacted": compact_stats.total_compacted,
                "auto_compactions": compact_stats.auto_compactions,
                "emergency_compactions": compact_stats.emergency_compactions,
                "consecutive_failures": compact_stats.consecutive_failures,
            },
        }
