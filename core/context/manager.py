"""ContextManager — Agent 唯一的上下文管理接口。

Public API:
  Prompt & History:
    load_history()            — 从 DB 加载历史，token-aware 截止
    assemble_messages()       — 组装完整 message list（system + history + user）
    assemble_system_prompt()  — 构造 system prompt
    build_history_summary()   — 简短历史摘要（用于 intent 识别）
    invalidate_skills_cache() — 清除技能摘要缓存

  Context Runtime:
    init_budget()             — 设置 token 预算（input_budget）+ 派生 compress/compact 阈值
    append()                  — 追加消息 + 自动 compress / compact（compact 时归档到 scratchpad）
    persist_tool_result()     — 返回内联 tool message（不写盘、不截断）
    write_to_scratchpad()     — 手动写入 scratchpad
"""

from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine

from core.prompts.templates import (
    AGENT_IDENTITY_OPENING,
    BUILTIN_TOOLS_SECTION,
    ENVIRONMENT_SECTION,
    EXECUTION_CONSTRAINTS_SECTION,
    IDENTITY_SECTION,
    IMPORTANT_DIRECT_REPLY,
    PROTOCOL_AND_FORMAT,
    SKILLS_SECTION,
)
from core.skill.gateway import SkillGateway
from core.utils import format_user_content
from middleware.config import g_config
from utils.logger import get_logger
from utils.token_utils import count_tokens, count_tokens_messages

from core.prompts.prompt_builder import PromptBuilder

from .block import BlockManager, Block, make_event, ensure_session_dir
from .compaction import compact_messages, compress_message
from .memory import ContextMemory
from .runtime_state import RuntimeState, RuntimeStateStore, sync_from_agent_run
from .schemas import ContextConfig
from .scratchpad import Scratchpad

logger = get_logger(__name__)

HistoryLoader = Callable[[str, int], Coroutine[Any, Any, list[dict[str, Any]]]]


class ContextManager:
    """Session 级别的上下文管理器。

    生命周期: agent.reply_stream() 创建 → assemble_messages() → execution 使用。
    """

    def __init__(
        self,
        session_id: str,
        config: ContextConfig,
        *,
        skill_gateway: SkillGateway | None = None,
        history_loader: HistoryLoader | None = None,
        embedding_client: Any | None = None,
        vector_storage: Any | None = None,
    ) -> None:
        self.session_id = session_id
        self._cfg = config

        self._skill_gateway = skill_gateway
        self._history_loader = history_loader
        self._skills_summary_cache: str | None = None

        self._embedding_client = embedding_client
        self._vector_storage = vector_storage

        self.workspace = g_config.paths.workspace_dir

        # token 状态（init_budget 时设置）
        self._total_tokens: int = 0
        self._context_max_tokens: int = 0
        self._compress_threshold: int = 0
        self._compact_trigger: int = 0
        self._summary_tokens: int = 0

        # directories
        ctx_dir: Path = g_config.paths.context_dir
        today_str = datetime.now().strftime("%Y-%m-%d")
        date_dir = ctx_dir / today_str
        date_dir.mkdir(parents=True, exist_ok=True)

        # session directory (block-based storage)
        self._session_dir = ensure_session_dir(ctx_dir, session_id)

        # scratchpad (still uses date_dir for backward compat)
        self._scratchpad = Scratchpad(
            session_id,
            date_dir,
            artifact_fold_char_limit=config.artifact_fold_char_limit,
            artifact_fold_line_limit=config.artifact_fold_line_limit,
            artifact_preview_max_lines=config.artifact_preview_max_lines,
            artifact_preview_max_chars=config.artifact_preview_max_chars,
        )

        self._last_compacted: bool = False

        # runtime state (session-level, lives in session dir)
        self._runtime_state_store = RuntimeStateStore(session_id, self._session_dir)

        # block manager
        self._block_manager = BlockManager(session_id, self._session_dir)

        self._memory: ContextMemory | None = None
        if config.memory_enabled:
            self._memory = ContextMemory(ctx_dir, date_dir, config)

    @property
    def bounded_prompt_enabled(self) -> bool:
        """是否启用 bounded prompt 模式。"""
        return self._cfg.bounded_prompt_enabled

    @property
    def _model(self) -> str:
        try:
            return g_config.llm.current_profile.model
        except Exception:
            return ""

    # ═══════════════════════════════════════════════════════════════
    # Token 状态 & append
    # ═══════════════════════════════════════════════════════════════

    def init_budget(self, context_max_tokens: int) -> None:
        """设置 token 预算，所有阈值直接从 input_budget * ratio 派生。"""
        if context_max_tokens <= 0:
            logger.warning(
                "input_budget={} is non-positive (context_window < max_tokens?), "
                "falling back to 4096",
                context_max_tokens,
            )
            context_max_tokens = 4096
        self._context_max_tokens = context_max_tokens
        self._compress_threshold = max(
            int(context_max_tokens * self._cfg.compress_threshold_ratio),
            512,
        )
        self._compact_trigger = max(
            int(context_max_tokens * self._cfg.compaction_trigger_ratio),
            1024,
        )
        self._summary_tokens = max(
            int(context_max_tokens * self._cfg.summary_ratio),
            200,
        )
        logger.info(
            "Budget: context_max={}, compact_trigger={}, "
            "compress_threshold={}, summary_tokens={}",
            context_max_tokens,
            self._compact_trigger,
            self._compress_threshold,
            self._summary_tokens,
        )

    def sync_tokens(self, messages: list[dict[str, Any]]) -> None:
        """用完整消息列表同步 _total_tokens（首次/重置时调用）。"""
        self._total_tokens = count_tokens_messages(messages, model=self._model)
        logger.debug("Token state synced: {}", self._total_tokens)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def consume_compacted_flag(self) -> bool:
        """上一次 append() 是否触发了 compact。调用后自动重置。"""
        val = self._last_compacted
        self._last_compacted = False
        return val

    async def append(
        self,
        messages: list[dict[str, Any]],
        new_msgs: list[dict[str, Any]],
        plan_status: str = "",
    ) -> list[dict[str, Any]]:
        """追加消息。

        bounded prompt 模式下：直接追加，不触发 LLM compress/compact。
        上下文大小由第 2 层（artifact fold）和第 3 层（block event slim）控制。

        非 bounded 模式下：保留 LLM compress + compact 逻辑。
        """
        self._last_compacted = False

        if self._cfg.bounded_prompt_enabled:
            # bounded 模式：直接追加，跳过 LLM 压缩
            result = list(messages) + new_msgs
            added_tokens = count_tokens_messages(new_msgs, model=self._model)
            self._total_tokens += added_tokens
            return result

        # ── 非 bounded 模式：LLM compress + compact ──
        compressed = []
        for msg in new_msgs:
            if self._compact_trigger > 0:
                compressed.append(
                    await compress_message(
                        msg,
                        max_msg_tokens=self._compress_threshold,
                        summary_tokens=self._summary_tokens,
                        model=self._model,
                    )
                )
            else:
                compressed.append(msg)

        result = list(messages) + compressed
        added_tokens = count_tokens_messages(compressed, model=self._model)
        self._total_tokens += added_tokens

        if self._compact_trigger > 0 and self._total_tokens > self._compact_trigger:
            logger.info(
                "Compact trigger: {} > {} (max={})",
                self._total_tokens,
                self._compact_trigger,
                self._context_max_tokens,
            )
            pre_compact = list(result)
            compacted, new_tokens = await compact_messages(
                result,
                summary_tokens=self._summary_tokens,
                has_scratchpad=self._scratchpad.path.exists(),
                plan_status=plan_status,
                model=self._model,
            )
            if len(compacted) < len(pre_compact):
                self._archive_to_scratchpad(pre_compact)
                result = compacted
                self._total_tokens = new_tokens
                self._last_compacted = True

        return result

    # ═══════════════════════════════════════════════════════════════
    # History loading
    # ═══════════════════════════════════════════════════════════════

    async def load_history(
        self, current_user_message: str | None = None
    ) -> list[dict[str, Any]]:
        """从 DB 加载历史，两层窗口 + token-aware 截止 + 可选 embedding 检索。

        1. 从 DB 取最近 history_load_limit 条消息
        2. 精简 tool result（规则提取，零 LLM 成本）
        3. 分两层：最近 N 轮保留原文，更早消息 compact 为摘要
        4. 如果 embedding 启用，用语义检索补充相关早期消息
        5. 整体 token 不超过 input_budget * history_budget_ratio
        """
        if not self._history_loader:
            return []

        load_limit = self._cfg.history_load_limit
        raw = await self._history_loader(self.session_id, load_limit)
        if not raw:
            return []

        raw = self._slim_tool_results(raw, mark_historical=True)

        input_budget = g_config.llm.current_profile.input_budget
        if input_budget <= 0:
            return raw

        budget = int(input_budget * self._cfg.history_budget_ratio)

        recent, earlier = self._split_by_rounds(raw, self._cfg.recent_rounds_keep)

        # 最近层：token-aware 截止
        model = self._model
        recent_selected: list[dict[str, Any]] = []
        recent_tokens = 0
        for msg in reversed(recent):
            t = count_tokens(str(msg.get("content", "")), model=model)
            if recent_tokens + t > budget:
                break
            recent_selected.append(msg)
            recent_tokens += t
        recent_selected.reverse()

        remaining_budget = budget - recent_tokens

        # 早期层：embedding 语义检索 或 compact
        earlier_result: list[dict[str, Any]] = []
        if earlier and remaining_budget > 200:
            if (
                self._cfg.embedding_enabled
                and current_user_message
                and self._embedding_client
                and self._vector_storage
            ):
                earlier_result = await self._retrieve_relevant_earlier(
                    earlier, current_user_message, remaining_budget
                )
            else:
                earlier_tokens = sum(
                    count_tokens(str(m.get("content", "")), model=model)
                    for m in earlier
                )
                if earlier_tokens > remaining_budget:
                    summary_tokens = min(self._summary_tokens or 2000, remaining_budget)
                    compacted, _ = await compact_messages(
                        earlier,
                        summary_tokens=summary_tokens,
                        has_scratchpad=self._scratchpad.path.exists(),
                        model=model,
                    )
                    earlier_result = compacted
                else:
                    earlier_result = earlier

        result = earlier_result + recent_selected
        logger.info(
            "Two-tier load: {} raw -> {} earlier + {} recent = {} msgs, budget {}/{}",
            len(raw),
            len(earlier_result),
            len(recent_selected),
            len(result),
            recent_tokens,
            budget,
        )
        return result

    async def _retrieve_relevant_earlier(
        self,
        earlier: list[dict[str, Any]],
        user_message: str,
        token_budget: int,
    ) -> list[dict[str, Any]]:
        """用 embedding 从早期消息中检索语义最相关的消息。"""
        try:
            query_vec = await self._embedding_client.embed_query(user_message)
            results = await self._vector_storage.search(
                query_vec, k=self._cfg.history_load_limit
            )
        except Exception as e:
            logger.warning("Embedding retrieval failed, falling back to compact: {}", e)
            summary_tokens = min(self._summary_tokens or 2000, token_budget)
            compacted, _ = await compact_messages(
                earlier,
                summary_tokens=summary_tokens,
                has_scratchpad=self._scratchpad.path.exists(),
                model=self._model,
            )
            return compacted

        relevant_ids = {conv_id for conv_id, _score in results}

        conv_id_map: dict[str, dict[str, Any]] = {}
        for msg in earlier:
            cid = msg.get("conversation_id")
            if cid:
                conv_id_map[cid] = msg

        selected: list[dict[str, Any]] = []
        accumulated = 0
        for msg in earlier:
            cid = msg.get("conversation_id")
            if cid and cid in relevant_ids:
                t = count_tokens(str(msg.get("content", "")), model=self._model)
                if accumulated + t > token_budget:
                    break
                selected.append(msg)
                accumulated += t

        if not selected:
            summary_tokens = min(self._summary_tokens or 2000, token_budget)
            compacted, _ = await compact_messages(
                earlier,
                summary_tokens=summary_tokens,
                has_scratchpad=self._scratchpad.path.exists(),
                model=self._model,
            )
            return compacted

        logger.info(
            "Embedding retrieval: {} relevant out of {} earlier msgs ({} tokens)",
            len(selected),
            len(earlier),
            accumulated,
        )
        return selected

    @staticmethod
    def _split_by_rounds(
        messages: list[dict[str, Any]], keep_rounds: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """将消息按对话轮次分割为 (recent, earlier)。

        一轮 = 一条 user 消息 + 后续所有非 user 消息。
        从末尾倒数 keep_rounds 轮。
        """
        if not messages or keep_rounds <= 0:
            return messages, []

        round_boundaries: list[int] = []
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                round_boundaries.append(i)

        if not round_boundaries:
            return messages, []

        if len(round_boundaries) <= keep_rounds:
            return messages, []

        split_idx = round_boundaries[-keep_rounds]
        return messages[split_idx:], messages[:split_idx]

    # ═══════════════════════════════════════════════════════════════
    # Prompt & History
    # ═══════════════════════════════════════════════════════════════

    def invalidate_skills_cache(self) -> None:
        """清除技能摘要缓存（安装新技能后调用）。"""
        self._skills_summary_cache = None

    async def force_compact_now(self) -> tuple[int, int, str]:
        """立即压缩历史上下文。

        Returns:
            (old_tokens, new_tokens, summary_preview)
        """
        history = await self.load_history()
        if not history or len(history) <= 2:
            return 0, 0, ""

        old_tokens = count_tokens_messages(history, model=self._model)
        target = max(self._summary_tokens, 200) if self._summary_tokens else 2000
        compacted, new_tokens = await compact_messages(
            history,
            summary_tokens=target,
            model=self._model,
        )

        preview = ""
        for msg in compacted:
            if "[历史摘要" in (msg.get("content") or ""):
                preview = msg["content"]
                break

        self._total_tokens = new_tokens
        logger.info("Force compact: {} -> {} tokens", old_tokens, new_tokens)
        return old_tokens, new_tokens, preview

    def build_history_summary(
        self,
        history: list[dict[str, Any]] | None,
        max_rounds: int = 3,
        max_tokens: int = 800,
    ) -> str:
        """构建简短历史摘要（用于 intent 识别）。

        除了 user/assistant 内容外，还包含 tool_calls 信息，
        帮助 intent 识别阶段区分"讨论了某工具"和"实际执行了某工具"。
        """
        if not history:
            return "(no prior context)"

        meaningful = [
            m
            for m in history
            if m.get("role") in ("user", "assistant")
            and (str(m.get("content", "")).strip() or m.get("tool_calls"))
        ]
        if not meaningful:
            return "(no prior context)"

        candidates = meaningful[-(max_rounds * 2) :]
        selected: list[str] = []
        remaining_tokens = max_tokens

        for m in reversed(candidates):
            content = str(m.get("content", "")).strip()
            tool_calls = m.get("tool_calls")

            line = f"{m['role']}: {content}"
            if tool_calls and isinstance(tool_calls, list):
                tool_names = [
                    tc.get("function", {}).get("name", "unknown")
                    if isinstance(tc, dict) else "unknown"
                    for tc in tool_calls
                ]
                line += f" [called tools: {', '.join(tool_names)}]"

            tokens = count_tokens(line, model=self._model)
            if tokens <= remaining_tokens:
                selected.append(line)
                remaining_tokens -= tokens
            else:
                char_budget = remaining_tokens * 3
                if char_budget > 50:
                    selected.append(f"{m['role']}: {content[:char_budget]}...")
                break

        selected.reverse()
        return "\n".join(selected) if selected else "(no prior context)"

    async def assemble_system_prompt(
        self,
        *,
        mode: str = "agentic",
        intent_shifted: bool = False,
        matched_skills_context: str = "",
        agent_profile: Any = None,
        session_context: Any = None,
    ) -> str:
        """构造完整 system prompt（内部使用 PromptBuilder）。"""
        mode = str(mode) if not isinstance(mode, str) else mode
        pb = PromptBuilder()

        # ── Identity (priority 10) ──
        identity = self._identity_section()
        if agent_profile is not None and hasattr(agent_profile, "to_prompt_section"):
            identity += "\n\n" + agent_profile.to_prompt_section()
        pb.add(identity, priority=10, label="identity")

        # ── Behavior (priority 20) ──
        behavior = [
            "## runtime_behavior",
            "- Prefer direct concise reply for simple chit-chat; avoid unnecessary tool calls.",
            "- For task-oriented requests, use tools/skills step-by-step.",
        ]
        if intent_shifted:
            behavior.append(
                "- Current user intent has shifted from previous turns; prioritize latest user message."
            )
        if mode in ("direct", "interrupt"):
            behavior.append(
                "- This turn is classified as direct. Answer directly unless the user explicitly asks for tools."
            )
        pb.add("\n".join(behavior), priority=20, label="behavior")

        # ── Protocol (priority 30) ──
        pb.add(PROTOCOL_AND_FORMAT, priority=30, label="protocol")

        # ── Tools & Skills (priority 40) ──
        if mode not in ("direct", "interrupt"):
            pb.add(BUILTIN_TOOLS_SECTION, priority=40, label="builtin_tools")

            skills_summary = await self._build_skills_summary()
            if skills_summary:
                skills_section = SKILLS_SECTION.format(skills_summary=skills_summary)
                if matched_skills_context:
                    skills_section += "\n\n" + matched_skills_context
                pb.add(skills_section, priority=41, label="skills")
            elif matched_skills_context:
                pb.add(matched_skills_context, priority=41, label="matched_skills")
        else:
            skills_summary = await self._build_skills_summary()
            if skills_summary:
                pb.add(
                    "## Available Skills (reference only)\n\n"
                    "You have the following skills installed. "
                    "In this turn you are answering directly without tool calls, "
                    "but you can reference this list to answer questions about your capabilities.\n\n"
                    f"{skills_summary}",
                    priority=40,
                    label="skills_ref",
                )

        # ── Context (priority 50) ──
        ctx_section = self._get_context_section()
        pb.add(ctx_section, priority=50, label="context")

        # ── Memory (priority 55) ──
        if self._memory:
            memory_section = self._memory.get_context_section(self._scratchpad)
            pb.add(memory_section, priority=55, label="memory")

        # ── Session state (priority 60) ──
        if session_context is not None and hasattr(
            session_context, "to_prompt_section"
        ):
            session_section = session_context.to_prompt_section()
            pb.add(session_section, priority=60, label="session_state")

        return pb.build()

    async def assemble_messages(
        self,
        history: list[dict[str, Any]] | None,
        current_message: str,
        media: list[str] | list[Path] | None = None,
        matched_skills_context: str = "",
        agent_profile: Any = None,
        session_context: Any = None,
        mode: str = "agentic",
        intent_shifted: bool = False,
        effective_context_window: int | None = None,
    ) -> list[dict[str, Any]]:
        """组装完整 message list 并初始化 token 状态。

        history 应由调用方通过 load_history() 预先加载并传入。
        """
        if history is None:
            history = []

        selected_history = self._select_history_for_intent(
            history,
            mode=mode,
            intent_shifted=intent_shifted,
        )
        logger.info(
            "History: raw={}, selected={} (mode={}, shifted={})",
            len(history),
            len(selected_history),
            mode,
            intent_shifted,
        )

        system_prompt = await self.assemble_system_prompt(
            mode=mode,
            intent_shifted=intent_shifted,
            matched_skills_context=matched_skills_context,
            agent_profile=agent_profile,
            session_context=session_context,
        )
        user_content = await format_user_content(current_message, media)

        model = self._model
        system_tokens = count_tokens(system_prompt, model=model)
        user_tokens = count_tokens(
            user_content if isinstance(user_content, str) else current_message,
            model=model,
        )
        input_budget = g_config.llm.current_profile.input_budget
        if effective_context_window and effective_context_window > 0:
            max_tokens = g_config.llm.current_profile.max_tokens
            adjusted = effective_context_window - max_tokens
            if adjusted > 0:
                input_budget = min(input_budget, adjusted)
        history_budget = input_budget - system_tokens - user_tokens

        logger.info(
            "Budget: input_budget={}, sys={}, user={}, history_budget={}",
            input_budget,
            system_tokens,
            user_tokens,
            history_budget,
        )

        if selected_history:
            history_tokens = count_tokens_messages(selected_history, model=model)
            if history_budget > 0 and history_tokens > history_budget:
                summary_target = max(
                    int(input_budget * self._cfg.summary_ratio),
                    200,
                )
                logger.info(
                    "History ({} tokens) exceeds budget ({}), compacting to ~{} tokens",
                    history_tokens,
                    history_budget,
                    summary_target,
                )
                plan_status = self._extract_plan_status(session_context)
                selected_history, _ = await compact_messages(
                    selected_history,
                    summary_tokens=summary_target,
                    plan_status=plan_status,
                    model=model,
                )
            elif history_budget <= 0:
                selected_history = selected_history[-4:]
                logger.warning(
                    "Budget exhausted ({}), keeping last {} msgs",
                    history_budget,
                    len(selected_history),
                )

        if selected_history:
            last = selected_history[-1]
            current_str = (
                user_content if isinstance(user_content, str) else str(user_content)
            )
            if last.get("role") == "user" and last.get("content") == current_str:
                selected_history = selected_history[:-1]

        result = [
            {"role": "system", "content": system_prompt},
            *selected_history,
            {"role": "user", "content": user_content},
        ]

        self.sync_tokens(result)
        self.init_budget(input_budget)

        return result

    # ═══════════════════════════════════════════════════════════════
    # Context Runtime（scratchpad）
    # ═══════════════════════════════════════════════════════════════

    @property
    def scratchpad_path(self) -> Path:
        return self._scratchpad.path

    def persist_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        result: str,
        *,
        status: str = "",
    ) -> dict[str, Any]:
        """返回 tool message。统一由 active block 处理 fold 判定 + event 记录。

        bounded 模式: block.persist_tool_result() 一步完成
        非 bounded 模式 (无 active block): 降级走 scratchpad
        """
        block = self._block_manager.active_block
        if block is not None:
            msg = block.persist_tool_result(
                tool_call_id, tool_name, result,
                fold_char_limit=self._cfg.artifact_fold_char_limit,
                fold_line_limit=self._cfg.artifact_fold_line_limit,
                preview_max_chars=self._cfg.artifact_preview_max_chars,
                preview_max_lines=self._cfg.artifact_preview_max_lines,
                status=status,
            )
            # Track artifact ref in runtime state
            artifact_ref = msg.pop("_artifact_ref", None)
            if artifact_ref:
                self._runtime_state_store.get().on_new_artifact(artifact_ref)
            return msg

        # Fallback: no active block → use scratchpad (non-bounded mode)
        return self._scratchpad.persist_tool_result(tool_call_id, tool_name, result)

    def write_to_scratchpad(self, section: str, content: str) -> str:
        """手动写入 scratchpad，返回引用标记。"""
        return self._scratchpad.write(section, content)

    def _archive_to_scratchpad(self, messages: list[dict[str, Any]]) -> None:
        """compact 触发时，将待压缩的消息批量归档到 scratchpad。"""
        system_start = 1 if messages and messages[0].get("role") == "system" else 0
        rest = messages[system_start:]
        if rest:
            self._scratchpad.archive_messages(rest)
            logger.info("Archived {} messages to scratchpad before compact", len(rest))

    def _get_context_section(self) -> str:
        """生成 system prompt 注入段: scratchpad ref（仅 compact 后有内容时注入）。"""
        ref = self._scratchpad.build_reference()
        return ref if ref else ""

    # ═══════════════════════════════════════════════════════════════
    # Runtime State (session-level persistent control state)
    # ═══════════════════════════════════════════════════════════════

    @property
    def runtime_state(self) -> RuntimeState:
        """Get the current runtime state (loads from file on first access)."""
        return self._runtime_state_store.get()

    def save_runtime_state(self, state: RuntimeState | None = None) -> None:
        """Persist runtime state to disk."""
        self._runtime_state_store.save(state)

    def sync_and_save_runtime_state(
        self,
        agent_run_state: Any = None,
        session_ctx: Any = None,
    ) -> None:
        """Sync from AgentRunState + SessionContext, then persist.

        Called at the end of reply_stream to capture the run's outcome
        into the session-level persistent state.
        """
        rs = self._runtime_state_store.get()
        sync_from_agent_run(rs, agent_run_state, session_ctx)
        self._runtime_state_store.save(rs)

    # ═══════════════════════════════════════════════════════════════
    # Block Management
    # ═══════════════════════════════════════════════════════════════

    @property
    def block_manager(self) -> BlockManager:
        """Access the block manager."""
        return self._block_manager

    @property
    def active_block(self) -> Block | None:
        """Get the current active block."""
        return self._block_manager.active_block

    def start_new_block(self, user_input_text: str) -> Block:
        """Seal current block (if any) and create a new one.

        This is the main entry point for block lifecycle management.
        Called when a new user input arrives.
        """
        # Seal previous block
        sealed = self._block_manager.seal_active_block()
        if sealed:
            # Record seal event in runtime state
            rs = self._runtime_state_store.get()
            rs.on_block_sealed()

        # Create new block
        block = self._block_manager.create_block(user_input_text)

        # Update runtime state with new block
        rs = self._runtime_state_store.get()
        rs.active_block_id = block.block_id

        return block

    def append_block_event(self, event: dict[str, Any]) -> None:
        """Append an event to the active block's events.jsonl."""
        block = self._block_manager.active_block
        if block is not None:
            block.append_event(event)

    def compact_active_block_if_needed(self) -> bool:
        """检查 active block 的 event 数，超过阈值时触发 compact。

        Returns:
            True if compact was performed, False otherwise.
        """
        block = self._block_manager.active_block
        if block is None:
            return False

        threshold = self._cfg.block_compact_threshold
        keep_recent = self._cfg.bounded_recent_events_k

        if block.meta.event_count <= threshold:
            return False

        folded = block.compact_old_events(keep_recent=keep_recent)
        if folded is not None:
            logger.info(
                "Active block compacted: {} events → kept {}",
                folded.get("folded_event_count", 0),
                keep_recent,
            )
            return True
        return False

    # ═══════════════════════════════════════════════════════════════
    # Bounded Prompt Assembly (Phase 4)
    # ═══════════════════════════════════════════════════════════════

    def _build_runtime_state_section(self) -> str:
        """将 RuntimeState 序列化为 system prompt section。"""
        rs = self._runtime_state_store.get()
        d = rs.to_dict()
        # 移除不需要注入 prompt 的元数据字段
        d.pop("session_id", None)
        d.pop("updated_at", None)
        lines = ["## runtime_state (session-level control state)", "```json"]
        lines.append(json.dumps(d, ensure_ascii=False, indent=2))
        lines.append("```")
        return "\n".join(lines)

    def _build_ref_previews_section(self) -> str:
        """从 recent_refs 构建 artifact preview section（全量）。

        读取 artifact 文件前几行作为 preview 注入 prompt。
        block.py 已对 tool_result 做了 fold，上下文本身已精炼，无需截断。
        """
        rs = self._runtime_state_store.get()
        refs = rs.recent_refs
        if not refs:
            return ""
        lines = ["## recent_artifacts (read-only previews)"]
        for ref_path in refs:
            p = Path(ref_path)
            if p.exists():
                try:
                    raw = p.read_text(encoding="utf-8", errors="replace")
                    preview_lines = raw.splitlines()[:self._cfg.artifact_preview_max_lines]
                    preview = "\n".join(preview_lines)
                    if len(preview) > self._cfg.artifact_preview_max_chars:
                        preview = preview[:self._cfg.artifact_preview_max_chars] + "…"
                    lines.append(f"\n### {p.name}")
                    lines.append(f"ref: `{ref_path}`")
                    lines.append(f"```\n{preview}\n```")
                except Exception:
                    lines.append(f"\n### {p.name} (unreadable)")
            else:
                lines.append(f"\n### {ref_path} (file not found)")
        return "\n".join(lines)

    @staticmethod
    def _events_to_messages(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """将 block events 转换为 LLM message 格式。

        Event types → message mapping:
          user_input    → {"role": "user", "content": text}
          assistant     → {"role": "assistant", "content": text}
          tool_call     → {"role": "assistant", "content": None, "tool_calls": [...]}
          tool_result   → {"role": "tool", "tool_call_id": ..., "content": ...}
          tool_result_ref → {"role": "tool", "tool_call_id": ..., "content": ref+preview}
          plan / other  → (skip, info only)
        """
        messages: list[dict[str, Any]] = []
        for ev in events:
            t = ev.get("type", "")

            if t == "user_input":
                messages.append({"role": "user", "content": ev.get("text", "")})

            elif t == "assistant":
                messages.append({"role": "assistant", "content": ev.get("text", "")})

            elif t == "tool_call":
                tc_entry = {
                    "id": ev.get("tool_call_id", ev.get("event_id", "")),
                    "type": "function",
                    "function": {
                        "name": ev.get("tool_name", "unknown"),
                        "arguments": ev.get("args_summary", "{}"),
                    },
                }
                # Merge consecutive tool_calls into one assistant message
                if (messages
                        and messages[-1].get("role") == "assistant"
                        and "tool_calls" in messages[-1]):
                    messages[-1]["tool_calls"].append(tc_entry)
                else:
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tc_entry],
                    })

            elif t == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": ev.get("tool_call_id", ev.get("event_id", "")),
                    "content": ev.get("text", ""),
                })

            elif t == "tool_result_ref":
                ref = ev.get("ref", "")
                preview = ev.get("preview", "")
                size = ev.get("size", 0)
                folded = (
                    f"[artifact_ref: {ref}]\n"
                    f"{preview}\n"
                    f"[{size} chars, full content archived]"
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": ev.get("tool_call_id", ev.get("event_id", "")),
                    "content": folded,
                })

            # plan, block_seal, etc. → skip (not LLM-visible)

        return messages

    async def assemble_messages_bounded(
        self,
        current_message: str,
        media: list[str] | list[Path] | None = None,
        matched_skills_context: str = "",
        agent_profile: Any = None,
        session_context: Any = None,
        mode: str = "agentic",
        intent_shifted: bool = False,
        effective_context_window: int | None = None,
    ) -> list[dict[str, Any]]:
        """Bounded prompt assembly: 从 block events 构建有界 prompt。

        与 assemble_messages() 的区别：
        - 不从 DB 加载全量 history
        - history 由 active block 的 recent events 转换而来
        - system prompt 额外注入 runtime_state 和 artifact previews
        - 不触发 LLM-based compaction（bounded by design）
        """
        # ── System prompt (with runtime_state injected) ──
        system_prompt = await self.assemble_system_prompt(
            mode=mode,
            intent_shifted=intent_shifted,
            matched_skills_context=matched_skills_context,
            agent_profile=agent_profile,
            session_context=session_context,
        )

        # Inject runtime_state section (between memory and session_state)
        runtime_section = self._build_runtime_state_section()
        if runtime_section:
            system_prompt += "\n\n" + runtime_section

        # Inject artifact previews
        ref_section = self._build_ref_previews_section()
        if ref_section:
            system_prompt += "\n\n" + ref_section

        # ── User content ──
        user_content = await format_user_content(current_message, media)

        # ── History from sealed blocks (cross-block context) ──
        past_events = self._block_manager.load_recent_sealed_events(
            past_blocks_k=self._cfg.bounded_past_blocks_k,
        )
        past_messages = self._events_to_messages(past_events) if past_events else []

        # ── History from active block events ──
        block = self._block_manager.active_block
        if block is not None:
            recent_events = block.load_recent_events(
                k=self._cfg.bounded_recent_events_k,
            )
            block_messages = self._events_to_messages(recent_events)
        else:
            block_messages = []

        # Deduplicate trailing user message (same as legacy)
        if block_messages:
            last = block_messages[-1]
            current_str = (
                user_content if isinstance(user_content, str) else str(user_content)
            )
            if last.get("role") == "user" and last.get("content") == current_str:
                block_messages = block_messages[:-1]

        result = [
            {"role": "system", "content": system_prompt},
            *past_messages,
            *block_messages,
            {"role": "user", "content": user_content},
        ]

        # ── Token accounting ──
        self.sync_tokens(result)
        input_budget = g_config.llm.current_profile.input_budget
        if effective_context_window and effective_context_window > 0:
            max_tokens = g_config.llm.current_profile.max_tokens
            adjusted = effective_context_window - max_tokens
            if adjusted > 0:
                input_budget = min(input_budget, adjusted)
        self.init_budget(input_budget)

        logger.info(
            "Bounded prompt: {} past_msgs + {} block_msgs, {} total_tokens (budget={})",
            len(past_messages),
            len(block_messages),
            self._total_tokens,
            input_budget,
        )

        return result

    # ═══════════════════════════════════════════════════════════════
    # Internal
    # ═══════════════════════════════════════════════════════════════

    def _identity_section(self) -> str:
        now_dt = datetime.now()
        now = now_dt.strftime("%Y-%m-%d %H:%M (%A)")
        current_year = str(now_dt.year)
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        environment_section = ENVIRONMENT_SECTION

        return IDENTITY_SECTION.format(
            identity_opening=AGENT_IDENTITY_OPENING,
            current_time=now,
            current_year=current_year,
            runtime=runtime,
            environment_section=environment_section,
            execution_constraints=EXECUTION_CONSTRAINTS_SECTION,
            important_direct_reply=IMPORTANT_DIRECT_REPLY,
        )

    async def _build_skills_summary(self) -> str:
        if self._skills_summary_cache is not None:
            return self._skills_summary_cache
        if not self._skill_gateway:
            return ""
        manifests = await self._skill_gateway.discover()
        if not manifests:
            return ""
        lines = []
        for m in manifests:
            name = m.name.strip()
            desc = (m.description or "").strip()
            if desc and len(desc) > 400:
                desc = desc[:397] + "..."
            lines.append(f"- **{name}**: {desc} (call via `execute_skill`)")
        self._skills_summary_cache = "\n".join(sorted(lines))
        return self._skills_summary_cache

    @staticmethod
    def _extract_plan_status(session_context: Any) -> str:
        """从 SessionContext 提取计划状态，用于 compact 时保留结构化信息。"""
        if session_context is None:
            return ""
        steps = getattr(session_context, "_plan_steps", [])
        statuses = getattr(session_context, "_plan_statuses", [])
        if not steps:
            return ""
        lines: list[str] = []
        for i, step_desc in enumerate(steps):
            status = statuses[i] if i < len(statuses) else "pending"
            tag = {"done": "[DONE]", "pending": "[PENDING]", "failed": "[FAILED]"}.get(
                status, "[PENDING]"
            )
            lines.append(f"- Step {i + 1}: {step_desc} {tag}")
        goal = getattr(session_context, "session_goal", "")
        header = f"Goal: {goal}\n" if goal else ""
        return header + "\n".join(lines)

    @staticmethod
    def _select_history_for_intent(
        history: list[dict[str, Any]],
        *,
        mode: str,
        intent_shifted: bool,
    ) -> list[dict[str, Any]]:
        if not history:
            return []
        if mode in ("direct", "interrupt"):
            return history[-4:]
        if intent_shifted:
            candidate = history[-4:]
            return [m for m in candidate if m.get("role") in {"user", "assistant"}]
        return history

    def _slim_tool_results(
        self,
        messages: list[dict[str, Any]],
        max_fallback_chars: int = 300,
        *,
        mark_historical: bool = False,
    ) -> list[dict[str, Any]]:
        """规则精简 tool result 消息，零 LLM 成本。

        将冗长的 JSON payload 转为一行摘要：
        - skill payload (ok/summary/skill_name) → [tool: name] OK/FAIL — summary
        - 批量 results → 每个 tool 一行
        - 其他 → 截断到 max_fallback_chars

        Args:
            mark_historical: 为 True 时在输出前添加 [historical] 标记，
                             帮助 LLM 区分历史结果与当前执行结果。
        """
        result: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") != "tool":
                result.append(msg)
                continue

            content = msg.get("content", "")
            if not content or not isinstance(content, str):
                result.append(msg)
                continue

            slim = self._slim_single_tool_result(
                content, max_chars=max_fallback_chars, mark_historical=mark_historical,
            )
            slimmed = dict(msg)
            slimmed["content"] = slim
            result.append(slimmed)

        return result

    @staticmethod
    def _slim_single_tool_result(
        content: str,
        max_chars: int = 300,
        *,
        mark_historical: bool = False,
    ) -> str:
        """精简单条 tool result content。"""
        prefix = "[historical] " if mark_historical else ""

        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            if len(content) > max_chars:
                return f"{prefix}{content[:max_chars]}...[truncated]"
            return f"{prefix}{content}" if prefix else content

        if not isinstance(parsed, dict):
            if len(content) > max_chars:
                return f"{prefix}{content[:max_chars]}...[truncated]"
            return f"{prefix}{content}" if prefix else content

        if "skill_name" in parsed or "summary" in parsed:
            skill = parsed.get("skill_name", "unknown")
            ok = parsed.get("ok")
            status = "OK" if ok else ("FAIL" if ok is False else "?")
            summary = parsed.get("summary", "")
            return f"{prefix}[tool: {skill}] {status} — {summary}"

        results = parsed.get("results")
        if isinstance(results, list) and results:
            lines: list[str] = []
            for r in results[:10]:
                tool = r.get("tool", "unknown")
                err = r.get("error")
                if err:
                    lines.append(f"  - {tool}: FAIL — {str(err)[:80]}")
                else:
                    res_str = str(r.get("result", ""))
                    lines.append(f"  - {tool}: OK — {res_str[:80]}")
            return f"{prefix}[batch results]\n" + "\n".join(lines)

        if len(content) > max_chars:
            return f"{prefix}{content[:max_chars]}...[truncated]"
        return f"{prefix}{content}" if prefix else content
