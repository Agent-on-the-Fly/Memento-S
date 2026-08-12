# SPDX-License-Identifier: Apache-2.0
"""ContextManager — Agent 唯一的上下文管理接口。

Public API:
  Prompt & History:
    load_history()            — 从 DB 加载历史，token-aware 截止
    load_and_assemble()       — 统一消息初始化
    assemble_system_prompt()  — 构造 system prompt
    build_history_summary()   — 简短历史摘要 (用于 intent 识别)
    invalidate_skills_cache() — 清除技能摘要缓存

  Context Runtime:
    init_budget()             — 设置 token 预算
    append()                  — 纯追加 + token 计数 + _seq 注入
    prepare_for_api()         — Pre-API Pipeline (每次 LLM 调用前)
    persist_tool_result()     — 委托 ArtifactStore 处理大结果

  Session Memory (L1):
    session_memory            — SessionMemory 实例访问
"""
from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .session.types import SessionGoal

from infra.memory.impl.long_term_memory import LongTermMemory
from .config import ContextManagerConfig
from .history.manager import HistoryManager
from .pre_api_pipeline import PreApiPipeline
from infra.memory.impl.session_memory import SessionMemory
from .session_context import SessionContext
from core.prompts.prompt_builder import PromptBuilder
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
from core.context.session import build_session_context_block
from middleware.config import g_config
from utils.logger import get_logger
from utils.token_utils import count_tokens, count_tokens_messages

from infra.compact.storage import ArtifactStore
from infra.memory.context_block import build_memory_context_block

# ── Debug logging ──────────────────────────────────────────────────────

_LOG_FILE = Path("/Users/manson/ai/memento/opc_memento_s/.cursor/debug-e72ebc.log")


def _log_debug(
    sessionId: str,
    location: str,
    message: str,
    data: dict,
    runId: str,
    hypothesisId: str,
) -> None:
    import time

    entry = {
        "id": f"log_{int(time.time() * 1000)}",
        "timestamp": int(time.time() * 1000),
        "sessionId": sessionId,
        "location": location,
        "message": message,
        "data": data,
        "runId": runId,
        "hypothesisId": hypothesisId,
    }
    try:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


logger = get_logger(__name__)

HistoryLoader = Any


class ContextManager:
    """Session 级别的上下文管理器。"""

    def __init__(
        self,
        ctx: SessionContext,
        config: ContextManagerConfig,
        *,
        skill_gateway: SkillGateway | None = None,
        history_loader: HistoryLoader | None = None,
    ) -> None:
        self._ctx = ctx
        self._cfg = config

        self._skill_gateway = skill_gateway
        self._skills_summary_cache: str | None = None

        self.workspace = g_config.paths.workspace_dir

        # token 状态
        self._total_tokens: int = 0
        self._context_max_tokens: int = 0
        self._summary_tokens: int = 0
        self._slim_budget: int = 100

        # 消息序列号 (单调递增，用于 SM compact 精确定位)
        self._msg_seq: int = 0

        # session directory from SessionContext
        self._ctx.session_dir.mkdir(parents=True, exist_ok=True)

        # L1 Session Memory (CC-style summary.md)
        self._session_memory = SessionMemory(
            session_dir=self._ctx.session_dir,
            model=self._model,
            llm_update_interval=config.sm_llm_update_interval,
        )

        # Artifact Store (file-backed, lives in session_dir/artifacts/)
        self._artifact_store: ArtifactStore = ArtifactStore(
            session_dir=self._ctx.session_dir,
            persist_ratio=config.persist_ratio,
            extract_ratio=config.extract_ratio,
            model=self._model,
        )

        # History manager (extracted)
        self._history = HistoryManager(
            config=config,
            history_loader=history_loader,
            session_memory=self._session_memory,
            model_getter=lambda: self._model,
        )

        # Pre-API Pipeline (initialized after init_budget)
        self._pipeline: PreApiPipeline | None = None

        # L2 Long Memory
        ctx_dir: Path = g_config.paths.context_dir
        memory_dir = ctx_dir / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        self._memory = LongTermMemory(memory_dir, model=self._model)

    # ═══════════════════════════════════════════════════════════════
    # Properties
    # ═══════════════════════════════════════════════════════════════

    @property
    def session_id(self) -> str:
        return self._ctx.session_id

    @property
    def session_memory(self) -> SessionMemory:
        return self._session_memory

    @property
    def artifact_store(self) -> ArtifactStore:
        return self._artifact_store

    @property
    def context_memory(self) -> LongTermMemory | None:
        return self._memory

    @property
    def _model(self) -> str:
        try:
            return g_config.llm.current_profile.model
        except Exception:
            return ""

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    # ═══════════════════════════════════════════════════════════════
    # Token 状态 & Budget
    # ═══════════════════════════════════════════════════════════════

    _BUDGET_FLOOR = 100

    def init_budget(self, context_max_tokens: int) -> None:
        """设置 token 预算并初始化 Pipeline。"""
        if context_max_tokens <= 0:
            logger.warning(
                "input_budget={} is non-positive, falling back to 4096",
                context_max_tokens,
            )
            context_max_tokens = 4096

        self._context_max_tokens = context_max_tokens
        self._summary_tokens = max(
            int(context_max_tokens * self._cfg.summary_ratio),
            self._BUDGET_FLOOR,
        )

        floor = self._BUDGET_FLOOR
        cfg = self._cfg
        self._slim_budget = max(floor, int(context_max_tokens * cfg.slim_ratio))
        self._history.update_slim_budget(self._slim_budget)

        self._pipeline = PreApiPipeline(
            model=self._model,
            context_budget=context_max_tokens,
            microcompact_keep_recent=cfg.microcompact_keep_recent,
            microcompact_compactable_tools=set(cfg.microcompact_compactable_tools),
            emergency_keep_tail=cfg.emergency_keep_tail,
            max_compact_failures=cfg.max_compact_failures,
            sm_compact_min_tokens=max(floor, int(context_max_tokens * cfg.sm_compact_min_ratio)),
            sm_compact_max_tokens=max(floor, int(context_max_tokens * cfg.sm_compact_max_ratio)),
            artifact_store=self._artifact_store,
            session_memory=self._session_memory,
            breaker_cooldown_s=cfg.breaker_cooldown_s,
            pipeline_preview_budget=max(floor, int(context_max_tokens * cfg.preview_ratio)),
        )
        logger.info(
            "Budget: context_max={}, summary_tokens={}, slim={}, pipeline initialized",
            context_max_tokens, self._summary_tokens, self._slim_budget,
        )

    def sync_tokens(self, messages: list[dict[str, Any]] | None = None) -> None:
        """同步 token 计数。"""
        if messages is not None:
            self._total_tokens = count_tokens_messages(messages, model=self._model)

    # ═══════════════════════════════════════════════════════════════
    # Append (纯追加 + _seq 注入)
    # ═══════════════════════════════════════════════════════════════

    async def append(
        self,
        messages: list[dict[str, Any]],
        new_msgs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """纯追加 + token 计数 + _seq 注入。"""
        for msg in new_msgs:
            self._msg_seq += 1
            msg["_seq"] = self._msg_seq
        result = list(messages) + new_msgs
        added_tokens = count_tokens_messages(new_msgs, model=self._model)
        self._total_tokens += added_tokens
        return result

    # ═══════════════════════════════════════════════════════════════
    # Pre-API Pipeline
    # ═══════════════════════════════════════════════════════════════

    async def prepare_for_api(
        self,
        messages: list[dict[str, Any]],
        *,
        state_messages: list[dict[str, Any]] | None = None,
        force_compact: bool = False,
    ) -> list[dict[str, Any]]:
        """Pre-API Pipeline 入口。"""
        if self._pipeline is None:
            logger.warning("Pipeline not initialized, returning messages as-is")
            return messages

        result = await self._pipeline.prepare_for_api(
            messages,
            state_messages_ref=state_messages,
            force_compact=force_compact,
        )
        if result.was_compacted:
            self._total_tokens = result.tokens_after
        return result.messages_for_api

    # ═══════════════════════════════════════════════════════════════
    # Tool Result Persistence
    # ═══════════════════════════════════════════════════════════════

    async def force_compact_now(self) -> tuple[int, int, str]:
        """立即压缩历史上下文。"""
        history = await self._history.load_history()
        if not history or len(history) <= 2:
            return 0, 0, ""

        old_tokens = count_tokens_messages(history, model=self._model)
        target = max(self._summary_tokens, 200) if self._summary_tokens else 2000
        from core.context.history.manager import _emergency_compact

        compacted = await _emergency_compact(
            history, model=self._model, summary_tokens=target
        )

        new_tokens = count_tokens_messages(compacted, model=self._model)
        preview = ""
        for msg in compacted:
            content = msg.get("content", "")
            if isinstance(content, str) and "summary" in content.lower():
                preview = content
                break

        return old_tokens, new_tokens, preview

    async def persist_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> dict[str, Any]:
        """委托 ArtifactStore 处理大结果持久化。"""
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict) and parsed.get("_no_persist"):
                return {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": result,
                    "_persisted": False,
                }
        except (json.JSONDecodeError, TypeError):
            pass

        remaining = max(0, self._context_max_tokens - self._total_tokens)
        return await self._artifact_store.process_tool_result(
            tool_call_id, tool_name, result,
            remaining_budget_tokens=remaining,
        )

    # ═══════════════════════════════════════════════════════════════
    # History Loading (delegated to HistoryManager)
    # ═══════════════════════════════════════════════════════════════

    async def load_history(self) -> list[dict[str, Any]]:
        """从 DB 加载历史。委托给 HistoryManager。"""
        return await self._history.load_history()

    def build_history_summary(
        self,
        history: list[dict[str, Any]] | None,
        max_rounds: int = 3,
        max_tokens: int = 800,
    ) -> str:
        """构建简短历史摘要 (用于 intent 识别)。委托给 HistoryManager。"""
        return self._history.build_history_summary(history, max_rounds, max_tokens)

    # ═══════════════════════════════════════════════════════════════
    # Unified Message Assembly
    # ═══════════════════════════════════════════════════════════════

    async def load_and_assemble(
        self,
        current_message: str,
        *,
        history: list[dict[str, Any]] | None = None,
        media: list[str] | list[Path] | None = None,
        matched_skills_context: str = "",
        agent_profile: Any = None,
        session_context: SessionGoal | None = None,
        mode: str = "agentic",
        intent_shifted: bool = False,
        effective_context_window: int | None = None,
    ) -> list[dict[str, Any]]:
        """统一的消息初始化。"""
        # [DEBUG] Instrument: log key fields before assembly
        _log_debug(
            sessionId="e72ebc",
            location="context_manager.py:load_and_assemble",
            message="load_and_assemble called",
            data={
                "current_message_len": len(current_message),
                "history_len": len(history) if history is not None else None,
                "mode": mode,
                "session_context_type": type(session_context).__name__ if session_context else None,
            },
            runId="initial",
            hypothesisId="A",
        )

        if history is None:
            history = await self._history.load_history(
                mode=mode, intent_shifted=intent_shifted
            )

        system_prompt = await self.assemble_system_prompt(
            current_message,
            mode=mode,
            intent_shifted=intent_shifted,
            matched_skills_context=matched_skills_context,
            agent_profile=agent_profile,
            session_context=session_context,
        )

        model = self._model
        system_tokens = count_tokens(system_prompt, model=model)
        user_tokens = count_tokens(current_message, model=model)
        input_budget = g_config.llm.current_profile.input_budget
        if effective_context_window and effective_context_window > 0:
            max_tokens = g_config.llm.current_profile.max_tokens
            adjusted = effective_context_window - max_tokens
            if adjusted > 0:
                input_budget = min(input_budget, adjusted)

        history_budget = input_budget - system_tokens - user_tokens

        selected_history = history
        if selected_history:
            history_tokens = count_tokens_messages(selected_history, model=model)
            if history_budget <= 0 or history_tokens > history_budget:
                selected_history = selected_history[-4:]
                logger.warning(
                    "Budget exhausted ({}), keeping last {} msgs",
                    history_budget, len(selected_history),
                )

        if selected_history:
            last = selected_history[-1]
            if last.get("role") == "user" and last.get("content") == current_message:
                selected_history = selected_history[:-1]

        result = [
            {"role": "system", "content": system_prompt},
            *selected_history,
            {"role": "user", "content": current_message},
        ]

        self.sync_tokens(result)
        self.init_budget(input_budget)

        # [DEBUG] Instrument: log result message structure to detect tool role issue
        tool_role_count = sum(1 for m in result if m.get("role") == "tool")
        assistant_with_tc_count = sum(
            1 for m in result
            if m.get("role") == "assistant" and m.get("tool_calls")
        )
        user_role_count = sum(1 for m in result if m.get("role") == "user")
        # Check for tool role without preceding assistant with tool_calls
        problem_found = False
        problem_detail = ""
        for i, m in enumerate(result):
            if m.get("role") == "tool":
                # Find preceding assistant
                prev_assistant = None
                for j in range(i - 1, -1, -1):
                    if result[j].get("role") == "assistant":
                        prev_assistant = result[j]
                        break
                if not (prev_assistant and prev_assistant.get("tool_calls")):
                    problem_found = True
                    problem_detail = f"tool msg at idx {i} has no preceding assistant with tool_calls. tool_call_id={m.get('tool_call_id')}, content_preview={str(m.get('content',''))[:80]}"
        _log_debug(
            sessionId="e72ebc",
            location="context_manager.py:load_and_assemble:result",
            message="assembled messages structure",
            data={
                "total_msgs": len(result),
                "tool_role_count": tool_role_count,
                "assistant_with_tool_calls_count": assistant_with_tc_count,
                "user_role_count": user_role_count,
                "problem_found": problem_found,
                "problem_detail": problem_detail,
                "msg_roles": [m.get("role") for m in result],
            },
            runId="initial",
            hypothesisId="A",
        )

        return result

    # ═══════════════════════════════════════════════════════════════
    # System Prompt
    # ═══════════════════════════════════════════════════════════════

    async def assemble_system_prompt(
        self,
        current_message: str,
        *,
        mode: str = "agentic",
        intent_shifted: bool = False,
        matched_skills_context: str = "",
        agent_profile: Any = None,
        session_context: SessionGoal | None = None,
    ) -> str:
        """构造完整 system prompt。"""
        mode = str(mode) if not isinstance(mode, str) else mode
        pb = PromptBuilder()

        identity = self._identity_section()
        if agent_profile is not None and hasattr(agent_profile, "to_prompt_section"):
            identity += "\n\n" + agent_profile.to_prompt_section()
        pb.add(identity, priority=10, label="identity")

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

        pb.add(PROTOCOL_AND_FORMAT, priority=30, label="protocol")

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

        # Memory injection: combine L1 + L2, wrap with <memory-context> fence tag
        # (prevents the model from treating recalled context
        # as user input). Added at priority 55, above bare L2 (45) and below
        # session state (60). Individual L1 (50) and bare L2 (45) sections are
        # suppressed when the fenced block is added.
        if self._session_memory is not None or self._memory is not None:
            combined_parts: list[str] = []
            if self._session_memory is not None:
                l1 = self._session_memory.to_prompt_section()
                if l1:
                    combined_parts.append(l1)
            if self._memory is not None:
                l2 = self._memory.load_memory_prompt()
                if l2:
                    combined_parts.append(l2)
            combined = "\n\n".join(combined_parts)
            fenced = build_memory_context_block(combined)
            if fenced:
                pb.add(fenced, priority=55, label="memory_context")
        else:
            # Fallback: individual L1/L2 without fence tag
            l1_section = self._session_memory.to_prompt_section() if self._session_memory else None
            if l1_section:
                pb.add(l1_section, priority=50, label="session_memory")
            if self._memory:
                memory_section = self._memory.load_memory_prompt()
                if memory_section:
                    pb.add(memory_section, priority=45, label="memory")

        if session_context is not None:
            session_section = build_session_context_block(
                session_context, current_message
            )
            pb.add(session_section, priority=60, label="session_state")

        return pb.build()

    def invalidate_skills_cache(self) -> None:
        """清除技能摘要缓存。"""
        self._skills_summary_cache = None

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
