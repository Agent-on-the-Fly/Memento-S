# SPDX-License-Identifier: Apache-2.0
"""Memento-S Agent — thin orchestration layer.

All heavy logic lives in ``phases/``, ``core/context/``, and ``utils.py``.
This file is responsible only for initialisation and the top-level
``reply_stream`` coordination.

Routing:
  DIRECT / INTERRUPT → simple_reply  (no tools, no plan)
  AGENTIC            → plan → execute → reflect
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, AsyncGenerator

import tools as tools_registry
from infra.service import InfraService, InfraContextConfig
from core.context.session import SessionGoal, SessionStatus
from core.context import RuntimeState, RuntimeStateStore
from core.context.session_context import SessionContext
from core.protocol import (
    AGUIProtocolAdapter,
    AgentFinishReason,
    IntentMode,
    RunEmitter,
    StepStatus,
    ToolTranscriptSink,
    new_run_id,
)
from utils.log_config import log_preview_long
from shared.security.policy import PolicyManager
from core.skill.gateway import SkillGateway
from middleware.config import g_config
from middleware.config.mcp_config_manager import g_mcp_config_manager
from middleware.llm import LLMClient
from shared.chat import ChatManager
from shared.schema import SkillConfig
from utils.debug_logger import log_agent_phase, log_debug_marker
from utils.logger import get_logger
from core.agent_profile import AgentProfile, apm
from .finalize import stream_and_finalize
from .phases import (
    AgentRunState,
    generate_plan,
    recognize_intent,
    run_plan_execution,
)
from .phases.planning import PlanContext, SkillBrief, validate_plan
from .schemas import AgentRuntimeConfig
from .skill_dispatch import SkillDispatcher
from .utils import extract_explicit_skill_name

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Module-level helpers (extracted from class to avoid nested defs)
# ═══════════════════════════════════════════════════════════════════


async def _load_history(
    sid: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Load conversation history via ChatManager."""
    items = await ChatManager.get_conversation_history(sid, limit=limit)
    result: list[dict[str, Any]] = []
    for m in items:
        msg: dict[str, Any] = {
            "role": m.get("role"),
            "content": m.get("content", ""),
        }
        if m.get("conversation_id"):
            msg["conversation_id"] = m["conversation_id"]
        if m.get("tokens"):
            msg["tokens"] = m["tokens"]
        if m.get("tool_call_id"):
            msg["tool_call_id"] = m["tool_call_id"]
        if m.get("tool_calls"):
            msg["tool_calls"] = m["tool_calls"]
        result.append(msg)
    return result


async def _persist_tool_to_db(
    session_id: str,
    role: str,
    title: str,
    content: str,
    tool_call_id: str | None,
    tool_calls: list[dict] | None,
    *,
    conversation_id: str | None = None,
) -> None:
    """已由 chat_service.py 在 yield 侧统一持久化，此处保留为空操作。"""


def _detect_project_type(path: Path) -> str:
    """Detect project type from marker files."""
    markers = {
        "pyproject.toml": "python",
        "setup.py": "python",
        "requirements.txt": "python",
        "package.json": "node",
        "Cargo.toml": "rust",
        "go.mod": "go",
        "pom.xml": "java",
        "build.gradle": "java",
        "Gemfile": "ruby",
        "composer.json": "php",
    }
    for marker, ptype in markers.items():
        if (path / marker).exists():
            return ptype
    return ""


@dataclass
class SessionBundle:
    """Grouped per-session state — avoids two parallel LRU caches."""

    session_goal: SessionGoal
    infra: InfraService


# ═══════════════════════════════════════════════════════════════════
# Agent
# ═══════════════════════════════════════════════════════════════════


class MementoSAgent:
    """Memento-S Agent — thin orchestrator with skill-based task execution."""

    def __init__(
        self,
        *,
        skill_gateway: SkillGateway | None = None,
    ) -> None:
        self.llm = LLMClient()
        self._gateway = skill_gateway
        self._initialized = skill_gateway is not None

        self.infra: InfraService | None = None
        self.policy_manager = PolicyManager()
        self.skill_dispatcher: SkillDispatcher | None = None

        self._agent_profile: AgentProfile | None = None
        self._agent_profile_skill_hash: int = 0
        self._sessions: OrderedDict[str, SessionBundle] = OrderedDict()
        self._agent_config_raw: AgentRuntimeConfig | None = None
        self._init_lock = asyncio.Lock()
        self._reply_locks: dict[str, asyncio.Lock] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}

        self._on_skill_step_callback: Any | None = None

        if self._initialized and self._gateway is not None:
            self.skill_dispatcher = SkillDispatcher(
                skill_gateway=self._gateway,
            )

    @property
    def _agent_config(self) -> AgentRuntimeConfig:
        """Lazily build AgentRuntimeConfig from g_config on first access."""
        if self._agent_config_raw is None:
            self._agent_config_raw = self._build_agent_config()
        return self._agent_config_raw

    def _build_agent_config(self) -> AgentRuntimeConfig:
        """从 g_config.agent 读取运行时配置。"""
        from core.context.config import ContextManagerConfig
        cfg = g_config.load()
        return AgentRuntimeConfig(
            max_iterations=cfg.agent.max_iterations,
            context=ContextManagerConfig(),
        )

    def reload_llm_config(self) -> None:
        """重新加载 LLM 配置。"""
        self.llm.reload_config()

    def cancel(self, session_id: str) -> None:
        """发出取消信号，中止指定 session 的当前任务。"""
        if session_id not in self._cancel_events:
            self._cancel_events[session_id] = asyncio.Event()
        self._cancel_events[session_id].set()

    def is_cancelled(self, session_id: str) -> bool:
        """检查指定 session 是否已被请求取消。"""
        ev = self._cancel_events.get(session_id)
        return ev is not None and ev.is_set()

    def _clear_cancel(self, session_id: str) -> None:
        """清除取消标志（在新一轮 reply_stream 开始前调用）。"""
        ev = self._cancel_events.get(session_id)
        if ev is not None:
            ev.clear()

    def set_on_skill_step(self, callback: Any | None) -> None:
        """Set callback for skill execution step updates."""
        self._on_skill_step_callback = callback

    # ── Initialisation ───────────────────────────────────────────────

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            log_agent_phase("AGENT_INIT", "system", "Creating SkillGateway...")
            skill_config = SkillConfig.from_global_config()
            self._gateway = await SkillGateway.from_config(config=skill_config)
            self.skill_dispatcher = SkillDispatcher(
                skill_gateway=self._gateway,
            )
            mcp_cfg = g_mcp_config_manager.get_mcp_config()
            await tools_registry.bootstrap(mcp_config=mcp_cfg)
            self._agent_profile = await apm.build_profile(
                skill_gateway=self._gateway,
                config=g_config,
            )
            self._initialized = True

    async def _compute_skill_hash(self) -> int:
        if not self._gateway:
            return 0
        try:
            manifests = await self._gateway.discover()
            names = sorted(m.name for m in manifests)
            return hash(tuple(names))
        except Exception:
            return 0

    def _get_or_create_bundle(self, session_id: str) -> SessionBundle:
        """Get or create a SessionBundle with LRU eviction."""
        bundle = self._sessions.get(session_id)
        if bundle is not None:
            self._sessions.move_to_end(session_id)
            log_debug_marker(f"SessionBundle cache hit: {session_id}", level="debug")
            return bundle

        log_debug_marker(f"Creating new SessionBundle: {session_id}", level="debug")
        session_goal = SessionGoal()

        context_dir = g_config.paths.context_dir
        if context_dir is None:
            context_dir = Path.home() / "memento_s" / "context"
        session_dir = context_dir / "sessions" / session_id
        data_dir = context_dir

        ctx_config = self._agent_config.context

        logger.info(
            "[Agent] _get_or_create_bundle: sm_llm_update_interval={}, session_dir={}, context_dir={}",
            ctx_config.sm_llm_update_interval,
            session_dir, context_dir,
        )
        infra = InfraService(
            session_id=session_id,
            session_dir=session_dir,
            data_dir=data_dir,
            model=g_config.llm.current.model if g_config.llm.current else "",
            context_config=InfraContextConfig.from_core_context_config(ctx_config),
            sm_llm_update_interval=ctx_config.sm_llm_update_interval,
            skill_gateway=self._gateway,
            history_loader=partial(_load_history, session_id, 100),
            context_dir=context_dir,
        )

        bundle = SessionBundle(
            session_goal=session_goal,
            infra=infra,
        )
        self._sessions[session_id] = bundle
        max_ctx = self._agent_config.max_session_contexts
        if len(self._sessions) > max_ctx:
            removed = self._sessions.popitem(last=False)
            log_debug_marker(f"SessionBundle LRU evicted: {removed[0]}", level="debug")
        return bundle

    async def _refresh_profile_if_needed(self, session_id: str) -> None:
        current_hash = await self._compute_skill_hash()
        if (
            self._agent_profile is None
            or current_hash != self._agent_profile_skill_hash
        ):
            log_agent_phase(
                "PROFILE_REBUILD",
                session_id,
                f"hash changed: {self._agent_profile_skill_hash} -> {current_hash}",
            )
            self._agent_profile = await apm.build_profile(
                skill_gateway=self._gateway,
                config=g_config,
            )
            self._agent_profile_skill_hash = current_hash

    # ── Main entry point ─────────────────────────────────────────────

    async def reply_stream(
        self,
        session_id: str,
        user_content: str,
        history: list[dict[str, Any]] | None = None,
        conversation_id: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        await self._ensure_initialized()
        cfg = g_config

        if self.skill_dispatcher is None:
            raise RuntimeError("Agent initialisation failed: dispatcher unavailable")

        # Clear any stale cancel flag from a previous run
        self._clear_cancel(session_id)

        # Per-session lock to prevent concurrent reply_stream
        if session_id not in self._reply_locks:
            self._reply_locks[session_id] = asyncio.Lock()

        # Create SessionContext inside core/ — callers outside core/ only see strings
        context_dir = g_config.paths.context_dir
        if context_dir is None:
            context_dir = Path.home() / "memento_s" / "context"
        ctx = SessionContext.create(session_id, base_dir=context_dir)
        self.skill_dispatcher.set_context(ctx)

        if hasattr(self, "_on_skill_step_callback") and self._on_skill_step_callback:
            self.skill_dispatcher.set_on_skill_step(self._on_skill_step_callback)

        bundle = self._get_or_create_bundle(session_id)
        self.infra = bundle.infra

        # Wire infra into tool_dispatcher for recall_context
        self.skill_dispatcher._infra = self.infra

        self.skill_dispatcher.set_on_skills_changed(
            self.infra.context.session_memory.is_empty  # no-op, kept for compat
        )

        # Initialize SessionMemory (CC-style summary.md)
        await self.infra.session_memory.setup()
        logger.info(
            "[Agent] session_memory initialized: type={}, empty={}",
            type(self.infra.session_memory).__name__,
            self.infra.session_memory.is_empty(),
        )

        if history is None:
            history = await self.infra.context.load_history()

        session_goal = bundle.session_goal
        session_goal.refine(user_content)

        await self._refresh_profile_if_needed(session_id)

        run_id = new_run_id()
        max_iter = cfg.agent.max_iterations

        adapter = AGUIProtocolAdapter()
        emitter = RunEmitter(run_id, session_id, adapter)

        yield emitter.run_started(input_text=user_content)

        try:
            # ════════════════════════════════════════════════════════════
            # Phase 1: Intent Recognition
            # ════════════════════════════════════════════════════════════
            log_agent_phase(
                "INTENT_START", session_id, f"message_len={len(user_content)}"
            )

            intent = await recognize_intent(
                user_content,
                history,
                self.llm,
                self.infra.context,
                session_goal,
            )
            logger.info(
                "Intent: mode={}, task={}, shifted={}",
                intent.mode.value,
                intent.task,
                intent.intent_shifted,
            )

            yield emitter.intent_recognized(
                mode=intent.mode.value,
                task=intent.task,
            )

            # ════════════════════════════════════════════════════════════
            # Route: DIRECT / INTERRUPT → streaming reply
            # ════════════════════════════════════════════════════════════
            if intent.mode in (IntentMode.DIRECT, IntentMode.INTERRUPT):
                log_agent_phase("DIRECT_REPLY", session_id, f"mode={intent.mode.value}")
                messages = await self.infra.context.load_and_assemble(
                    current_message=user_content,
                    history=history,
                    agent_profile=self._agent_profile,
                    mode=intent.mode.value,
                    intent_shifted=intent.intent_shifted,
                    effective_context_window=self.llm.context_window,
                )
                total_tokens = self.infra.context.total_tokens
                yield emitter.step_started(step=1, name="direct_reply")
                result_info: dict[str, Any] = {}
                async for event in stream_and_finalize(
                    messages=messages,
                    llm=self.llm,
                    tools=None,
                    emitter=emitter,
                    step=1,
                    context_tokens=total_tokens,
                    session_ctx=self.infra.context,
                    result_info=result_info,
                ):
                    yield event
                return

            # ════════════════════════════════════════════════════════════
            # Route: CONFIRM → ask clarification question
            # ════════════════════════════════════════════════════════════
            if intent.mode == IntentMode.CONFIRM:
                log_agent_phase(
                    "CONFIRM_REPLY", session_id,
                    f"ambiguity={(intent.ambiguity or '')[:60]}"
                )
                question = (
                    intent.clarification_question
                    or intent.ambiguity
                    or "Could you please clarify?"
                )
                confirm_messages = await self.infra.context.load_and_assemble(
                    current_message=user_content,
                    history=history,
                    agent_profile=self._agent_profile,
                    mode="direct",
                    intent_shifted=False,
                    effective_context_window=self.llm.context_window,
                )
                confirm_messages.append({"role": "assistant", "content": question})
                total_tokens = self.infra.context.total_tokens
                yield emitter.step_started(step=1, name="confirm_question")
                msg_id = emitter.new_message_id()
                yield emitter.text_message_start(message_id=msg_id, role="assistant")
                yield emitter.text_delta(message_id=msg_id, delta=question)
                yield emitter.text_message_end(message_id=msg_id)
                yield emitter.step_finished(step=1, status=StepStatus.DONE)
                yield emitter.run_finished(
                    output_text=question,
                    reason=AgentFinishReason.FINAL_ANSWER,
                    context_tokens=total_tokens,
                )
                return

            # ════════════════════════════════════════════════════════════
            # Route: AGENTIC → plan → execute → reflect
            # ════════════════════════════════════════════════════════════
            session_goal.text = intent.task

            log_agent_phase("PLAN_START", session_id, f"goal={intent.task[:60]}")

            manifests = await self._gateway.route(intent.task) if self._gateway else []
            if not manifests and self._gateway:
                manifests = await self._gateway.discover()
            skill_briefs = [
                SkillBrief(
                    name=m.name,
                    description=m.description or "",
                    parameters=m.parameters,
                )
                for m in manifests
            ]
            plan_ctx = PlanContext(
                available_skills=skill_briefs,
                history_summary=self.infra.context.build_history_summary(
                    history,
                    max_rounds=self._agent_config.history_summary_max_rounds,
                    max_tokens=self._agent_config.history_summary_max_tokens,
                ),
            )
            task_plan = await generate_plan(
                goal=intent.task,
                context=plan_ctx,
                llm=self.llm,
            )
            task_plan = validate_plan(task_plan, {m.name for m in manifests})
            logger.info("Plan generated: {} steps", len(task_plan.steps))

            yield emitter.plan_generated(**task_plan.to_event_payload())

            # [ANALYSIS-LOG] Log the full validated plan as structured JSON
            import json as _json
            plan_dump = {
                "goal": task_plan.goal,
                "steps": [s.model_dump() for s in task_plan.steps],
            }
            logger.info(
                "[ANALYSIS-LOG] === PLAN_VALIDATED: {} step(s) ===\n{}",
                len(task_plan.steps),
                _json.dumps(plan_dump, indent=2, ensure_ascii=False),
            )

            messages = await self.infra.context.load_and_assemble(
                current_message=user_content,
                history=history,
                agent_profile=self._agent_profile,
                mode=intent.mode.value,
                intent_shifted=intent.intent_shifted,
                effective_context_window=self.llm.context_window,
            )

            local_skill_names = []
            if self._gateway:
                try:
                    manifests = await self._gateway.discover()
                    local_skill_names = [m.name for m in manifests]
                except Exception:
                    local_skill_names = []

            state = AgentRunState(
                config=self._agent_config,
                mode=intent.mode,
                task_plan=task_plan,
                messages=messages,
                explicit_skill_name=extract_explicit_skill_name(
                    user_content,
                    local_skill_names,
                ),
            )

            self.skill_dispatcher.set_step_summary_source(
                lambda: state.last_step_summary
            )

            total_tokens = self.infra.context.total_tokens
            tool_sink = ToolTranscriptSink(
                persister=partial(_persist_tool_to_db, session_id, conversation_id=conversation_id),
            )
            async for event in run_plan_execution(
                state=state,
                llm=self.llm,
                tool_dispatcher=self.skill_dispatcher,
                tool_schemas=self.skill_dispatcher.get_skill_tool_schemas(),
                session_goal_text=session_goal.text,
                emitter=emitter,
                user_content=user_content,
                max_iter=max_iter,
                ctx=self.infra.context,
                context_tokens=total_tokens,
            ):
                await tool_sink.handle(event)
                yield event

            # L1 save + 累积到 staging
            sm = self.infra.session_memory
            sm.save()
            if not sm.is_empty():
                await self.infra.context_memory.accumulate_session(sm.get_content())

                # 检查是否应触发自动整合
                engine = getattr(self.infra, "memory_engine", None)
                if engine is not None and engine.check_should_consolidate():
                    asyncio.create_task(engine.quick_run())

            # 触发 USER.md / SOUL.md 进化（完全异步，不阻塞会话结束流程）
            # 与 session memory 无关，conversation history 由 ChatManager 提供
            try:
                asyncio.create_task(self._trigger_evolution(session_id))
            except RuntimeError:
                pass

        except Exception as e:
            log_agent_phase(
                "RUN_ERROR",
                session_id,
                f"error={type(e).__name__}: {log_preview_long(str(e))}",
            )
            logger.exception("Agent run error")
            yield emitter.run_error(message=str(e))
            ctx_tokens = None
            if self.infra and hasattr(self.infra.context, "total_tokens"):
                ctx_tokens = self.infra.context.total_tokens
            yield emitter.run_finished(
                output_text=f"Error: {e}",
                reason=AgentFinishReason.ERROR,
                context_tokens=ctx_tokens,
            )

    async def _trigger_evolution(self, session_id: str) -> None:
        """会话结束触发 USER/SOUL 进化。evolver 内部已做 task 调度，此处仅转发。"""
        try:
            from core.agent_profile import apm
            apm.on_session_end(session_id)
        except Exception:
            pass
