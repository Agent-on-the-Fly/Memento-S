"""Memento-S Agent — thin orchestration layer.

All heavy logic lives in ``phases/``, ``core/context/``, and ``utils.py``.
This file is responsible only for initialisation and the top-level
``reply_stream`` coordination.

Routing:
  DIRECT / INTERRUPT → simple_reply  (no tools, no plan)
  AGENTIC            → plan → execute → reflect
"""

from __future__ import annotations

from collections import OrderedDict
from functools import partial
from typing import Any, AsyncGenerator

from core.context import ContextManager
from core.manager import SessionManager
from core.manager.conversation_manager import ConversationManager
from core.manager.session_context import EnvironmentSnapshot, SessionContext
from core.skill.gateway import SkillGateway
from core.skill.provider import SkillProvider
from middleware.config import g_config
from middleware.llm import LLMClient
from utils.debug_logger import log_agent_phase, log_debug_marker
from utils.logger import get_logger

from .agent_profile import AgentProfile
from .emitters import inject_context_tokens, stream_and_finalize
from .phases import AgentRunState, IntentMode, generate_plan, recognize_intent, run_plan_execution
from .policies import PolicyManager
from .schemas import AgentConfig
from .stream_output import AGUIEventType, build_event, new_run_id
from .tools import AGENT_TOOL_SCHEMAS, ToolDispatcher
from .utils import extract_explicit_skill_name

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Module-level helpers (extracted from class to avoid nested defs)
# ═══════════════════════════════════════════════════════════════════


async def _load_history(
    conv_mgr: ConversationManager, sid: str, limit: int,
) -> list[dict[str, Any]]:
    """Load conversation history via ConversationManager (used as history_loader)."""
    items = await conv_mgr.get_conversation_history(session_id=sid, limit=limit)
    return [{"role": m.get("role"), "content": m.get("content", "")} for m in items]


def _build_plan_context(
    session_ctx: SessionContext,
    history: list[dict[str, Any]] | None,
) -> list[str]:
    """Build context strings for plan generation."""
    parts: list[str] = []
    if session_ctx.environment.cwd:
        parts.append(f"Working directory: {session_ctx.environment.cwd}")
    if session_ctx.environment.project_type:
        parts.append(f"Project type: {session_ctx.environment.project_type}")
    if history:
        recent = [
            f"{m.get('role', '')}: {str(m.get('content', ''))[:100]}"
            for m in history[-3:]
        ]
        parts.append("Recent conversation:\n" + "\n".join(recent))
    return parts


# ═══════════════════════════════════════════════════════════════════
# Agent
# ═══════════════════════════════════════════════════════════════════


class MementoSAgent:
    """Memento-S Agent — thin orchestrator with skill-based task execution."""

    def __init__(
        self,
        *,
        skill_gateway: SkillGateway | None = None,
        session_manager: SessionManager | None = None,
    ) -> None:
        self.llm = LLMClient()
        self._gateway = skill_gateway
        self._initialized = skill_gateway is not None

        self.session_manager = session_manager or SessionManager()
        self.context_manager: ContextManager | None = None
        self.policy_manager = PolicyManager()
        self.tool_dispatcher: ToolDispatcher | None = None

        self._agent_profile: AgentProfile | None = None
        self._agent_profile_skill_hash: int = 0
        self._session_contexts: OrderedDict[str, SessionContext] = OrderedDict()
        self._context_managers: OrderedDict[str, ContextManager] = OrderedDict()
        self._max_session_contexts: int = 100
        self._agent_config = AgentConfig()

        if self._initialized and self._gateway is not None:
            self.tool_dispatcher = ToolDispatcher(
                policy_manager=self.policy_manager,
                skill_gateway=self._gateway,
            )

    # ── Initialisation ───────────────────────────────────────────────

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        log_agent_phase("AGENT_INIT", "system", "Creating SkillProvider...")
        self._gateway = await SkillProvider.create_default()
        self.tool_dispatcher = ToolDispatcher(
            policy_manager=self.policy_manager,
            skill_gateway=self._gateway,
        )
        self._agent_profile = AgentProfile.build_from_context(
            skill_gateway=self._gateway, config=g_config,
        )
        self._initialized = True

    def _compute_skill_hash(self) -> int:
        if not self._gateway:
            return 0
        try:
            names = sorted(m.name for m in self._gateway.discover())
            return hash(tuple(names))
        except Exception:
            return 0

    def _get_or_create_session_ctx(self, session_id: str) -> SessionContext:
        ctx = self._session_contexts.get(session_id)
        if ctx is not None:
            self._session_contexts.move_to_end(session_id)
            log_debug_marker(f"Session cache hit: {session_id}", level="debug")
            return ctx

        log_debug_marker(f"Creating new session context: {session_id}", level="debug")
        ctx = SessionContext(
            session_id=session_id,
            environment=EnvironmentSnapshot.capture(),
        )
        self._session_contexts[session_id] = ctx
        if len(self._session_contexts) > self._max_session_contexts:
            removed = self._session_contexts.popitem(last=False)
            log_debug_marker(f"Session LRU evicted: {removed[0]}", level="debug")
        return ctx

    def _refresh_profile_if_needed(self, session_id: str) -> None:
        current_hash = self._compute_skill_hash()
        if self._agent_profile is None or current_hash != self._agent_profile_skill_hash:
            log_agent_phase(
                "PROFILE_REBUILD", session_id,
                f"hash changed: {self._agent_profile_skill_hash} -> {current_hash}",
            )
            self._agent_profile = AgentProfile.build_from_context(
                skill_gateway=self._gateway, config=g_config,
            )
            self._agent_profile_skill_hash = current_hash

    def _get_or_create_context_manager(self, session_id: str) -> ContextManager:
        """Session 级别缓存 ContextManager，保留 scratchpad 状态。"""
        ctx = self._context_managers.get(session_id)
        if ctx is not None:
            self._context_managers.move_to_end(session_id)
            log_debug_marker(f"ContextManager cache hit: {session_id}", level="debug")
            return ctx

        log_debug_marker(f"Creating new ContextManager: {session_id}", level="debug")
        conv_mgr = ConversationManager()
        ctx = ContextManager(
            session_id=session_id,
            config=self._agent_config.context,
            skill_gateway=self._gateway,
            history_loader=partial(_load_history, conv_mgr),
        )
        self._context_managers[session_id] = ctx
        if len(self._context_managers) > self._max_session_contexts:
            removed = self._context_managers.popitem(last=False)
            log_debug_marker(f"ContextManager LRU evicted: {removed[0]}", level="debug")
        return ctx

    # ── Main entry point ─────────────────────────────────────────────

    async def reply_stream(
        self,
        session_id: str,
        user_content: str,
        history: list[dict[str, Any]] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        await self._ensure_initialized()
        cfg = g_config

        if self.tool_dispatcher is None:
            raise RuntimeError("Agent initialisation failed: dispatcher unavailable")

        self.tool_dispatcher.set_session_id(session_id)

        self.context_manager = self._get_or_create_context_manager(session_id)

        # 加载一次，传给 intent + assemble（避免重复 DB 查询）
        if history is None:
            history = await self.context_manager.load_history()

        session_ctx = self._get_or_create_session_ctx(session_id)
        session_ctx.update_goal(user_content)
        self._refresh_profile_if_needed(session_id)

        run_id = new_run_id()
        max_iter = cfg.agent.max_iterations

        yield build_event(
            AGUIEventType.RUN_STARTED, run_id, session_id,
            inputText=user_content,
        )

        try:
            # ════════════════════════════════════════════════════════════
            # Phase 1: Intent Recognition
            # ════════════════════════════════════════════════════════════
            log_agent_phase("INTENT_START", session_id, f"message_len={len(user_content)}")

            intent = await recognize_intent(
                user_content, history, self.llm, self.context_manager,
                session_context=session_ctx, config=self._agent_config,
            )
            logger.info(
                "Intent: mode={}, task={}, shifted={}",
                intent.mode.value, intent.task, intent.intent_shifted,
            )

            yield build_event(
                AGUIEventType.INTENT_RECOGNIZED, run_id, session_id,
                mode=intent.mode.value, task=intent.task,
            )

            # ════════════════════════════════════════════════════════════
            # Route: DIRECT / INTERRUPT → streaming reply 
            # ════════════════════════════════════════════════════════════
            if intent.mode in (IntentMode.DIRECT, IntentMode.INTERRUPT):
                log_agent_phase("DIRECT_REPLY", session_id, f"mode={intent.mode.value}")
                messages = await self.context_manager.assemble_messages(
                    history=history, current_message=user_content,
                    media=None,
                    matched_skills_context="",
                    agent_profile=self._agent_profile,
                    session_context=session_ctx,
                    mode=intent.mode.value,
                    intent_shifted=intent.intent_shifted,
                )
                total_tokens = self.context_manager.total_tokens
                yield build_event(
                    AGUIEventType.STEP_STARTED, run_id, session_id,
                    step=1, name="direct_reply",
                )
                async for event in stream_and_finalize(
                    messages=messages, llm=self.llm, tools=None,
                    run_id=run_id, session_id=session_id, step=1,
                    session_ctx=session_ctx,
                    session_manager=self.session_manager,
                ):
                    yield inject_context_tokens(event, total_tokens)
                return

            # ════════════════════════════════════════════════════════════
            # Route: AGENTIC → plan → execute → reflect
            # ════════════════════════════════════════════════════════════
            log_agent_phase("PLAN_START", session_id, f"goal={intent.task[:60]}")

            plan_context = _build_plan_context(session_ctx, history)
            task_plan = await generate_plan(
                goal=intent.task,
                context="\n".join(plan_context),
                llm=self.llm,
            )
            logger.info("Plan generated: {} steps", len(task_plan.steps))

            session_ctx.set_plan([
                f"Step {s.step_id}: {s.action}" for s in task_plan.steps
            ])

            yield build_event(
                AGUIEventType.PLAN_GENERATED, run_id, session_id,
                **task_plan.to_event_payload(),
            )

            messages = await self.context_manager.assemble_messages(
                history=history, current_message=user_content,
                media=None,
                matched_skills_context="",
                agent_profile=self._agent_profile,
                session_context=session_ctx,
                mode=intent.mode.value,
                intent_shifted=intent.intent_shifted,
            )

            state = AgentRunState(
                config=self._agent_config,
                mode=intent.mode,
                task_plan=task_plan,
                messages=messages,
                explicit_skill_name=extract_explicit_skill_name(
                    user_content,
                    self._gateway.discover if self._gateway else lambda: [],
                ),
            )

            total_tokens = self.context_manager.total_tokens
            async for event in run_plan_execution(
                state=state, llm=self.llm,
                tool_dispatcher=self.tool_dispatcher,
                tool_schemas=list(AGENT_TOOL_SCHEMAS),
                session_ctx=session_ctx, session_id=session_id,
                run_id=run_id, user_content=user_content,
                max_iter=max_iter,
                session_manager=self.session_manager,
                ctx=self.context_manager,
            ):
                yield inject_context_tokens(event, total_tokens)

        except Exception as e:
            log_agent_phase(
                "RUN_ERROR", session_id,
                f"error={type(e).__name__}: {str(e)[:100]}",
            )
            logger.exception("Agent run error")
            yield build_event(
                AGUIEventType.RUN_ERROR, run_id, session_id, message=str(e),
            )
