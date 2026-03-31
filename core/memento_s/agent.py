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
from typing import Any, AsyncGenerator

from core.context import ContextManager
from core.context.block import make_event
from core.manager.session_context import EnvironmentSnapshot, SessionContext
from core.protocol import (
    AGUIProtocolAdapter,
    AgentFinishReason,
    RunEmitter,
    StepStatus,
    ToolTranscriptSink,
    new_run_id,
)
from shared.chat import ChatManager
from core.skill.gateway import SkillGateway
from core.skill.config import SkillConfig
from middleware.config import g_config
from middleware.llm import LLMClient
from middleware.llm.embedding_client import EmbeddingClient
from middleware.storage.vector_storage import VectorStorage, SQLITE_VEC_AVAILABLE
from utils.debug_logger import log_agent_phase, log_debug_marker
from utils.logger import get_logger

from .agent_profile import AgentProfile
from .finalize import stream_and_finalize
from .phases import (
    AgentRunState,
    IntentMode,
    generate_plan,
    recognize_intent,
    run_plan_execution,
)
from .phases.planning import PlanContext, SkillBrief, validate_plan
from core.shared import PolicyManager
from .schemas import AgentConfig
from .tools import AGENT_TOOL_SCHEMAS, ToolDispatcher
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
) -> None:
    """Persist a tool-call or tool-result message to the database."""
    await ChatManager.create_conversation(
        session_id=session_id,
        role=role,
        title=title,
        content=content,
        tool_call_id=tool_call_id,
        tool_calls=tool_calls,
    )


def _build_plan_context(
    session_ctx: SessionContext,
    history: list[dict[str, Any]] | None,
) -> list[str]:
    """Build context strings for plan generation.

    Keep planning context path-agnostic to avoid inducing absolute-path hallucinations.
    """
    parts: list[str] = []
    parts.append(
        "Path policy: avoid absolute paths in planning; prefer session-root relative paths or aliases."
    )
    if session_ctx.environment.project_type:
        parts.append(f"Project type: {session_ctx.environment.project_type}")
    if history:
        recent = [
            f"{m.get('role', '')}: {str(m.get('content', ''))[:100]}"
            for m in history[-3:]
        ]
        parts.append("Recent conversation:\n" + "\n".join(recent))
    return parts


@dataclass
class SessionBundle:
    """Grouped per-session state — avoids two parallel LRU caches."""

    session_ctx: SessionContext
    context_mgr: ContextManager


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

        self.context_manager: ContextManager | None = None
        self.policy_manager = PolicyManager()
        self.tool_dispatcher: ToolDispatcher | None = None

        self._agent_profile: AgentProfile | None = None
        self._agent_profile_skill_hash: int = 0
        self._sessions: OrderedDict[str, SessionBundle] = OrderedDict()
        self._agent_config = AgentConfig()
        self._init_lock = asyncio.Lock()

        # Callback for skill execution step updates (GUI real-time display)
        self._on_skill_step_callback: Any | None = None

        if self._initialized and self._gateway is not None:
            self.tool_dispatcher = ToolDispatcher(
                skill_gateway=self._gateway,
            )

    def reload_llm_config(self) -> None:
        """重新加载 LLM 配置。

        当模型配置发生变化（如删除模型、切换模型）时调用此方法，
        确保后续的 LLM 调用使用最新的配置。
        """
        self.llm.reload_config()

    def set_on_skill_step(self, callback: Any | None) -> None:
        """Set callback for skill execution step updates.

        Called during skill execution for each tool call to enable
        real-time GUI display of execution progress.

        Args:
            callback: Async function(step_number, tool_name, status, signal, summary)
        """
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
            self.tool_dispatcher = ToolDispatcher(
                skill_gateway=self._gateway,
            )
            self._agent_profile = await AgentProfile.build_from_context(
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
        """Get or create a SessionBundle (session_ctx + context_mgr + conv_mgr) with LRU eviction."""
        bundle = self._sessions.get(session_id)
        if bundle is not None:
            self._sessions.move_to_end(session_id)
            log_debug_marker(f"SessionBundle cache hit: {session_id}", level="debug")
            return bundle

        log_debug_marker(f"Creating new SessionBundle: {session_id}", level="debug")
        session_ctx = SessionContext(
            session_id=session_id,
            environment=EnvironmentSnapshot.capture(),
        )

        embedding_client = None
        vector_storage = None
        ctx_config = self._agent_config.context
        if ctx_config.embedding_enabled:
            embedding_client, vector_storage = self._init_embedding()

        context_mgr = ContextManager(
            session_id=session_id,
            config=ctx_config,
            skill_gateway=self._gateway,
            history_loader=_load_history,
            embedding_client=embedding_client,
            vector_storage=vector_storage,
        )

        bundle = SessionBundle(
            session_ctx=session_ctx,
            context_mgr=context_mgr,
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
            self._agent_profile = await AgentProfile.build_from_context(
                skill_gateway=self._gateway,
                config=g_config,
            )
            self._agent_profile_skill_hash = current_hash

    @staticmethod
    def _init_embedding() -> tuple[EmbeddingClient | None, VectorStorage | None]:
        """初始化 embedding 客户端和向量存储。"""
        try:
            if not SQLITE_VEC_AVAILABLE:
                logger.warning("sqlite-vec not available, embedding disabled")
                return None, None

            client = EmbeddingClient.from_config()
            if not client._config.base_url:
                logger.info("Embedding service not configured, embedding disabled")
                return None, None

            if g_config.paths.data_dir is None:
                logger.warning("data_dir not configured, embedding disabled")
                return None, None

            db_path = g_config.paths.data_dir / "memento.db"
            storage = VectorStorage(
                db_path=db_path,
                table_name="conversation_embeddings",
                id_column="conversation_id",
            )
            asyncio.ensure_future(storage.init())

            # 设置 embedding 到 ChatManager
            ChatManager.initialize(embedding_client=client, vector_storage=storage)
            logger.info("Embedding enabled for conversation persistence and retrieval")
            return client, storage
        except Exception as e:
            logger.warning("Failed to initialize embedding: {}", e)
            return None, None

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

        # Set up skill execution step callback for real-time GUI updates
        if hasattr(self, "_on_skill_step_callback") and self._on_skill_step_callback:
            self.tool_dispatcher.set_on_skill_step(self._on_skill_step_callback)

        bundle = self._get_or_create_bundle(session_id)
        self.context_manager = bundle.context_mgr
        self.tool_dispatcher.set_on_skills_changed(
            self.context_manager.invalidate_skills_cache
        )

        if history is None:
            history = await self.context_manager.load_history(
                current_user_message=user_content
            )

        session_ctx = bundle.session_ctx
        session_ctx.update_goal(
            user_content
        )  # increments turn_count; sets goal on first turn only

        # Start new block + update runtime state on user input
        self.context_manager.start_new_block(user_content)
        rs = self.context_manager.runtime_state
        rs.on_user_input(
            user_content,
            block_id=self.context_manager.active_block.block_id
            if self.context_manager.active_block
            else "",
        )
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
                self.context_manager,
                session_context=session_ctx,
                config=self._agent_config,
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
                if self.context_manager.bounded_prompt_enabled:
                    messages = await self.context_manager.assemble_messages_bounded(
                        current_message=user_content,
                        media=None,
                        matched_skills_context="",
                        agent_profile=self._agent_profile,
                        session_context=session_ctx,
                        mode=intent.mode.value,
                        intent_shifted=intent.intent_shifted,
                        effective_context_window=self.llm.context_window,
                    )
                else:
                    messages = await self.context_manager.assemble_messages(
                        history=history,
                        current_message=user_content,
                        media=None,
                        matched_skills_context="",
                        agent_profile=self._agent_profile,
                        session_context=session_ctx,
                        mode=intent.mode.value,
                        intent_shifted=intent.intent_shifted,
                        effective_context_window=self.llm.context_window,
                    )
                total_tokens = self.context_manager.total_tokens
                yield emitter.step_started(step=1, name="direct_reply")
                async for event in stream_and_finalize(
                    messages=messages,
                    llm=self.llm,
                    tools=None,
                    emitter=emitter,
                    step=1,
                    session_ctx=session_ctx,
                    context_tokens=total_tokens,
                ):
                    yield event

                # Persist runtime state for DIRECT/INTERRUPT
                rs.on_run_finished()
                self.context_manager.sync_and_save_runtime_state(
                    session_ctx=session_ctx,
                )
                return

            # ════════════════════════════════════════════════════════════
            # Route: CONFIRM → ask clarification question
            # ════════════════════════════════════════════════════════════
            if intent.mode == IntentMode.CONFIRM:
                log_agent_phase(
                    "CONFIRM_REPLY", session_id, f"ambiguity={intent.ambiguity[:60]}"
                )
                question = (
                    intent.clarification_question
                    or intent.ambiguity
                    or "Could you please clarify?"
                )
                if self.context_manager.bounded_prompt_enabled:
                    confirm_messages = (
                        await self.context_manager.assemble_messages_bounded(
                            current_message=user_content,
                            media=None,
                            matched_skills_context="",
                            agent_profile=self._agent_profile,
                            session_context=session_ctx,
                            mode="direct",
                            intent_shifted=False,
                            effective_context_window=self.llm.context_window,
                        )
                    )
                else:
                    confirm_messages = await self.context_manager.assemble_messages(
                        history=history,
                        current_message=user_content,
                        media=None,
                        matched_skills_context="",
                        agent_profile=self._agent_profile,
                        session_context=session_ctx,
                        mode="direct",
                        intent_shifted=False,
                        effective_context_window=self.llm.context_window,
                    )
                confirm_messages.append({"role": "assistant", "content": question})
                total_tokens = self.context_manager.total_tokens
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
            session_ctx.session_goal = intent.task

            log_agent_phase("PLAN_START", session_id, f"goal={intent.task[:60]}")

            manifests = await self._gateway.discover() if self._gateway else []
            skill_briefs = [
                SkillBrief(
                    name=m.name,
                    description=m.description or "",
                    parameters=m.parameters,
                )
                for m in manifests
            ]
            env = session_ctx.environment
            plan_ctx = PlanContext(
                environment_summary=f"Project: {env.project_type}, OS: {env.os_info}",
                available_skills=skill_briefs,
                history_summary=self.context_manager.build_history_summary(
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

            # Update runtime state: plan generated
            rs.on_plan_generated(step_count=len(task_plan.steps))

            # Record plan event in block
            plan_text = "; ".join(s.action for s in task_plan.steps)
            self.context_manager.append_block_event(make_event("plan", text=plan_text))

            yield emitter.plan_generated(**task_plan.to_event_payload())

            if self.context_manager.bounded_prompt_enabled:
                messages = await self.context_manager.assemble_messages_bounded(
                    current_message=user_content,
                    media=None,
                    matched_skills_context="",
                    agent_profile=self._agent_profile,
                    session_context=session_ctx,
                    mode=intent.mode.value,
                    intent_shifted=intent.intent_shifted,
                    effective_context_window=self.llm.context_window,
                )
            else:
                messages = await self.context_manager.assemble_messages(
                    history=history,
                    current_message=user_content,
                    media=None,
                    matched_skills_context="",
                    agent_profile=self._agent_profile,
                    session_context=session_ctx,
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
            state.sync_plan_state(session_ctx)

            total_tokens = self.context_manager.total_tokens
            tool_sink = ToolTranscriptSink(
                persister=partial(_persist_tool_to_db, session_id),
            )
            async for event in run_plan_execution(
                state=state,
                llm=self.llm,
                tool_dispatcher=self.tool_dispatcher,
                tool_schemas=list(AGENT_TOOL_SCHEMAS),
                session_ctx=session_ctx,
                emitter=emitter,
                user_content=user_content,
                max_iter=max_iter,
                ctx=self.context_manager,
                context_tokens=total_tokens,
            ):
                await tool_sink.handle(event)
                yield event

            # Persist runtime state after AGENTIC execution
            rs.on_run_finished()
            self.context_manager.sync_and_save_runtime_state(
                agent_run_state=state,
                session_ctx=session_ctx,
            )

        except Exception as e:
            log_agent_phase(
                "RUN_ERROR",
                session_id,
                f"error={type(e).__name__}: {str(e)[:100]}",
            )
            logger.exception("Agent run error")
            yield emitter.run_error(message=str(e))
            ctx_tokens = None
            if self.context_manager and hasattr(self.context_manager, "total_tokens"):
                ctx_tokens = self.context_manager.total_tokens
            yield emitter.run_finished(
                output_text=f"Error: {e}",
                reason=AgentFinishReason.ERROR,
                context_tokens=ctx_tokens,
            )
