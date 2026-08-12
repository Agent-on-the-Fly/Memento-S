# SPDX-License-Identifier: Apache-2.0
"""AgentProfileEvolver — USER.md + SOUL.md 异步进化编排器。

定时循环 + 会话结束触发，对标 Hermes Honcho 的 peer-card 机制。

文件操作通过 core/agent_profile/AgentProfileManager (Facade) 统一访问。
推理委托给 UserEvolver（USER）和 SoulEvolver（SOUL）。
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING

from utils.logger import get_logger

from core.agent_profile import apm
from daemon.agent_profile.constants import (
    EVOLVE_INTERVAL_SECONDS,
    EVOLVE_SOUL_INTERVAL_SECONDS,
    INPUT_MAX_CHARS,
    MAX_FACTS_PER_SESSION,
    MAX_FACTS_PER_TIMED,
    MSG_TRUNCATE_CHARS,
)
from daemon.agent_profile.soul_evolver import SoulEvolver
from core.agent_profile.soul_manager import SoulManager
from daemon.agent_profile.user_evolver import UserEvolver

if TYPE_CHECKING:
    from middleware.llm.llm_client import LLMClient

logger = get_logger(__name__)


class AgentProfileEvolver:
    """USER.md + SOUL.md 异步进化器。

    USER.md: 会话结束即时触发 + 每小时定时触发，保守追加事实。
    SOUL.md: 跨会话收集 + 每 2 小时定时触发，需满足跨会话验证才更新。
    """

    def __init__(self, llm_client: "LLMClient") -> None:
        self._llm = llm_client
        self._user_evolver = UserEvolver(llm_client)
        self._soul_manager = SoulManager()
        self._soul_evolver = SoulEvolver(llm_client, self._soul_manager)
        self._running = False
        self._task: asyncio.Task | None = None
        self._soul_task: asyncio.Task | None = None
        self._last_run: datetime | None = None
        self._pending_session_ids: set[str] = set()

    # ── 启动 / 停止 ─────────────────────────────────────────────────

    async def start(self) -> None:
        """启动后台定时进化循环（USER 和 SOUL 各自独立间隔）。"""
        if self._running:
            return
        self._running = True
        # 启动时不做 seed，避免大量 LLM 调用；依赖定时循环 + 会话结束触发
        # await self._soul_evolver.seed_from_history()
        self._task = asyncio.create_task(self._run_loop())
        self._soul_task = asyncio.create_task(self._run_soul_loop())
        logger.info(
            "[AgentProfileEvolver] started (user_interval={}s, soul_interval={}s)",
            EVOLVE_INTERVAL_SECONDS,
            EVOLVE_SOUL_INTERVAL_SECONDS,
        )

    async def stop(self) -> None:
        """停止所有后台进化循环。"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._soul_task:
            self._soul_task.cancel()
            try:
                await self._soul_task
            except asyncio.CancelledError:
                pass
        logger.info("[AgentProfileEvolver] stopped")

    # ── USER.md 定时循环 ────────────────────────────────────────────

    async def _run_loop(self) -> None:
        # 启动时先 sleep 一个 interval，避免启动立即触发大量 LLM 调用
        await asyncio.sleep(EVOLVE_INTERVAL_SECONDS)
        while self._running:
            try:
                await self._evolve_user_timed()
            except Exception:
                logger.exception("[AgentProfileEvolver] evolve failed")
            await asyncio.sleep(EVOLVE_INTERVAL_SECONDS)

    # ── SOUL.md 定时循环 ────────────────────────────────────────────

    async def _run_soul_loop(self) -> None:
        # 启动时先 sleep 一个 interval，避免启动立即触发（即使 seed 被注释也会执行一次）
        await asyncio.sleep(EVOLVE_SOUL_INTERVAL_SECONDS)
        while self._running:
            try:
                await self._soul_evolver.evolve_timed()
            except Exception:
                logger.exception("[SoulEvolver] timed evolve failed")
            await asyncio.sleep(EVOLVE_SOUL_INTERVAL_SECONDS)
            if not self._running:
                break

    # ── 会话结束触发 ───────────────────────────────────────────────

    def on_session_end(self, session_id: str) -> None:
        """会话结束时调用，零成本 enqueue + 调度异步执行，不阻塞调用方。"""
        self._pending_session_ids.add(session_id)
        asyncio.create_task(self._soul_evolver.evolve_session(session_id))
        asyncio.create_task(self._evolve_pending())

    async def _evolve_pending(self) -> None:
        pending = list(self._pending_session_ids)
        self._pending_session_ids.clear()
        for sid in pending:
            try:
                await self._evolve_user_session(sid)
            except Exception:
                logger.exception("[AgentProfileEvolver] session evolve failed: {}", sid)

    # ── USER.md 进化逻辑 ───────────────────────────────────────────

    async def _evolve_user_timed(self) -> None:
        """定时进化：加载近期会话，提取用户事实，写入 USER.md。"""
        from shared.chat import ChatManager

        sessions = await ChatManager.list_sessions(limit=20)
        now = datetime.now()
        for session in sessions:
            ts = session.updated_at if hasattr(session, "updated_at") else None
            if ts and (now - ts).total_seconds() < 86400:
                await self._evolve_user_session(session.id, max_facts=MAX_FACTS_PER_TIMED)
        logger.info("[AgentProfileEvolver] timed evolve done")

    async def _evolve_user_session(
        self,
        session_id: str,
        max_facts: int = MAX_FACTS_PER_SESSION,
    ) -> None:
        """单个会话的 USER.md 进化（通过 Facade 访问 USER.md）。"""
        from shared.chat import ChatManager

        history = await ChatManager.get_conversation_history(session_id, limit=50)
        if len(history) < 2:
            return
        transcript = self._format_transcript(history)
        facts = await self._user_evolver.extract(
            transcript,
            existing_context=apm.load_user_context(),
            max_facts=max_facts,
        )
        if facts:
            added = apm.append_user_facts(facts)
            if added:
                total = sum(len(v) for v in facts.values())
                logger.info(
                    "[AgentProfileEvolver] evolved {} facts ({} sections) from session {}",
                    total,
                    list(facts.keys()),
                    session_id,
                )
                self._last_run = datetime.now()

    def _format_transcript(self, history: list[dict]) -> str:
        lines = []
        for msg in history[-20:]:
            role = msg.get("role", "")
            content = msg.get("content", "")[:MSG_TRUNCATE_CHARS]
            lines.append(f"[{role}] {content}")
        transcript = "\n\n".join(lines)
        if len(transcript) > INPUT_MAX_CHARS:
            transcript = transcript[:INPUT_MAX_CHARS]
        return transcript
