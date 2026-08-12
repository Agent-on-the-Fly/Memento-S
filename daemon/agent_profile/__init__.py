# SPDX-License-Identifier: Apache-2.0
"""daemon.agent_profile — AgentProfile 异步进化 Daemon。

启动：
    from daemon.agent_profile import AgentProfileEvolverDaemon
    AgentProfileEvolverDaemon.start()

文件结构：
  backup.py       — SOUL.md / USER.md 写入前备份
  soul_evolver.py — SoulEvolver 进化引擎
  user_evolver.py — UserEvolver 进化引擎
  soul_prompts.py — SOUL.md 进化提示词
  user_prompts.py — USER.md 进化提示词
  orchestrator.py — AgentProfileEvolver 编排器（统一调度两者）
  constants.py    — 配置常量
"""

from __future__ import annotations

import asyncio

from utils.logger import get_logger

from .orchestrator import AgentProfileEvolver
from .soul_evolver import SoulEvolver
from .user_evolver import UserEvolver

logger = get_logger(__name__)

__all__ = [
    "AgentProfileEvolverDaemon",
    "AgentProfileEvolver",
    "SoulEvolver",
    "UserEvolver",
]


class AgentProfileEvolverDaemon:
    """USER.md + SOUL.md 进化 Daemon，Timer 协程驱动，幂等启动。"""

    _timer_task: asyncio.Task | None = None

    @classmethod
    def start(cls) -> None:
        if cls._timer_task is not None:
            logger.debug("[AgentProfileEvolverDaemon] already started, skipping")
            return

        from middleware.llm.llm_client import LLMClient

        llm = LLMClient()
        evolver = AgentProfileEvolver(llm_client=llm)

        cls._timer_task = asyncio.create_task(evolver.start())
        cls._evolver = evolver
        logger.info("[AgentProfileEvolverDaemon] started")

    @classmethod
    def stop(cls) -> None:
        if cls._timer_task is None:
            return
        cls._timer_task.cancel()
        cls._timer_task = None
        logger.info("[AgentProfileEvolverDaemon] stopped")

    @classmethod
    def get_evolver(cls) -> AgentProfileEvolver | None:
        return getattr(cls, "_evolver", None)
