# SPDX-License-Identifier: Apache-2.0
"""DreamLoop — Timer 协程 + 线程调度。"""

from __future__ import annotations

import asyncio
import threading
import uuid
from pathlib import Path

from infra.memory.impl.long_term_memory import LongTermMemory
from middleware.config.schemas.config_models import DreamConfig
from utils.logger import get_logger

from infra.memory.consolidation import MemoryConsolidationEngine
from .consolidator import DreamConsolidator

logger = get_logger(__name__)


class DreamLoop:
    """Dream 后台循环：Timer 协程驱动，每次 spawn 一个 daemon thread 执行整合。"""

    def __init__(
        self,
        memory_dir: str | Path | None = None,
        config: DreamConfig | None = None,
    ) -> None:
        from middleware.config import g_config

        if config is None and memory_dir is None:
            config = g_config.load().dream
        elif config is None:
            config = DreamConfig()
        self._cfg = config

        if memory_dir is None:
            cfg = g_config.load()
            if cfg.paths.context_dir is not None:
                memory_dir = cfg.paths.context_dir / "memory"

        if memory_dir is None:
            raise ValueError("memory_dir must be provided")

        if isinstance(memory_dir, str):
            memory_dir = Path(memory_dir)

        self._memory = LongTermMemory(memory_dir, model="")
        self._engine = MemoryConsolidationEngine(
            memory=memory_dir,
            config=g_config.load().memory,
        )
        self._consolidator = DreamConsolidator(
            memory=self._memory,
            engine=self._engine,
            config=self._cfg,
        )
        self._poll_interval = float(self._cfg.poll_interval_seconds)

    async def run_timer(self) -> None:
        """Timer 协程，永远运行，每 poll_interval 触发一次 Dream。

        协程跟随 app 生命周期，线程崩溃不影响协程自愈。
        """
        while True:
            await asyncio.sleep(self._poll_interval)
            self._spawn_dream_thread()

    def _spawn_dream_thread(self) -> None:
        """Spawn a daemon thread to run one Dream consolidation."""
        def _run() -> None:
            try:
                asyncio.run(self._consolidator.maybe_trigger())
            except Exception:
                logger.opt(exception=True).error("[Dream] consolidation thread failed")

        t = threading.Thread(target=_run, daemon=True, name=f"dream-{uuid.uuid4().hex[:8]}")
        t.start()
        logger.debug("[Dream] triggered, thread={}", t.name)
