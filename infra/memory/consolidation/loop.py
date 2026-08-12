# SPDX-License-Identifier: Apache-2.0
"""AutoConsolidationLoop — 独立于 Dream 的自动整合后台任务。"""

from __future__ import annotations

import asyncio

from utils.logger import get_logger

logger = get_logger(__name__)


class AutoConsolidationLoop:
    """独立于 Dream 的自动整合循环。

    每隔 poll_interval_seconds 检查一次 staging 内容，
    满足阈值时触发 quick_run()。
    """

    def __init__(
        self,
        engine: "MemoryConsolidationEngine",
        poll_interval_seconds: float = 60.0,
    ) -> None:
        self._engine = engine
        self._poll_interval = poll_interval_seconds
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        """启动后台整合循环。"""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "AutoConsolidationLoop started with poll_interval={}s",
            self._poll_interval,
        )

    async def stop(self) -> None:
        """停止后台整合循环。"""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("AutoConsolidationLoop stopped")

    async def _run_loop(self) -> None:
        """主循环。"""
        while self._running:
            try:
                await asyncio.sleep(self._poll_interval)
                if self._engine.check_should_consolidate():
                    try:
                        await self._engine.quick_run()
                    except Exception:
                        logger.opt(exception=True).warning(
                            "AutoConsolidationLoop: quick_run failed"
                        )
            except asyncio.CancelledError:
                break
            except Exception:
                logger.opt(exception=True).warning(
                    "AutoConsolidationLoop: unexpected error in loop"
                )
