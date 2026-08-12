# SPDX-License-Identifier: Apache-2.0
"""DreamConsolidator — Dream 触发器，只做 Gate 检查，核心整合委托给 engine."""

from __future__ import annotations

import time
from pathlib import Path

from infra.memory.impl.long_term_memory import LongTermMemory
from middleware.config.schemas.config_models import DreamConfig
from utils.logger import get_logger

from infra.memory.consolidation import MemoryConsolidationEngine

logger = get_logger(__name__)


class DreamConsolidator:
    """Dream 触发器 — 只做 Gate 检查，核心整合委托给 engine.

    Gate 逻辑:
      1. 距上次 Dream > dream.min_hours (默认 24h)
      2. staging 内容满足 engine 的 check_should_consolidate()
      3. 获取 .dream_lock 文件锁
    """

    def __init__(
        self,
        memory: LongTermMemory,
        engine: MemoryConsolidationEngine,
        config: DreamConfig,
    ) -> None:
        self._memory = memory
        self._engine = engine
        self._min_hours = config.min_hours
        self._scan_interval = config.scan_interval_seconds
        self._lock_path = memory._dir / ".dream_lock"
        self._last_scan_time: float = 0.0
        self._last_dream_time: float = 0.0

    async def maybe_trigger(self) -> None:
        """检查 gates 并触发 Dream。"""
        now = time.monotonic()

        if now - self._last_scan_time < self._scan_interval:
            return
        self._last_scan_time = now

        if self._last_dream_time > 0 and (now - self._last_dream_time) < self._min_hours * 3600:
            return

        if not self._staging_gate_passed():
            return
        if not self._acquire_lock():
            return

        try:
            context = self._build_deep_context()
            await self._engine.deep_run(context)
            self._last_dream_time = time.monotonic()
            logger.info("DreamConsolidator: deep_run completed")
        except Exception:
            logger.opt(exception=True).warning("DreamConsolidator: deep_run failed")
        finally:
            self._release_lock()

    def _staging_gate_passed(self) -> bool:
        """委托 engine 的 check_should_consolidate() 判断是否应触发 Dream。"""
        return self._engine.check_should_consolidate()

    def _build_deep_context(self):
        """构建 deep_run 的 context。"""
        from infra.memory.consolidation.engine import ConsolidationContext

        return ConsolidationContext(
            index_content=self._memory.get_index_content(),
            staging_content=self._memory.get_staging_content(),
            topics=self._memory.list_topics_with_content(),
            mode="deep",
        )

    def _acquire_lock(self) -> bool:
        """Filesystem lock to prevent concurrent Dream runs."""
        try:
            if self._lock_path.exists():
                lock_age = time.time() - self._lock_path.stat().st_mtime
                if lock_age < self._min_hours * 3600:
                    return False
                self._lock_path.unlink(missing_ok=True)

            self._lock_path.write_text(str(time.time()), encoding="utf-8")
            return True
        except OSError:
            return False

    def _release_lock(self) -> None:
        """Release the filesystem lock."""
        try:
            self._lock_path.unlink(missing_ok=True)
        except OSError:
            pass