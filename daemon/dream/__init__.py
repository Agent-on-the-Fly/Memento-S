# SPDX-License-Identifier: Apache-2.0
"""daemon.dream — Dream 后台整合 Daemon。

启动（bootstrap.py 中调用）：
    from daemon.dream import DreamDaemon, DreamConfig
    DreamDaemon.start(config=DreamConfig(...))

配置字段：
    DreamConfig.enabled                # bool，默认 True
    DreamConfig.min_hours              # int，默认 24
    DreamConfig.poll_interval_seconds  # int，默认 600
    DreamConfig.scan_interval_seconds  # int，默认 600

注意：触发阈值已统一由 MemoryConsolidationConfig.min_staging_sessions 管理，
Dream 不再单独维护 min_sessions 阈值。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from middleware.config.schemas.config_models import DreamConfig
from utils.logger import get_logger

from .consolidator import DreamConsolidator
from .loop import DreamLoop

logger = get_logger(__name__)

__all__ = ["DreamDaemon", "DreamConsolidator", "DreamLoop", "DreamConfig"]


class DreamDaemon:
    """Dream 后台 Daemon，Timer 协程驱动，幂等启动。"""

    _timer_task: asyncio.Task[None] | None = None

    @classmethod
    def start(
        cls,
        config: DreamConfig | None = None,
        memory_dir: Path | str | None = None,
    ) -> None:
        """在后台启动 DreamDaemon Timer 协程（幂等调用）。

        Args:
            config: DreamConfig 实例，若为 None 则从 g_config.dream 读取
            memory_dir: 可选，覆盖配置中的 memory 目录
        """
        if cls._timer_task is not None:
            logger.debug("[DreamDaemon] already started, skipping")
            return

        if config is None:
            from middleware.config import g_config
            config = g_config.load().dream

        if not config.enabled:
            logger.info("[DreamDaemon] disabled by config, skipping")
            return

        loop = DreamLoop(memory_dir=memory_dir, config=config)

        cls._timer_task = asyncio.create_task(loop.run_timer())
        logger.info(
            "[DreamDaemon] started: poll_interval={}s, min_hours={}",
            config.poll_interval_seconds,
            config.min_hours,
        )
