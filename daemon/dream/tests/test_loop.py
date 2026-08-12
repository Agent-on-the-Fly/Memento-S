# SPDX-License-Identifier: Apache-2.0
"""Tests for daemon/dream/loop.py — DreamLoop."""

from __future__ import annotations

import asyncio
import tempfile
import threading
from pathlib import Path

import pytest


@pytest.fixture
def tmp_memory_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def dream_config():
    from middleware.config.schemas.config_models import DreamConfig
    return DreamConfig(
        enabled=True,
        min_hours=1,
        min_sessions=2,
        poll_interval_seconds=600,
        scan_interval_seconds=600,
    )


class TestSpawnThread:
    """Test _spawn_dream_thread."""

    def test_spawns_daemon_thread(self, tmp_memory_dir, dream_config):
        from daemon.dream.loop import DreamLoop

        loop = DreamLoop(memory_dir=tmp_memory_dir, config=dream_config)

        created_threads = []

        class SpyThread(threading.Thread):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created_threads.append(self)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("daemon.dream.loop.threading.Thread", SpyThread)
            loop._spawn_dream_thread()

        assert len(created_threads) == 1
        assert created_threads[0].daemon is True
        assert created_threads[0].name.startswith("dream-")


class TestRunTimer:
    """Test run_timer coroutine (short interval)."""

    @pytest.mark.asyncio
    async def test_triggers_after_interval(self, tmp_memory_dir, monkeypatch):
        from daemon.dream.loop import DreamLoop

        trigger_count = {"value": 0}
        original_spawn = DreamLoop._spawn_dream_thread

        def counting_spawn(self):
            trigger_count["value"] += 1
            original_spawn(self)

        monkeypatch.setattr(DreamLoop, "_spawn_dream_thread", counting_spawn)

        from middleware.config.schemas.config_models import DreamConfig
        cfg = DreamConfig(
            poll_interval_seconds=0.05,
            min_hours=1,
            min_sessions=2,
            scan_interval_seconds=600.0,
        )
        loop = DreamLoop(memory_dir=tmp_memory_dir, config=cfg)

        async def run_and_cancel():
            task = asyncio.create_task(loop.run_timer())
            await asyncio.sleep(0.22)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        await run_and_cancel()
        assert trigger_count["value"] >= 3
