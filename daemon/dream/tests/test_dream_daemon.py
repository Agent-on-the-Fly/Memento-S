# SPDX-License-Identifier: Apache-2.0
"""Tests for daemon/dream/__init__.py — DreamDaemon."""

from __future__ import annotations

import pytest


class TestDreamDaemon:
    """Test DreamDaemon.start() idempotency and config-driven behavior."""

    def test_start_is_idempotent(self, monkeypatch):
        """多次调用 start() 不会创建多个协程。"""
        from daemon.dream import DreamDaemon, DreamConfig

        DreamDaemon._timer_task = None

        # Patch DreamLoop.__init__ to avoid actual instantiation
        monkeypatch.setattr(
            "daemon.dream.loop.DreamLoop.__init__",
            lambda self, **kwargs: None,
        )

        task_count = {"value": 0}

        def counting_create_task(coro):
            task_count["value"] += 1
            return coro

        monkeypatch.setattr("asyncio.create_task", counting_create_task)

        cfg = DreamConfig(enabled=True)

        DreamDaemon.start(config=cfg)
        first = DreamDaemon._timer_task

        DreamDaemon.start(config=cfg)
        second = DreamDaemon._timer_task

        assert task_count["value"] == 1
        assert first is second

    def test_disabled_skips_start(self, monkeypatch):
        """enabled=False 时不启动任何协程。"""
        from daemon.dream import DreamDaemon, DreamConfig

        DreamDaemon._timer_task = None

        loop_called = {"value": False}

        def fake_loop_init(self, **kwargs):
            loop_called["value"] = True

        monkeypatch.setattr("daemon.dream.loop.DreamLoop.__init__", fake_loop_init)

        cfg = DreamConfig(enabled=False)

        DreamDaemon.start(config=cfg)

        assert DreamDaemon._timer_task is None
        assert loop_called["value"] is False
