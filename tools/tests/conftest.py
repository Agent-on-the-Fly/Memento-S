# SPDX-License-Identifier: Apache-2.0
"""Shared test fixtures for tools module tests."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from tools import ToolRegistry, init_registry


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests (session-scoped)."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def fresh_registry() -> ToolRegistry:
    """Provide a fresh ToolRegistry for each test.

    autouse=True ensures complete test isolation — no shared state.
    """
    reg = init_registry()
    yield reg
    reg.clear()
    reg.reset_stats()


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """A temporary workspace directory for tool integration tests."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def sample_tool():
    """A minimal async tool for testing registry operations."""

    async def dummy_tool(arg: str = "default") -> str:
        return f"dummy: {arg}"

    dummy_tool._schema = {
        "type": "object",
        "properties": {
            "arg": {
                "type": "string",
                "description": "An argument for the dummy tool.",
            },
        },
        "required": [],
    }

    return dummy_tool


@pytest.fixture
def failing_tool():
    """An async tool that always raises an exception."""

    async def always_fail(reason: str = "intentional failure") -> str:
        raise ValueError(reason)

    always_fail._schema = {
        "type": "object",
        "properties": {
            "reason": {"type": "string", "description": "Why it fails."},
        },
        "required": [],
    }

    return always_fail
