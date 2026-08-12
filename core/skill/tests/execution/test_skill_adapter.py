# SPDX-License-Identifier: Apache-2.0
"""Tests for SkillToolAdapter — core/skill/execution/adapter.py"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.schema import SkillConfig
from shared.hooks import HookExecutor
from shared.hooks.types import HookEvent, HookPayload, HookResult
from core.skill.execution.adapter import SkillToolAdapter
from core.skill.schema import ErrorType, Skill


class TestSkillToolAdapterInit:
    """Test SkillToolAdapter initialization."""

    def test_init_with_config(self):
        """Adapter initializes with config only."""
        config = MagicMock(spec=SkillConfig)
        adapter = SkillToolAdapter(config=config)
        assert adapter._config is config
        assert adapter._result_cache is None

    def test_init_with_result_cache(self):
        """Adapter initializes with result cache."""
        config = MagicMock(spec=SkillConfig)
        mock_cache = MagicMock()
        adapter = SkillToolAdapter(config=config, result_cache=mock_cache)
        assert adapter._result_cache is mock_cache


class TestSkillToolAdapterContext:
    """Test set_context method."""

    def test_set_context_creates_tool_context(self):
        """set_context creates ToolContext from skill and workspace."""
        config = MagicMock(spec=SkillConfig)
        adapter = SkillToolAdapter(config=config)

        skill = MagicMock(spec=Skill)
        skill.name = "test_skill"
        workspace = Path("/test/workspace")

        with patch("core.skill.execution.adapter.RuntimeToolContext") as MockContext:
            adapter.set_context(skill, workspace)
            MockContext.from_skill.assert_called_once()
            call_kwargs = MockContext.from_skill.call_args
            assert call_kwargs.kwargs["config"] is config
            assert call_kwargs.kwargs["skill"] is skill
            assert call_kwargs.kwargs["workspace_dir"] == workspace


class TestSkillToolAdapterExecute:
    """Test execute method."""

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        """Unknown tool returns error tuple."""
        config = MagicMock(spec=SkillConfig)
        adapter = SkillToolAdapter(config=config)

        mock_registry = MagicMock()
        mock_registry.is_registered.return_value = False

        with patch("core.skill.execution.adapter.get_registry", return_value=mock_registry):
            obs, err = await adapter.execute(
                tool_name="unknown_tool",
                raw_args={},
                tool_props=None,
                env_vars=None,
            )

        assert "unknown_tool" in obs
        assert "not found" in obs
        assert err is not None
        assert err["error_type"] == ErrorType.TOOL_NOT_FOUND

    @pytest.mark.asyncio
    async def test_registered_tool_calls_registry_execute(self):
        """Registered tool calls registry.execute()."""
        config = MagicMock(spec=SkillConfig)
        adapter = SkillToolAdapter(config=config)

        mock_registry = MagicMock()
        mock_registry.is_registered.return_value = True
        mock_registry.execute = AsyncMock(return_value="tool result")

        mock_args_processor = MagicMock()
        mock_args_processor.process.return_value = ({}, [])
        adapter._args_processor = mock_args_processor

        mock_result_processor = AsyncMock()
        mock_result_processor.process.return_value = (
            MagicMock(warning=None),
            MagicMock(decision_basis={}),
        )
        adapter._result_processor = mock_result_processor

        with patch("core.skill.execution.adapter.get_registry", return_value=mock_registry):
            obs, err = await adapter.execute(
                tool_name="read_file",
                raw_args={"path": "/test.txt"},
                tool_props=None,
                env_vars=None,
            )

        mock_registry.execute.assert_called_once_with("read_file", {})
        assert err is None

    @pytest.mark.asyncio
    async def test_policy_blocked_returns_error(self):
        """Policy blocked tool returns error tuple via hook_executor."""
        config = MagicMock(spec=SkillConfig)
        adapter = SkillToolAdapter(config=config)

        mock_registry = MagicMock()
        mock_registry.is_registered.return_value = True

        mock_args_processor = MagicMock()
        mock_args_processor.process.return_value = ({"command": "rm -rf /"}, [])
        adapter._args_processor = mock_args_processor

        mock_hook_executor = AsyncMock()
        mock_hook_executor.execute.return_value = HookResult(
            allowed=False, reason="forbidden"
        )
        adapter._hook_executor = mock_hook_executor

        mock_result_processor = AsyncMock()
        mock_result_processor.process.return_value = (
            MagicMock(warning=None),
            MagicMock(decision_basis={}),
        )
        adapter._result_processor = mock_result_processor

        with patch("core.skill.execution.adapter.get_registry", return_value=mock_registry):
            obs, err = await adapter.execute(
                tool_name="bash",
                raw_args={"command": "rm -rf /"},
                tool_props=None,
                env_vars=None,
            )

        assert "blocked by hook" in obs
        assert err is not None
        assert err["error_type"] == ErrorType.POLICY_BLOCKED


class TestSkillToolAdapterResultCache:
    """Test result caching behavior."""

    @pytest.mark.asyncio
    async def test_json_result_registered_to_cache(self):
        """JSON result is parsed and registered to cache."""
        config = MagicMock(spec=SkillConfig)
        mock_cache = MagicMock()
        adapter = SkillToolAdapter(config=config, result_cache=mock_cache)

        mock_registry = MagicMock()
        mock_registry.is_registered.return_value = True
        mock_registry.execute = AsyncMock(
            return_value=json.dumps({"uuid": "123", "result": "data"})
        )

        mock_args_processor = MagicMock()
        mock_args_processor.process.return_value = ({}, [])
        adapter._args_processor = mock_args_processor

        mock_result_processor = AsyncMock()
        mock_result_processor.process.return_value = (
            MagicMock(warning=None),
            MagicMock(decision_basis={"state": "succeeded"}),
        )
        adapter._result_processor = mock_result_processor

        with patch("core.skill.execution.adapter.get_registry", return_value=mock_registry):
            await adapter.execute(
                tool_name="search",
                raw_args={"query": "test"},
                tool_props=None,
                env_vars=None,
            )

        mock_cache.register.assert_called()
        registered_calls = [str(c) for c in mock_cache.register.call_args_list]
        assert any("search.uuid" in c for c in registered_calls)
        assert any("search.result" in c for c in registered_calls)

    @pytest.mark.asyncio
    async def test_list_result_registered_to_cache(self):
        """List result is parsed and registered to cache."""
        config = MagicMock(spec=SkillConfig)
        mock_cache = MagicMock()
        adapter = SkillToolAdapter(config=config, result_cache=mock_cache)

        mock_registry = MagicMock()
        mock_registry.is_registered.return_value = True
        mock_registry.execute = AsyncMock(
            return_value=json.dumps([
                {"id": "1", "name": "item1"},
                {"id": "2", "name": "item2"},
            ])
        )

        mock_args_processor = MagicMock()
        mock_args_processor.process.return_value = ({}, [])
        adapter._args_processor = mock_args_processor

        mock_result_processor = AsyncMock()
        mock_result_processor.process.return_value = (
            MagicMock(warning=None),
            MagicMock(decision_basis={"state": "succeeded"}),
        )
        adapter._result_processor = mock_result_processor

        with patch("core.skill.execution.adapter.get_registry", return_value=mock_registry):
            await adapter.execute(
                tool_name="list_items",
                raw_args={},
                tool_props=None,
                env_vars=None,
            )

        mock_cache.register.assert_called()

    @pytest.mark.asyncio
    async def test_invalid_json_not_registered(self):
        """Non-JSON result is not registered to cache."""
        config = MagicMock(spec=SkillConfig)
        mock_cache = MagicMock()
        adapter = SkillToolAdapter(config=config, result_cache=mock_cache)

        mock_registry = MagicMock()
        mock_registry.is_registered.return_value = True
        mock_registry.execute = AsyncMock(return_value="plain text result")

        mock_args_processor = MagicMock()
        mock_args_processor.process.return_value = ({}, [])
        adapter._args_processor = mock_args_processor

        mock_result_processor = AsyncMock()
        mock_result_processor.process.return_value = (
            MagicMock(warning=None),
            MagicMock(decision_basis={"state": "succeeded"}),
        )
        adapter._result_processor = mock_result_processor

        with patch("core.skill.execution.adapter.get_registry", return_value=mock_registry):
            obs, err = await adapter.execute(
                tool_name="bash",
                raw_args={"command": "echo hello"},
                tool_props=None,
                env_vars=None,
            )

        # Cache should not be called for non-JSON results
        mock_cache.register.assert_not_called()
