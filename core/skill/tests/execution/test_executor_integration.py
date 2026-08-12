# SPDX-License-Identifier: Apache-2.0
"""Integration tests for SkillAgent."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.schema import SkillConfig
from core.skill.execution import SkillAgent
from core.skill.schema import Skill


class TestAgentBootstrap:
    """Test agent initialization and run method."""

    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Agent initializes with config and LLM."""
        config = MagicMock(spec=SkillConfig)
        config.workspace_dir = Path("/test/workspace")
        agent = SkillAgent(config=config)

        assert agent._config is config
        assert agent._llm is not None
        assert agent._skill_env_cache is None

    @pytest.mark.asyncio
    async def test_agent_accepts_policy_manager(self):
        """Agent accepts optional policy_manager."""
        config = MagicMock(spec=SkillConfig)
        mock_pm = MagicMock()
        agent = SkillAgent(config=config, policy_manager=mock_pm)
        assert agent._policy_manager is mock_pm


class TestAgentRunDir:
    """Test run directory handling."""

    @pytest.mark.asyncio
    async def test_goal_met_with_created_files(self):
        """Goal is met when files were created."""
        config = MagicMock(spec=SkillConfig)
        config.workspace_dir = Path("/test/workspace")
        agent = SkillAgent(config=config)

        from core.skill.execution.state import ReActState

        state = ReActState(query="test", params={}, max_turns=30)
        state.created_files.append("/test/workspace/output.txt")

        result = agent._goal_met(state, config.workspace_dir)
        assert result is True

    @pytest.mark.asyncio
    async def test_goal_met_with_primary_artifact(self):
        """Goal is met when primary artifact exists."""
        config = MagicMock(spec=SkillConfig)
        workspace = Path("/test/workspace")
        config.workspace_dir = workspace
        agent = SkillAgent(config=config)

        from core.skill.execution.state import ReActState

        state = ReActState(query="test", params={}, max_turns=30)
        state.core_artifacts[".pdf"] = str(workspace / "report.pdf")

        # Create the artifact file
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "report.pdf").touch()

        result = agent._goal_met(state, workspace)
        assert result is True

        # Cleanup
        (workspace / "report.pdf").unlink()


class TestAgentToolSchemas:
    """Test tool schema retrieval."""

    @pytest.mark.asyncio
    async def test_get_tool_schemas_allows_all_when_no_restriction(self):
        """All tools are returned when no allowed_tools restriction."""
        config = MagicMock(spec=SkillConfig)
        agent = SkillAgent(config=config)

        skill = MagicMock(spec=Skill)
        skill.name = "test"
        skill.allowed_tools = None

        # Mock get_tool_schemas
        mock_schemas = [
            {"type": "function", "function": {"name": "read_file", "description": "Read"}},
            {"type": "function", "function": {"name": "bash", "description": "Bash"}},
        ]

        with patch("core.skill.execution.agent.tools.get_tool_schemas", return_value=mock_schemas):
            schemas = agent._get_tool_schemas(skill)
            assert len(schemas) == 2

    @pytest.mark.asyncio
    async def test_get_tool_schemas_filters_by_allowed_list(self):
        """Tools are filtered by allowed_tools list."""
        config = MagicMock(spec=SkillConfig)
        agent = SkillAgent(config=config)

        skill = MagicMock(spec=Skill)
        skill.name = "test"
        skill.allowed_tools = ["read_file"]

        mock_schemas = [
            {"type": "function", "function": {"name": "read_file", "description": "Read"}},
            {"type": "function", "function": {"name": "bash", "description": "Bash"}},
        ]

        with patch("core.skill.execution.agent.tools.get_tool_schemas", return_value=mock_schemas):
            schemas = agent._get_tool_schemas(skill)
            assert len(schemas) == 1
            assert schemas[0]["function"]["name"] == "read_file"


class TestAgentEnvVars:
    """Test environment variable building."""

    @pytest.mark.asyncio
    async def test_build_env_vars_includes_workspace_root(self):
        """ENV VAR JAIL includes WORKSPACE_ROOT."""
        config = MagicMock(spec=SkillConfig)
        agent = SkillAgent(config=config)

        from core.skill.execution.state import ReActState

        state = ReActState(query="test", params={}, max_turns=30)
        workspace = Path("/test/workspace")

        env_vars = agent._build_env_vars(workspace, state)
        assert "WORKSPACE_ROOT" in env_vars
        assert env_vars["WORKSPACE_ROOT"] == str(workspace)

    @pytest.mark.asyncio
    async def test_build_env_vars_includes_primary_artifact(self):
        """ENV VAR JAIL includes PRIMARY_ARTIFACT_PATH when set."""
        config = MagicMock(spec=SkillConfig)
        agent = SkillAgent(config=config)

        from core.skill.execution.state import ReActState

        state = ReActState(query="test", params={}, max_turns=30)
        state.core_artifacts[".pdf"] = "/test/workspace/report.pdf"
        workspace = Path("/test/workspace")

        env_vars = agent._build_env_vars(workspace, state)
        assert "PRIMARY_ARTIFACT_PATH" in env_vars
        assert env_vars["PRIMARY_ARTIFACT_PATH"] == "/test/workspace/report.pdf"
