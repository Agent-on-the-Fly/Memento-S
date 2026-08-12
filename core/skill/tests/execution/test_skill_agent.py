# SPDX-License-Identifier: Apache-2.0
"""Tests for SkillAgent — core/skill/execution/agent.py"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.schema import SkillConfig
from core.skill.execution.agent import SkillAgent
from core.skill.schema import Skill


class TestSkillAgentInit:
    """Test SkillAgent initialization."""

    def test_init_with_default_llm(self):
        """SkillAgent initializes with default LLM client."""
        config = MagicMock(spec=SkillConfig)
        agent = SkillAgent(config=config)
        assert agent._config is config
        assert agent._llm is not None

    def test_init_with_custom_llm(self):
        """SkillAgent initializes with custom LLM client."""
        config = MagicMock(spec=SkillConfig)
        custom_llm = MagicMock()
        agent = SkillAgent(config=config, llm=custom_llm)
        assert agent._llm is custom_llm


class TestSkillAgentToolSchemas:
    """Test _get_tool_schemas method."""

    def test_returns_empty_when_tools_not_init(self):
        """Returns empty list when tools registry is not initialized."""
        config = MagicMock(spec=SkillConfig)
        agent = SkillAgent(config=config)

        skill = MagicMock(spec=Skill)
        skill.allowed_tools = None

        with patch("core.skill.execution.agent.tools") as mock_tools:
            mock_tools.get_tool_schemas.side_effect = Exception("not initialized")
            schemas = agent._get_tool_schemas(skill)
            assert schemas == []

    def test_returns_all_when_no_allowed_list(self):
        """Returns all schemas when skill has no allowed_tools restriction."""
        config = MagicMock(spec=SkillConfig)
        agent = SkillAgent(config=config)

        skill = MagicMock(spec=Skill)
        skill.allowed_tools = None

        expected = [{"type": "function", "function": {"name": "read_file"}}]
        with patch("core.skill.execution.agent.tools") as mock_tools:
            mock_tools.get_tool_schemas.return_value = expected
            schemas = agent._get_tool_schemas(skill)
            assert schemas == expected

    def test_filters_by_allowed_list(self):
        """Filters schemas by skill.allowed_tools when set."""
        config = MagicMock(spec=SkillConfig)
        agent = SkillAgent(config=config)

        skill = MagicMock(spec=Skill)
        skill.allowed_tools = ["read_file", "bash"]

        all_schemas = [
            {"type": "function", "function": {"name": "read_file"}},
            {"type": "function", "function": {"name": "bash"}},
            {"type": "function", "function": {"name": "python_repl"}},
        ]

        with patch("core.skill.execution.agent.tools") as mock_tools:
            mock_tools.get_tool_schemas.return_value = all_schemas
            schemas = agent._get_tool_schemas(skill)
            assert len(schemas) == 2
            names = {s["function"]["name"] for s in schemas}
            assert names == {"read_file", "bash"}


class TestSkillAgentEnvVars:
    """Test _build_env_vars method."""

    def test_includes_workspace_root(self):
        """Environment includes WORKSPACE_ROOT."""
        config = MagicMock(spec=SkillConfig)
        config.primary_artifact_path = None
        agent = SkillAgent(config=config)

        workspace = Path("/test/workspace")
        env = agent._build_env_vars(workspace)
        assert "WORKSPACE_ROOT" in env
        assert env["WORKSPACE_ROOT"] == str(workspace)

    def test_includes_primary_artifact_path(self):
        """Environment includes PRIMARY_ARTIFACT_PATH when set."""
        config = MagicMock(spec=SkillConfig)
        config.primary_artifact_path = Path("/test/artifact.xlsx")
        agent = SkillAgent(config=config)

        workspace = Path("/test/workspace")
        env = agent._build_env_vars(workspace)
        assert "PRIMARY_ARTIFACT_PATH" in env
        assert env["PRIMARY_ARTIFACT_PATH"] == str(Path("/test/artifact.xlsx"))


class TestSkillAgentPromptBuilding:
    """Test _build_messages method."""

    def test_system_prompt_includes_skill_name(self):
        """System prompt includes skill name."""
        from core.skill.execution.state import ReActState

        config = MagicMock(spec=SkillConfig)
        agent = SkillAgent(config=config)

        skill = MagicMock(spec=Skill)
        skill.name = "test_skill"
        skill.description = "A test skill"
        skill.source_dir = None
        skill.content = "# Test Skill"

        state = ReActState(query="test query", params=None)
        workspace = Path("/test/workspace")

        with patch.object(agent, "_list_existing_scripts", return_value="- script.py"):
            with patch.object(agent, "_get_skill_content", return_value="# Test"):
                with patch.object(agent, "_get_real_file_tree_limited", return_value="- file.txt"):
                    messages = agent._build_messages(skill, state, workspace)

        assert len(messages) >= 1
        system_content = messages[0]["content"]
        assert "test_skill" in system_content
        assert "test query" in system_content


class TestSkillAgentRun:
    """Test end-to-end run method."""

    @pytest.mark.asyncio
    async def test_calls_llm_with_messages(self):
        """Agent calls LLM with properly built messages."""
        config = MagicMock(spec=SkillConfig)
        config.primary_artifact_path = None

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.has_tool_calls = False
        mock_response.text = "Final Answer: Task completed."
        mock_response.tool_calls = []
        mock_llm.async_chat = AsyncMock(return_value=mock_response)

        agent = SkillAgent(config=config, llm=mock_llm)

        skill = MagicMock(spec=Skill)
        skill.name = "test"
        skill.allowed_tools = None
        skill.source_dir = None
        skill.content = ""

        with patch("core.skill.execution.agent.tools") as mock_tools:
            mock_tools.get_tool_schemas.return_value = []
            outcome, code = await agent.run(
                skill=skill,
                query="test query",
                params=None,
                run_dir=Path("/tmp"),
                session_id="test_session",
                on_step=None,
            )

        mock_llm.async_chat.assert_called_once()
        call_kwargs = mock_llm.async_chat.call_args
        assert "messages" in call_kwargs.kwargs
        assert "tools" in call_kwargs.kwargs

    @pytest.mark.asyncio
    async def test_handles_tool_calls_via_adapter(self):
        """Agent handles tool calls through SkillToolAdapter."""
        config = MagicMock(spec=SkillConfig)
        config.primary_artifact_path = None

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.has_tool_calls = True
        mock_response.text = ""
        mock_response.tool_calls = [
            {"id": "call_1", "function": {"name": "read_file", "arguments": '{"path": "/test.txt"}'}}
        ]
        mock_llm.async_chat = AsyncMock(return_value=mock_response)

        agent = SkillAgent(config=config, llm=mock_llm)

        skill = MagicMock(spec=Skill)
        skill.name = "test"
        skill.allowed_tools = None
        skill.source_dir = None
        skill.content = ""

        mock_adapter = AsyncMock()
        mock_adapter.execute = AsyncMock(return_value=("file content", None))

        with patch("core.skill.execution.agent.tools") as mock_tools:
            mock_tools.get_tool_schemas.return_value = []
            with patch.object(agent, "_build_messages", return_value=[{"role": "system", "content": ""}]):
                with patch("core.skill.execution.agent.SkillToolAdapter") as MockAdapter:
                    MockAdapter.return_value = mock_adapter
                    outcome, code = await agent.run(
                        skill=skill,
                        query="test query",
                        params=None,
                        run_dir=Path("/tmp"),
                        session_id="test_session",
                        on_step=None,
                    )
