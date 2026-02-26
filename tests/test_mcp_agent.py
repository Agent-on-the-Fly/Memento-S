"""Tests for the MCPAgent class."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.mcp_server import configure


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    configure(base_dir=tmp_path)
    return tmp_path


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestMCPAgent:
    def test_import(self) -> None:
        from core.mcp_agent import MCPAgent
        assert MCPAgent is not None

    def test_start_loads_tools(self, tmp_workspace: Path, event_loop) -> None:
        """Agent start() discovers MCP tools and builds agent graph."""
        async def _test():
            from core.mcp_agent import MCPAgent

            mock_model = MagicMock()
            agent = MCPAgent(model=mock_model, base_dir=tmp_workspace)

            # Start initializes MCPToolManager and discovers tools
            await agent.start()
            try:
                tools = agent.tool_manager.get_langchain_tools()
                assert len(tools) == 7
                names = {t.name for t in tools}
                assert "bash_tool" in names
                assert "str_replace" in names
                assert "file_create" in names
                assert "view" in names

                # Agent graph should be built
                assert agent._agent_graph is not None
            finally:
                await agent.close()

        event_loop.run_until_complete(_test())

    def test_run_raises_if_not_started(self, event_loop) -> None:
        async def _test():
            from core.mcp_agent import MCPAgent

            mock_model = MagicMock()
            agent = MCPAgent(model=mock_model)
            with pytest.raises(RuntimeError, match="not started"):
                await agent.run("hello")

        event_loop.run_until_complete(_test())

    def test_stream_raises_if_not_started(self, event_loop) -> None:
        async def _test():
            from core.mcp_agent import MCPAgent

            mock_model = MagicMock()
            agent = MCPAgent(model=mock_model)
            with pytest.raises(RuntimeError, match="not started"):
                async for _ in agent.stream("hello"):
                    pass

        event_loop.run_until_complete(_test())

    def test_close_cleans_up(self, tmp_workspace: Path, event_loop) -> None:
        async def _test():
            from core.mcp_agent import MCPAgent

            mock_model = MagicMock()
            agent = MCPAgent(model=mock_model, base_dir=tmp_workspace)
            await agent.start()
            assert agent._agent_graph is not None

            await agent.close()
            assert agent._agent_graph is None

        event_loop.run_until_complete(_test())

    def test_custom_system_prompt(self) -> None:
        from core.mcp_agent import MCPAgent

        mock_model = MagicMock()
        agent = MCPAgent(model=mock_model, system_prompt="Custom prompt")
        assert agent._system_prompt == "Custom prompt"

    def test_default_system_prompt(self) -> None:
        from core.mcp_agent import MCPAgent

        mock_model = MagicMock()
        agent = MCPAgent(model=mock_model)
        assert "Memento-S" in agent._system_prompt

    def test_tool_names_property(self, tmp_workspace: Path, event_loop) -> None:
        """tool_names property returns list of tool name strings."""
        async def _test():
            from core.mcp_agent import MCPAgent

            mock_model = MagicMock()
            agent = MCPAgent(model=mock_model, base_dir=tmp_workspace)
            await agent.start()
            try:
                names = agent.tool_names
                assert isinstance(names, list)
                assert len(names) == 7
                assert "bash_tool" in names
            finally:
                await agent.close()

        event_loop.run_until_complete(_test())

    def test_to_lc_messages(self) -> None:
        """_to_lc_messages converts plain dicts to LangChain messages."""
        from core.mcp_agent import _to_lc_messages
        from langchain_core.messages import AIMessage, HumanMessage

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]
        lc_msgs = _to_lc_messages(messages)
        assert len(lc_msgs) == 3
        assert isinstance(lc_msgs[0], HumanMessage)
        assert isinstance(lc_msgs[1], AIMessage)
        assert isinstance(lc_msgs[2], HumanMessage)
        assert lc_msgs[0].content == "Hello"
        assert lc_msgs[1].content == "Hi there"

    def test_model_factory_import(self) -> None:
        """build_chat_model is importable from core.model_factory."""
        from core.model_factory import build_chat_model
        assert callable(build_chat_model)
