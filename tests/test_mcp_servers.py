"""Tests for the unified FastMCP server (7 tools) and MCP client."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from core.mcp_server import mcp, configure


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace for tests."""
    configure(base_dir=tmp_path)
    return tmp_path


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# -------------------------------------------------------------------
# bash_tool
# -------------------------------------------------------------------

class TestBashTool:
    def test_echo(self, tmp_workspace: Path) -> None:
        from core.mcp_server import bash_tool

        result = bash_tool(command="echo hello", description="test echo")
        assert "hello" in result

    def test_empty_command(self, tmp_workspace: Path) -> None:
        from core.mcp_server import bash_tool

        result = bash_tool(command="", description="empty")
        assert "ERR" in result

    def test_failure(self, tmp_workspace: Path) -> None:
        from core.mcp_server import bash_tool

        result = bash_tool(command="false", description="fail")
        assert "ERR" in result

    def test_working_dir(self, tmp_workspace: Path) -> None:
        from core.mcp_server import bash_tool

        result = bash_tool(command="pwd", description="check cwd")
        assert str(tmp_workspace) in result


# -------------------------------------------------------------------
# str_replace
# -------------------------------------------------------------------

class TestStrReplace:
    def test_basic_replace(self, tmp_workspace: Path) -> None:
        from core.mcp_server import str_replace

        f = tmp_workspace / "test.txt"
        f.write_text("foo bar baz", encoding="utf-8")
        result = str_replace(description="replace bar", path=str(f), old_str="bar", new_str="qux")
        assert "OK" in result
        assert f.read_text() == "foo qux baz"

    def test_not_found(self, tmp_workspace: Path) -> None:
        from core.mcp_server import str_replace

        f = tmp_workspace / "test.txt"
        f.write_text("foo bar", encoding="utf-8")
        result = str_replace(description="missing", path=str(f), old_str="NOPE", new_str="x")
        assert "ERR" in result
        assert "not found" in result

    def test_not_unique(self, tmp_workspace: Path) -> None:
        from core.mcp_server import str_replace

        f = tmp_workspace / "test.txt"
        f.write_text("foo foo", encoding="utf-8")
        result = str_replace(description="not unique", path=str(f), old_str="foo", new_str="bar")
        assert "ERR" in result
        assert "2 times" in result

    def test_delete_string(self, tmp_workspace: Path) -> None:
        from core.mcp_server import str_replace

        f = tmp_workspace / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        result = str_replace(description="delete", path=str(f), old_str=" world")
        assert "OK" in result
        assert f.read_text() == "hello"

    def test_file_not_exists(self, tmp_workspace: Path) -> None:
        from core.mcp_server import str_replace

        result = str_replace(description="missing file", path=str(tmp_workspace / "nope.txt"), old_str="a", new_str="b")
        assert "ERR" in result
        assert "not found" in result

    def test_relative_path(self, tmp_workspace: Path) -> None:
        from core.mcp_server import str_replace

        f = tmp_workspace / "rel.txt"
        f.write_text("alpha beta", encoding="utf-8")
        result = str_replace(description="relative", path="rel.txt", old_str="alpha", new_str="gamma")
        assert "OK" in result
        assert f.read_text() == "gamma beta"


# -------------------------------------------------------------------
# file_create
# -------------------------------------------------------------------

class TestFileCreate:
    def test_create_file(self, tmp_workspace: Path) -> None:
        from core.mcp_server import file_create

        result = file_create(description="create", path=str(tmp_workspace / "new.txt"), file_text="hello")
        assert "OK" in result
        assert (tmp_workspace / "new.txt").read_text() == "hello"

    def test_create_nested(self, tmp_workspace: Path) -> None:
        from core.mcp_server import file_create

        result = file_create(
            description="nested create",
            path=str(tmp_workspace / "a" / "b" / "c.txt"),
            file_text="nested",
        )
        assert "OK" in result
        assert (tmp_workspace / "a" / "b" / "c.txt").read_text() == "nested"

    def test_overwrite(self, tmp_workspace: Path) -> None:
        from core.mcp_server import file_create

        f = tmp_workspace / "over.txt"
        f.write_text("old", encoding="utf-8")
        file_create(description="overwrite", path=str(f), file_text="new")
        assert f.read_text() == "new"

    def test_relative_path(self, tmp_workspace: Path) -> None:
        from core.mcp_server import file_create

        result = file_create(description="relative", path="relfile.txt", file_text="content")
        assert "OK" in result
        assert (tmp_workspace / "relfile.txt").read_text() == "content"


# -------------------------------------------------------------------
# view
# -------------------------------------------------------------------

class TestView:
    def test_view_text_file(self, tmp_workspace: Path) -> None:
        from core.mcp_server import view

        f = tmp_workspace / "hello.txt"
        f.write_text("line1\nline2\nline3", encoding="utf-8")
        result = view(description="view text", path=str(f))
        assert "line1" in result
        assert "line2" in result
        assert "line3" in result
        # Check line numbers present
        assert "1\t" in result

    def test_view_range(self, tmp_workspace: Path) -> None:
        from core.mcp_server import view

        f = tmp_workspace / "lines.txt"
        f.write_text("a\nb\nc\nd\ne", encoding="utf-8")
        result = view(description="view range", path=str(f), view_range=[2, 4])
        assert "b" in result
        assert "c" in result
        assert "d" in result
        lines = result.strip().splitlines()
        assert len(lines) == 3

    def test_view_range_to_end(self, tmp_workspace: Path) -> None:
        from core.mcp_server import view

        f = tmp_workspace / "lines.txt"
        f.write_text("a\nb\nc\nd\ne", encoding="utf-8")
        result = view(description="view to end", path=str(f), view_range=[3, -1])
        lines = result.strip().splitlines()
        assert len(lines) == 3  # c, d, e

    def test_view_directory(self, tmp_workspace: Path) -> None:
        from core.mcp_server import view

        (tmp_workspace / "sub").mkdir()
        (tmp_workspace / "file.txt").write_text("x", encoding="utf-8")
        result = view(description="view dir", path=str(tmp_workspace))
        assert "sub/" in result
        assert "file.txt" in result

    def test_view_directory_ignores_hidden(self, tmp_workspace: Path) -> None:
        from core.mcp_server import view

        (tmp_workspace / ".hidden").mkdir()
        (tmp_workspace / "visible.txt").write_text("x", encoding="utf-8")
        result = view(description="view dir hidden", path=str(tmp_workspace))
        assert ".hidden" not in result
        assert "visible.txt" in result

    def test_view_not_found(self, tmp_workspace: Path) -> None:
        from core.mcp_server import view

        result = view(description="missing", path=str(tmp_workspace / "nope.txt"))
        assert "ERR" in result
        assert "not found" in result

    def test_view_image(self, tmp_workspace: Path) -> None:
        from core.mcp_server import view

        f = tmp_workspace / "img.png"
        f.write_bytes(b"\x89PNG\r\n")
        result = view(description="view image", path=str(f))
        assert "Image file" in result
        assert "bytes" in result


# -------------------------------------------------------------------
# MCP Client (integration)
# -------------------------------------------------------------------

class TestMCPClient:
    def test_start_and_list_tools(self, tmp_workspace: Path, event_loop) -> None:
        async def _test():
            from core.mcp_client import MCPToolManager

            mgr = MCPToolManager()
            await mgr.start(base_dir=tmp_workspace)
            try:
                tools = mgr.get_tool_definitions()
                assert len(tools) == 7
                tool_names = {t["function"]["name"] for t in tools}
                assert {"bash_tool", "str_replace", "file_create", "view"}.issubset(tool_names)
                assert {"search_cloud_skills", "read_skill", "list_local_skills"}.issubset(tool_names)
            finally:
                await mgr.shutdown()

        event_loop.run_until_complete(_test())

    def test_call_tool(self, tmp_workspace: Path, event_loop) -> None:
        async def _test():
            from core.mcp_client import MCPToolManager

            mgr = MCPToolManager()
            await mgr.start(base_dir=tmp_workspace)
            try:
                result = await mgr.call_tool(
                    "file_create",
                    {"description": "create", "path": str(tmp_workspace / "test.txt"), "file_text": "test content"},
                )
                assert "OK" in result

                result = await mgr.call_tool(
                    "view",
                    {"description": "view", "path": str(tmp_workspace / "test.txt")},
                )
                assert "test content" in result
            finally:
                await mgr.shutdown()

        event_loop.run_until_complete(_test())

    def test_langchain_tools(self, tmp_workspace: Path, event_loop) -> None:
        async def _test():
            from core.mcp_client import MCPToolManager

            mgr = MCPToolManager()
            await mgr.start(base_dir=tmp_workspace)
            try:
                lc_tools = mgr.get_langchain_tools()
                assert len(lc_tools) == 7
                names = {t.name for t in lc_tools}
                assert {"bash_tool", "str_replace", "file_create", "view"}.issubset(names)

                # Each tool should have a description
                for t in lc_tools:
                    assert t.description

                # Invoke a tool via LangChain interface
                bash = next(t for t in lc_tools if t.name == "bash_tool")
                result = await bash.ainvoke({"command": "echo langchain", "description": "test"})
                assert "langchain" in result
            finally:
                await mgr.shutdown()

        event_loop.run_until_complete(_test())

    def test_tool_definitions_format(self, tmp_workspace: Path, event_loop) -> None:
        async def _test():
            from core.mcp_client import MCPToolManager

            mgr = MCPToolManager()
            await mgr.start(base_dir=tmp_workspace)
            try:
                tools = mgr.get_tool_definitions()
                for tool in tools:
                    assert tool["type"] == "function"
                    assert "name" in tool["function"]
                    assert "description" in tool["function"]
                    assert "parameters" in tool["function"]
                    assert tool["function"]["parameters"].get("type") == "object"
            finally:
                await mgr.shutdown()

        event_loop.run_until_complete(_test())
