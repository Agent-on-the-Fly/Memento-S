# SPDX-License-Identifier: Apache-2.0
"""Tests for atomic tools (tools/atomics/)."""

from __future__ import annotations

import asyncio
import pytest

from tools import init_registry, load_atomics, get_registry


class TestAtomicToolsRegistration:
    """Test that all atomic tools are registered and have correct schemas."""

    @pytest.fixture(autouse=True)
    def setup(self, fresh_registry):
        init_registry()
        load_atomics()

    def test_list_dir_registered(self):
        reg = get_registry()
        assert reg.is_registered("list_dir")
        schema = reg.get_schema("list_dir")
        assert schema["function"]["name"] == "list_dir"
        assert "path" in schema["function"]["parameters"]["properties"]

    def test_read_file_registered(self):
        assert get_registry().is_registered("read_file")

    def test_file_create_registered(self):
        assert get_registry().is_registered("file_create")

    def test_edit_file_by_lines_registered(self):
        assert get_registry().is_registered("edit_file_by_lines")

    def test_grep_registered(self):
        assert get_registry().is_registered("grep")

    def test_bash_registered(self):
        assert get_registry().is_registered("bash")

    def test_python_repl_registered(self):
        assert get_registry().is_registered("python_repl")

    def test_js_repl_registered(self):
        assert get_registry().is_registered("js_repl")

    def test_search_web_registered(self):
        assert get_registry().is_registered("search_web")

    def test_fetch_webpage_registered(self):
        assert get_registry().is_registered("fetch_webpage")

    def test_glob_registered(self):
        assert get_registry().is_registered("glob")

    def test_mcp_list_resources_registered(self):
        assert get_registry().is_registered("mcp_list_resources")

    def test_mcp_read_resource_registered(self):
        assert get_registry().is_registered("mcp_read_resource")

    def test_all_13_tools_present(self):
        reg = get_registry()
        atomics = reg.list_by_category("atomic")
        names = {t.name for t in atomics}
        expected = {
            "list_dir", "read_file", "file_create", "edit_file_by_lines",
            "grep", "bash", "python_repl", "js_repl",
            "search_web", "fetch_webpage",
            "glob", "mcp_list_resources", "mcp_read_resource",
        }
        assert expected.issubset(names), f"Missing: {expected - names}"


class TestListDirTool:
    """Test list_dir tool."""

    @pytest.mark.asyncio
    async def test_list_dir_basic(self, fresh_registry, tmp_workspace):
        from tools.atomics import list_dir
        init_registry()
        result = await list_dir(path=str(tmp_workspace), max_depth=1)
        assert "Directory Tree" in result

    @pytest.mark.asyncio
    async def test_list_dir_nonexistent(self, fresh_registry):
        from tools.atomics import list_dir
        result = await list_dir(path="/nonexistent/path/123")
        assert "ERR" in result


class TestReadFileTool:
    """Test read_file tool."""

    @pytest.mark.asyncio
    async def test_read_file_basic(self, fresh_registry, tmp_workspace):
        from tools.atomics import read_file
        f = tmp_workspace / "hello.txt"
        f.write_text("hello world\nline 2\nline 3")
        result = await read_file(path=str(f))
        assert "hello world" in result
        assert "line 2" in result

    @pytest.mark.asyncio
    async def test_read_file_with_line_range(self, fresh_registry, tmp_workspace):
        from tools.atomics import read_file
        f = tmp_workspace / "hello.txt"
        f.write_text("line 1\nline 2\nline 3\nline 4\nline 5")
        result = await read_file(path=str(f), start_line=2, end_line=3)
        assert "line 2" in result
        assert "line 3" in result
        assert "line 1" not in result

    @pytest.mark.asyncio
    async def test_read_file_nonexistent(self, fresh_registry):
        from tools.atomics import read_file
        result = await read_file(path="/nonexistent/file.txt")
        assert "ERR" in result


class TestFileCreateTool:
    """Test file_create tool."""

    @pytest.mark.asyncio
    async def test_file_create_new(self, fresh_registry, tmp_workspace):
        from tools.atomics import file_create
        path = str(tmp_workspace / "created.txt")
        result = await file_create(path=path, content="hello")
        assert "SUCCESS" in result
        assert (tmp_workspace / "created.txt").read_text() == "hello"

    @pytest.mark.asyncio
    async def test_file_create_no_overwrite(self, fresh_registry, tmp_workspace):
        from tools.atomics import file_create
        f = tmp_workspace / "existing.txt"
        f.write_text("original")
        result = await file_create(path=str(f), content="new", overwrite=False)
        assert "ERR" in result or "already exists" in result.lower()
        assert f.read_text() == "original"


class TestEditFileByLinesTool:
    """Test edit_file_by_lines tool."""

    @pytest.mark.asyncio
    async def test_edit_replace_lines(self, fresh_registry, tmp_workspace):
        from tools.atomics import edit_file_by_lines
        f = tmp_workspace / "editme.txt"
        f.write_text("line 1\nline 2\nline 3\n")
        result = await edit_file_by_lines(
            path=str(f),
            start_line=2,
            end_line=2,
            new_content="REPLACED\n",
        )
        assert "SUCCESS" in result
        content = f.read_text()
        assert "REPLACED" in content
        assert "line 2" not in content.split("\n")[1]  # second line replaced

    @pytest.mark.asyncio
    async def test_edit_nonexistent_file(self, fresh_registry):
        from tools.atomics import edit_file_by_lines
        result = await edit_file_by_lines(
            path="/nonexistent/file.txt",
            start_line=1,
            end_line=1,
            new_content="x",
        )
        assert "ERR" in result


class TestGrepTool:
    """Test grep tool."""

    @pytest.mark.asyncio
    async def test_grep_in_text(self, fresh_registry):
        from tools.atomics import grep
        text = "def hello():\n    print('world')\ndef world():\n    pass"
        result = await grep(pattern="def hello", text=text)
        assert "def hello" in result

    @pytest.mark.asyncio
    async def test_grep_in_file(self, fresh_registry, tmp_workspace):
        from tools.atomics import grep
        f = tmp_workspace / "script.py"
        f.write_text("def foo():\n    pass\ndef bar():\n    pass")
        result = await grep(pattern="def foo", dir_path=str(tmp_workspace))
        assert "script.py" in result or "foo" in result

    @pytest.mark.asyncio
    async def test_grep_no_match(self, fresh_registry, tmp_workspace):
        from tools.atomics import grep
        f = tmp_workspace / "script.py"
        f.write_text("def foo():\n    pass")
        result = await grep(pattern="nonexistent", dir_path=str(tmp_workspace))
        assert "No matches" in result


class TestGlobTool:
    """Test glob tool."""

    @pytest.mark.asyncio
    async def test_glob_single_star(self, fresh_registry, tmp_workspace):
        from tools.atomics import glob
        (tmp_workspace / "a.txt").write_text("x")
        (tmp_workspace / "b.txt").write_text("x")
        (tmp_workspace / "c.py").write_text("x")
        result = await glob(path=str(tmp_workspace), pattern="*.txt")
        assert "a.txt" in result
        assert "b.txt" in result
        assert "c.py" not in result

    @pytest.mark.asyncio
    async def test_glob_no_match(self, fresh_registry, tmp_workspace):
        from tools.atomics import glob
        result = await glob(path=str(tmp_workspace), pattern="*.xyz")
        assert "No files match" in result


class TestMcpTools:
    """Test MCP resource tools (stubs)."""

    @pytest.mark.asyncio
    async def test_mcp_list_resources_returns_info(self, fresh_registry):
        from tools.atomics import mcp_list_resources
        result = await mcp_list_resources()
        assert "MCP" in result or "INFO" in result

    @pytest.mark.asyncio
    async def test_mcp_read_resource_returns_error(self, fresh_registry):
        from tools.atomics import mcp_read_resource
        result = await mcp_read_resource(server="test", uri="test://resource")
        assert "ERR" in result or "not found" in result.lower()


class TestToolSchemas:
    """Test that schemas are valid OpenAI function-calling format."""

    def test_all_schemas_have_required_fields(self, fresh_registry):
        init_registry()
        load_atomics()
        reg = get_registry()
        for tool_def in reg.list_by_category("atomic"):
            schema = tool_def.schema()
            func = schema["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            params = func["parameters"]
            assert params["type"] == "object"
            assert "properties" in params
