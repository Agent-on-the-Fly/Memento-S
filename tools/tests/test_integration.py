# SPDX-License-Identifier: Apache-2.0
"""End-to-end integration tests for the tools module bootstrap."""

from __future__ import annotations

import asyncio

import pytest

from tools import (
    bootstrap,
    get_registry,
    init_registry,
    get_tool_schemas,
    is_tool_registered,
)


class TestBootstrap:
    """Test the full bootstrap process."""

    @pytest.mark.asyncio
    async def test_bootstrap_loads_all_atomic_tools(self):
        reg = await bootstrap(mcp_config={"enabled": False})

        # Layer 1: atomic tools
        atomic_schemas = get_tool_schemas(category="atomic")
        atomic_names = {s["function"]["name"] for s in atomic_schemas}
        expected = {
            "list_dir", "read_file", "file_create", "edit_file_by_lines",
            "grep", "bash", "python_repl", "js_repl",
            "search_web", "fetch_webpage",
            "glob", "mcp_list_resources", "mcp_read_resource",
        }
        assert expected.issubset(atomic_names), f"Missing: {expected - atomic_names}"

    @pytest.mark.asyncio
    async def test_bootstrap_is_singleton(self):
        reg1 = await bootstrap(mcp_config={"enabled": False})
        reg2 = get_registry()
        assert reg1 is reg2  # same singleton instance

    @pytest.mark.asyncio
    async def test_bootstrap_idempotent(self):
        reg1 = await bootstrap(mcp_config={"enabled": False})
        reg2 = await bootstrap(mcp_config={"enabled": False})
        assert reg1 is reg2

    @pytest.mark.asyncio
    async def test_bootstrap_mcp_disabled(self):
        """When MCP is disabled, no mcp tools are registered."""
        reg = await bootstrap(mcp_config={"enabled": False})
        mcp_tools = reg.list_by_category("mcp")
        assert len(mcp_tools) == 0

    @pytest.mark.asyncio
    async def test_all_tools_have_valid_schemas(self):
        """Every registered tool must have a valid OpenAI function-calling schema."""
        await bootstrap(mcp_config={"enabled": False})
        reg = get_registry()
        for td in reg.list_all():
            schema = td.schema()
            func = schema["function"]
            assert "name" in func, f"{td.name}: missing 'name'"
            assert "description" in func, f"{td.name}: missing 'description'"
            assert "parameters" in func, f"{td.name}: missing 'parameters'"
            params = func["parameters"]
            assert params["type"] == "object", f"{td.name}: parameters.type must be 'object'"
            assert "properties" in params, f"{td.name}: missing 'properties'"


class TestConcurrentExecution:
    """Test concurrent tool execution safety."""

    @pytest.mark.asyncio
    async def test_concurrent_executions_are_isolated(self, fresh_registry):
        """Different invocations of the same tool with different args run independently."""
        from tools.registry import ToolRegistry

        reg: ToolRegistry = fresh_registry

        async def echo_tool(msg: str = "") -> str:
            return msg

        echo_tool._schema = {
            "type": "object",
            "properties": {"msg": {"type": "string"}},
            "required": [],
        }

        reg.register(name="echo", handler=echo_tool)

        # Execute same tool concurrently with different args
        results = await asyncio.gather(
            reg.execute("echo", {"msg": "a"}),
            reg.execute("echo", {"msg": "b"}),
            reg.execute("echo", {"msg": "c"}),
        )
        assert set(results) == {"a", "b", "c"}

    @pytest.mark.asyncio
    async def test_concurrent_different_tools(self, fresh_registry):
        """Multiple different tools can execute concurrently."""
        from tools.registry import ToolRegistry

        reg: ToolRegistry = fresh_registry

        async def tool1() -> str:
            await asyncio.sleep(0.05)
            return "tool1"

        async def tool2() -> str:
            await asyncio.sleep(0.05)
            return "tool2"

        tool1._schema = {"type": "object", "properties": {}, "required": []}
        tool2._schema = {"type": "object", "properties": {}, "required": []}

        reg.register(name="t1", handler=tool1)
        reg.register(name="t2", handler=tool2)

        import time
        start = time.perf_counter()
        results = await asyncio.gather(
            reg.execute("t1", {}),
            reg.execute("t2", {}),
        )
        elapsed = time.perf_counter() - start

        # If they ran concurrently, should take ~50ms not 100ms
        assert elapsed < 0.15, f"Took {elapsed:.2f}s — tools may not be running concurrently"
        assert set(results) == {"tool1", "tool2"}
