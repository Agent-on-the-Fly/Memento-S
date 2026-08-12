# SPDX-License-Identifier: Apache-2.0
"""Tests for ToolRegistry core functionality."""

from __future__ import annotations

import asyncio
import time

import pytest

from tools import (
    ToolRegistry,
    ToolNotFoundError,
    ToolExecutionError,
    ToolTimeoutError,
)
from tools.registry import ToolDefinition, ToolStats


class TestRegistration:
    """Test tool registration and management."""

    def test_register_single_tool(self, fresh_registry, sample_tool):
        fresh_registry.register(
            name="dummy_tool",
            handler=sample_tool,
            description="A dummy tool for testing.",
            parameters=sample_tool._schema,
            category="atomic",
            tags=["test"],
        )
        assert fresh_registry.is_registered("dummy_tool") is True
        assert len(fresh_registry.list_all()) == 1

    def test_register_duplicate_overwrites(self, fresh_registry, sample_tool):
        """Later registration always wins when priorities are equal (both atomic → last wins)."""
        fresh_registry.register(
            name="dummy_tool",
            handler=sample_tool,
            category="atomic",
        )
        # Same category: second registration overwrites the first
        fresh_registry.register(
            name="dummy_tool",
            handler=sample_tool,
            category="atomic",
        )
        assert fresh_registry.get("dummy_tool").category == "atomic"

    def test_unregister_existing(self, fresh_registry, sample_tool):
        fresh_registry.register(name="dummy_tool", handler=sample_tool)
        assert fresh_registry.unregister("dummy_tool") is True
        assert fresh_registry.is_registered("dummy_tool") is False

    def test_unregister_nonexistent_returns_false(self, fresh_registry):
        assert fresh_registry.unregister("nonexistent") is False

    def test_get_existing(self, fresh_registry, sample_tool):
        fresh_registry.register(name="dummy_tool", handler=sample_tool)
        td = fresh_registry.get("dummy_tool")
        assert td is not None
        assert td.name == "dummy_tool"

    def test_get_nonexistent(self, fresh_registry):
        assert fresh_registry.get("nonexistent") is None

    def test_list_by_category(self, fresh_registry, sample_tool):
        fresh_registry.register(name="t1", handler=sample_tool, category="atomic")
        fresh_registry.register(name="t2", handler=sample_tool, category="mcp")
        assert len(fresh_registry.list_by_category("atomic")) == 1
        assert len(fresh_registry.list_by_category("mcp")) == 1
        assert len(fresh_registry.list_by_category("skill")) == 0

    def test_list_by_tag(self, fresh_registry, sample_tool):
        fresh_registry.register(
            name="t1",
            handler=sample_tool,
            tags=["test", "atomic"],
        )
        fresh_registry.register(
            name="t2",
            handler=sample_tool,
            tags=["test"],
        )
        assert len(fresh_registry.list_by_tag("test")) == 2
        assert len(fresh_registry.list_by_tag("atomic")) == 1

    def test_names(self, fresh_registry, sample_tool):
        fresh_registry.register(name="a", handler=sample_tool)
        fresh_registry.register(name="b", handler=sample_tool)
        assert set(fresh_registry.names()) == {"a", "b"}


class TestSchemaGeneration:
    """Test OpenAI schema generation."""

    def test_schema_from_tool_definition(self, fresh_registry, sample_tool):
        fresh_registry.register(
            name="dummy_tool",
            handler=sample_tool,
            description="Test tool",
            parameters={"type": "object", "properties": {"arg": {"type": "string"}}},
        )
        schemas = fresh_registry.get_all_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "dummy_tool"

    def test_get_all_schemas(self, fresh_registry, sample_tool):
        fresh_registry.register(name="t1", handler=sample_tool)
        fresh_registry.register(name="t2", handler=sample_tool)
        schemas = fresh_registry.get_all_schemas()
        assert len(schemas) == 2

    def test_get_schemas_by_names(self, fresh_registry, sample_tool):
        fresh_registry.register(name="t1", handler=sample_tool)
        fresh_registry.register(name="t2", handler=sample_tool)
        schemas = fresh_registry.get_schemas_by_names(["t1"])
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "t1"

    def test_get_schemas_by_names_skips_missing(self, fresh_registry, sample_tool):
        fresh_registry.register(name="t1", handler=sample_tool)
        schemas = fresh_registry.get_schemas_by_names(["t1", "nonexistent"])
        assert len(schemas) == 1

    def test_get_schemas_by_category(self, fresh_registry, sample_tool):
        fresh_registry.register(name="t1", handler=sample_tool, category="atomic")
        fresh_registry.register(name="t2", handler=sample_tool, category="mcp")
        schemas = fresh_registry.get_schemas_by_category("atomic")
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "t1"

    def test_get_schema_single(self, fresh_registry, sample_tool):
        fresh_registry.register(name="t1", handler=sample_tool)
        schema = fresh_registry.get_schema("t1")
        assert schema is not None
        assert schema["function"]["name"] == "t1"

    def test_get_schema_nonexistent(self, fresh_registry):
        assert fresh_registry.get_schema("nonexistent") is None


class TestNameCollision:
    """Test that register() respects category priority (atomic > mcp)."""

    def test_atomic_wins_over_mcp(self, fresh_registry, sample_tool):
        """When the same name is registered by both atomic and mcp, atomic wins."""
        fresh_registry.register(name="collision", handler=sample_tool, category="atomic")
        fresh_registry.register(name="collision", handler=sample_tool, category="mcp")
        assert fresh_registry.get("collision").category == "atomic"

    def test_mcp_skipped_when_atomic_exists(self, fresh_registry, sample_tool):
        """MCP registration is silently dropped when atomic already registered."""
        fresh_registry.register(name="foo", handler=sample_tool, category="atomic")
        fresh_registry.register(name="foo", handler=sample_tool, category="mcp")
        assert len(fresh_registry.list_by_category("atomic")) == 1
        assert len(fresh_registry.list_by_category("mcp")) == 0

    def test_first_atomic_kept(self, fresh_registry, sample_tool):
        """First atomic registration is kept when later mcp tries to override."""
        fresh_registry.register(name="bar", handler=sample_tool, category="atomic")
        fresh_registry.register(name="bar", handler=sample_tool, category="mcp")
        assert fresh_registry.get("bar") is not None
        assert fresh_registry.get("bar").category == "atomic"

    def test_first_mcp_skipped_when_later_atomic(self, fresh_registry, sample_tool):
        """First mcp registration is replaced when later atomic registers same name."""
        fresh_registry.register(name="baz", handler=sample_tool, category="mcp")
        fresh_registry.register(name="baz", handler=sample_tool, category="atomic")
        assert fresh_registry.get("baz").category == "atomic"


class TestExecution:
    """Test tool execution with stats and error handling."""

    @pytest.mark.asyncio
    async def test_execute_success(self, fresh_registry, sample_tool):
        fresh_registry.register(name="dummy_tool", handler=sample_tool)
        result = await fresh_registry.execute("dummy_tool", {"arg": "hello"})
        assert result == "dummy: hello"

    @pytest.mark.asyncio
    async def test_execute_not_found_raises(self, fresh_registry):
        with pytest.raises(ToolNotFoundError) as exc_info:
            await fresh_registry.execute("nonexistent", {})
        assert "nonexistent" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_updates_stats(self, fresh_registry, sample_tool):
        fresh_registry.register(name="dummy_tool", handler=sample_tool)
        await fresh_registry.execute("dummy_tool", {"arg": "x"})
        stats = fresh_registry.get_stats("dummy_tool")
        assert stats["call_count"] == 1
        assert stats["error_count"] == 0
        assert stats["success_rate"] == 100.0

    @pytest.mark.asyncio
    async def test_execute_error_increments_error_count(self, fresh_registry, failing_tool):
        fresh_registry.register(name="failing_tool", handler=failing_tool)
        with pytest.raises(ToolExecutionError):
            await fresh_registry.execute("failing_tool", {})
        stats = fresh_registry.get_stats("failing_tool")
        assert stats["error_count"] == 1

    @pytest.mark.asyncio
    async def test_execute_error_contains_original(self, fresh_registry, failing_tool):
        fresh_registry.register(name="failing_tool", handler=failing_tool)
        with pytest.raises(ToolExecutionError) as exc_info:
            await fresh_registry.execute("failing_tool", {"reason": "test error"})
        assert exc_info.value.original_error is not None
        assert str(exc_info.value.original_error) == "test error"

    @pytest.mark.asyncio
    async def test_execute_timing_recorded(self, fresh_registry, sample_tool):
        fresh_registry.register(name="dummy_tool", handler=sample_tool)
        start = time.perf_counter()
        await fresh_registry.execute("dummy_tool", {})
        elapsed_ms = (time.perf_counter() - start) * 1000
        stats = fresh_registry.get_stats("dummy_tool")
        assert stats["total_duration_ms"] >= 0
        assert stats["avg_duration_ms"] >= 0
        # Sanity: should be very fast
        assert elapsed_ms < 1000

    @pytest.mark.asyncio
    async def test_execute_timeout(self, fresh_registry):
        async def slow_tool() -> str:
            await asyncio.sleep(2)
            return "done"

        fresh_registry.register(
            name="slow_tool",
            handler=slow_tool,
            timeout_ms=100,
        )
        with pytest.raises(ToolTimeoutError) as exc_info:
            await fresh_registry.execute("slow_tool", {})
        assert exc_info.value.timeout_ms == 100
        assert exc_info.value.tool_name == "slow_tool"

    @pytest.mark.asyncio
    async def test_execute_timeout_updates_stats(self, fresh_registry):
        async def slow_tool() -> str:
            await asyncio.sleep(2)
            return "done"

        fresh_registry.register(
            name="slow_tool",
            handler=slow_tool,
            timeout_ms=100,
        )
        try:
            await fresh_registry.execute("slow_tool", {})
        except ToolTimeoutError:
            pass
        stats = fresh_registry.get_stats("slow_tool")
        assert stats["timeout_count"] == 1
        assert stats["error_count"] == 1


class TestStats:
    """Test statistics collection and reporting."""

    def test_get_stats_single_tool(self, fresh_registry, sample_tool):
        fresh_registry.register(name="dummy_tool", handler=sample_tool)
        stats = fresh_registry.get_stats("dummy_tool")
        assert stats["call_count"] == 0
        assert stats["error_count"] == 0
        assert stats["success_rate"] == 100.0

    def test_get_stats_nonexistent(self, fresh_registry):
        assert fresh_registry.get_stats("nonexistent") == {}

    def test_get_stats_all(self, fresh_registry, sample_tool):
        fresh_registry.register(name="t1", handler=sample_tool)
        fresh_registry.register(name="t2", handler=sample_tool)
        all_stats = fresh_registry.get_stats()
        assert "t1" in all_stats
        assert "t2" in all_stats

    def test_reset_stats(self, fresh_registry, sample_tool):
        fresh_registry.register(name="dummy_tool", handler=sample_tool)
        fresh_registry.get_stats()  # trigger init
        fresh_registry.reset_stats()
        stats = fresh_registry.get_stats("dummy_tool")
        assert stats["call_count"] == 0
        assert stats["total_duration_ms"] == 0.0

    def test_tool_stats_success_rate_zero_calls(self, fresh_registry, sample_tool):
        fresh_registry.register(name="dummy_tool", handler=sample_tool)
        s = fresh_registry.get_stats("dummy_tool")["success_rate"]
        # No calls → treat as 100% (avoid division by zero in rate)
        assert s == 100.0

    def test_get_summary_format(self, fresh_registry, sample_tool):
        fresh_registry.register(name="t1", handler=sample_tool, category="atomic")
        summary = fresh_registry.get_summary()
        assert "Tool Registry" in summary
        assert "atomic" in summary
        assert "t1" in summary


class TestToolDefinition:
    """Test ToolDefinition dataclass."""

    def test_tool_definition_schema(self):
        td = ToolDefinition(
            name="test_tool",
            handler=lambda: "x",
            description="A test tool",
            parameters={"type": "object", "properties": {"arg": {"type": "string"}}},
            category="atomic",
            tags=["test"],
        )
        schema = td.schema()
        assert schema["function"]["name"] == "test_tool"
        assert schema["function"]["description"] == "A test tool"
        assert schema["function"]["parameters"]["properties"]["arg"]["type"] == "string"

    def test_tool_definition_default_parameters(self):
        td = ToolDefinition(
            name="test_tool",
            handler=lambda: "x",
        )
        schema = td.schema()
        # Should have default empty parameters
        assert schema["function"]["parameters"]["properties"] == {}


class TestClearAndReInit:
    """Test registry lifecycle."""

    def test_clear_removes_all(self, fresh_registry, sample_tool):
        fresh_registry.register(name="t1", handler=sample_tool)
        fresh_registry.register(name="t2", handler=sample_tool)
        fresh_registry.clear()
        assert len(fresh_registry.list_all()) == 0

    def test_reinit_after_clear(self, fresh_registry, sample_tool):
        fresh_registry.register(name="t1", handler=sample_tool)
        fresh_registry.clear()
        fresh_registry.register(name="t2", handler=sample_tool)
        assert fresh_registry.is_registered("t2")
        assert not fresh_registry.is_registered("t1")
