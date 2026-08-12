# SPDX-License-Identifier: Apache-2.0
"""Tool registry and execution engine.

Provides a unified registry for all tools (atomic, MCP) with statistics,
OpenAI schema generation, and error handling.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


# ─── Exceptions ─────────────────────────────────────────────────────────────────

_TOOL_NOT_FOUND_CODE = "TOOL_NOT_FOUND"
_TOOL_EXECUTION_CODE = "TOOL_EXECUTION_ERROR"
_TOOL_TIMEOUT_CODE = "TOOL_TIMEOUT"


class ToolRegistryError(Exception):
    """Base exception for tool registry errors."""

    def __init__(self, message: str, code: str = "TOOL_REGISTRY_ERROR"):
        self.message = message
        self.code = code
        super().__init__(message)


class ToolNotFoundError(ToolRegistryError):
    """Raised when a tool is not found in the registry."""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(
            message=f"Tool '{tool_name}' not found in registry",
            code=_TOOL_NOT_FOUND_CODE,
        )


class ToolExecutionError(ToolRegistryError):
    """Raised when a tool execution fails."""

    def __init__(
        self,
        tool_name: str,
        original_error: Exception,
    ):
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(
            message=f"Tool '{tool_name}' execution failed: {original_error}",
            code=_TOOL_EXECUTION_CODE,
        )


class ToolTimeoutError(ToolExecutionError):
    """Raised when a tool execution times out."""

    def __init__(self, tool_name: str, timeout_ms: float):
        self.timeout_ms = timeout_ms
        super().__init__(
            tool_name=tool_name,
            original_error=TimeoutError(f"Execution exceeded {timeout_ms}ms"),
        )


# ─── Statistics ─────────────────────────────────────────────────────────────────

@dataclass
class ToolStats:
    call_count: int = 0
    total_duration_ms: float = 0.0
    error_count: int = 0
    timeout_count: int = 0
    last_called: float = 0.0

    @property
    def avg_duration_ms(self) -> float:
        if not self.call_count:
            return 0.0
        return self.total_duration_ms / self.call_count

    @property
    def success_count(self) -> int:
        return self.call_count - self.error_count

    @property
    def success_rate(self) -> float:
        if not self.call_count:
            return 1.0
        return (self.call_count - self.error_count) / self.call_count


# ─── Tool Definition ────────────────────────────────────────────────────────────

ToolHandler = Callable[..., Any]  # sync or async callable


@dataclass
class ToolDefinition:
    """A registered tool in the registry."""

    name: str
    handler: ToolHandler
    description: str = ""
    parameters: dict[str, Any] | None = None
    category: str = "atomic"  # "atomic" | "mcp"
    tags: list[str] = field(default_factory=list)
    stats: ToolStats = field(default_factory=ToolStats)
    timeout_ms: float | None = None  # per-tool timeout override

    def schema(self) -> dict[str, Any]:
        """Return an OpenAI function-calling compatible schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }


# ─── Tool Registry ──────────────────────────────────────────────────────────────

ToolHandler = Callable[..., Any]  # sync or async callable


class ToolRegistry:
    """Unified tool registry and execution center.

    Registers and executes all tools (atomic, MCP) through a consistent
    interface. Provides statistics, schema generation, and structured error handling.

    Usage::

        registry = ToolRegistry()
        registry.register(handler=my_tool, name="my_tool", description="...", category="atomic")
        result = await registry.execute("my_tool", {"arg": "value"})
        schemas = registry.get_all_schemas()
        stats = registry.get_stats("my_tool")
    """

    # Category priority: higher number wins when names collide.
    # atomic > mcp: atomic tools are hand-crafted and trusted; MCP tools
    # from external servers may have names that collide (e.g. filesystem).
    _CATEGORY_PRIORITY: dict[str, int] = {"atomic": 2, "mcp": 1}

    def __init__(self, default_timeout_ms: float = 60_000.0):
        self._tools: dict[str, ToolDefinition] = {}
        self._default_timeout_ms = default_timeout_ms

    # ─── Registration ──────────────────────────────────────────────────────────────

    def register(
        self,
        *,
        name: str,
        handler: ToolHandler,
        description: str = "",
        parameters: dict[str, Any] | None = None,
        category: str = "atomic",
        tags: list[str] | None = None,
        timeout_ms: float | None = None,
    ) -> None:
        """Register a tool.

        Args:
            name: Unique tool name.
            handler: Sync or async callable that executes the tool.
            description: Human-readable description for LLM prompts.
            parameters: OpenAI function-calling parameter schema.
            category: Tool category ("atomic", "mcp").
            tags: Optional tags for filtering.
            timeout_ms: Per-tool timeout override (ms).

        When a name collision occurs, the tool with higher category priority wins:
        atomic (priority=2) > mcp (priority=1). The losing tool is silently skipped.
        """
        existing = self._tools.get(name)
        if existing is not None:
            new_prio = self._CATEGORY_PRIORITY.get(category, 0)
            existing_prio = self._CATEGORY_PRIORITY.get(existing.category, 0)
            if new_prio > existing_prio:
                import logging as _logging
                _logging.getLogger("tools.registry").warning(
                    "[ToolRegistry] name collision: '%s' (%s → %s, higher priority wins)",
                    name, existing.category, category,
                )
            else:
                return
        self._tools[name] = ToolDefinition(
            name=name,
            handler=handler,
            description=description,
            parameters=parameters,
            category=category,
            tags=tags or [],
            timeout_ms=timeout_ms,
        )

    def unregister(self, name: str) -> bool:
        """Unregister a tool. Returns True if it existed, False otherwise."""
        return self._tools.pop(name, None) is not None

    def is_registered(self, name: str) -> bool:
        """Return True if a tool is registered."""
        return name in self._tools

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool definition by name."""
        return self._tools.get(name)

    # ─── Query ──────────────────────────────────────────────────────────────────

    def list_all(self) -> list[ToolDefinition]:
        """List all registered tools."""
        return list(self._tools.values())

    def list_by_category(self, category: str) -> list[ToolDefinition]:
        """List all tools in a category."""
        return [t for t in self._tools.values() if t.category == category]

    def list_by_tag(self, tag: str) -> list[ToolDefinition]:
        """List all tools with a given tag."""
        return [t for t in self._tools.values() if tag in t.tags]

    def names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    # ─── Schema ─────────────────────────────────────────────────────────────────

    def get_all_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI function-calling schemas for all tools."""
        return [t.schema() for t in self._tools.values()]

    def get_schemas_by_names(self, names: list[str]) -> list[dict[str, Any]]:
        """Return schemas for specific tools (silently skips missing names)."""
        return [self._tools[n].schema() for n in names if n in self._tools]

    def get_schemas_by_category(self, category: str) -> list[dict[str, Any]]:
        """Return schemas for all tools in a category."""
        return [t.schema() for t in self.list_by_category(category)]

    def get_schemas_by_tags(self, tags: list[str]) -> list[dict[str, Any]]:
        """Return schemas for tools matching any of the given tags."""
        seen: set[str] = set()
        schemas: list[dict[str, Any]] = []
        for tag in tags:
            for t in self.list_by_tag(tag):
                if t.name not in seen:
                    seen.add(t.name)
                    schemas.append(t.schema())
        return schemas

    def get_schema(self, name: str) -> dict[str, Any] | None:
        """Return the schema for a single tool, or None if not found."""
        t = self._tools.get(name)
        return t.schema() if t else None

    # ─── Execution ──────────────────────────────────────────────────────────────

    async def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool by name with the given arguments.

        Args:
            name: Tool name.
            arguments: Tool arguments (kwargs-style).

        Returns:
            Tool execution result.

        Raises:
            ToolNotFoundError: If the tool is not registered.
            ToolTimeoutError: If execution exceeds the timeout.
            ToolExecutionError: If execution fails.
        """
        if name not in self._tools:
            raise ToolNotFoundError(name)

        definition = self._tools[name]
        start = time.perf_counter()
        definition.stats.call_count += 1
        definition.stats.last_called = start

        timeout = definition.timeout_ms or self._default_timeout_ms

        try:
            result = await asyncio.wait_for(
                definition.handler(**arguments),
                timeout=timeout / 1000.0,
            )
            return result
        except asyncio.TimeoutError:
            definition.stats.error_count += 1
            definition.stats.timeout_count += 1
            raise ToolTimeoutError(name, timeout)
        except Exception as exc:
            definition.stats.error_count += 1
            raise ToolExecutionError(tool_name=name, original_error=exc) from exc
        finally:
            definition.stats.total_duration_ms += (time.perf_counter() - start) * 1000

    async def execute_raw(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[Any, ToolStats]:
        """Execute a tool and return (result, stats snapshot) together."""
        t = self._tools.get(name)
        if t is None:
            raise ToolNotFoundError(name)
        snapshot_before = _copy_stats(t.stats)
        result = await self.execute(name, arguments)
        snapshot_after = _copy_stats(t.stats)
        return result, _diff_stats(snapshot_after, snapshot_before)

    # ─── Statistics ──────────────────────────────────────────────────────────────

    def get_stats(self, name: str | None = None) -> dict[str, Any]:
        """Get statistics for one tool or all tools.

        Args:
            name: If provided, return stats for that tool only.
                  If None, return a dict mapping each tool name to its stats.
        """
        if name:
            t = self._tools.get(name)
            if not t:
                return {}
            return _format_stats(t)
        return {t.name: _format_stats(t) for t in self._tools.values()}

    def get_summary(self) -> str:
        """Return a human-readable summary of all registered tools."""
        total_calls = sum(t.stats.call_count for t in self._tools.values())
        lines = [
            f"Tool Registry — {len(self._tools)} tools, {total_calls} total calls",
            "",
        ]
        for category in ("atomic", "mcp"):
            tools = self.list_by_category(category)
            if not tools:
                continue
            lines.append(f"## {category} ({len(tools)})")
            for t in sorted(tools, key=lambda x: -x.stats.call_count):
                rate = f"{t.stats.success_rate * 100:.1f}%" if t.stats.call_count else "-"
                lines.append(
                    f"  {t.name}: "
                    f"calls={t.stats.call_count}, "
                    f"avg={t.stats.avg_duration_ms:.1f}ms, "
                    f"success={rate}"
                )
            lines.append("")
        return "\n".join(lines)

    def reset_stats(self) -> None:
        """Reset statistics for all tools."""
        for t in self._tools.values():
            t.stats = ToolStats()

    def clear(self) -> None:
        """Unregister all tools. Use with caution."""
        self._tools.clear()


# ─── Helpers ────────────────────────────────────────────────────────────────────


def _copy_stats(s: ToolStats) -> dict[str, float]:
    return {
        "call_count": s.call_count,
        "error_count": s.error_count,
        "total_duration_ms": s.total_duration_ms,
        "timeout_count": s.timeout_count,
    }


def _diff_stats(after: dict[str, float], before: dict[str, float]) -> ToolStats:
    """Return a ToolStats reflecting only the delta between two snapshots."""
    stats = ToolStats()
    stats.call_count = after["call_count"] - before["call_count"]
    stats.error_count = after["error_count"] - before["error_count"]
    stats.total_duration_ms = after["total_duration_ms"] - before["total_duration_ms"]
    stats.timeout_count = after["timeout_count"] - before["timeout_count"]
    return stats


def _format_stats(t: ToolDefinition) -> dict[str, Any]:
    s = t.stats
    return {
        "name": t.name,
        "category": t.category,
        "call_count": s.call_count,
        "avg_duration_ms": round(s.avg_duration_ms, 2),
        "total_duration_ms": round(s.total_duration_ms, 2),
        "error_count": s.error_count,
        "timeout_count": s.timeout_count,
        "success_count": s.success_count,
        "success_rate": round(s.success_rate * 100, 2),
        "last_called": s.last_called,
    }
