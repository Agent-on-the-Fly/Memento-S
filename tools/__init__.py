# SPDX-License-Identifier: Apache-2.0
"""Memento-S unified tool registry and execution interface.

Architectural layers:
  - tools/atomics/ : Native atomic tools (async callables)
  - tools/mcp/     : MCP tool loader and configuration
  - tools/registry.py: ToolRegistry — registration + execution + stats

Testing:
  - tools/tests/: Complete test coverage for all layers
"""

from __future__ import annotations

from typing import Any

from middleware.config import g_config
from tools.registry import (
    ToolRegistry,
    ToolDefinition,
    ToolStats,
    ToolRegistryError,
    ToolNotFoundError,
    ToolExecutionError,
    ToolTimeoutError,
)

# ─── Global singleton registry ───────────────────────────────────────────────

_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """Get the global ToolRegistry singleton."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def init_registry(default_timeout_ms: float = 60_000.0) -> ToolRegistry:
    """Initialize the global registry. Call once during bootstrap.

    Idempotent: if a registry is already initialized, returns the existing
    instance without replacing it (unless default_timeout_ms differs, which
    is ignored for existing instances).
    """
    global _registry
    if _registry is None:
        _registry = ToolRegistry(default_timeout_ms=default_timeout_ms)
    return _registry


# ─── Execution ───────────────────────────────────────────────────────────────

async def execute_tool(name: str, arguments: dict) -> Any:
    """Execute a tool by name with given arguments."""
    return await get_registry().execute(name, arguments)


# ─── Schema ──────────────────────────────────────────────────────────────────

def get_tool_schemas(
    names: list[str] | None = None,
    category: str | None = None,
    tags: list[str] | None = None,
) -> list[dict]:
    """Get tool schemas filtered by names, category, or tags."""
    reg = get_registry()
    if names is not None:
        return reg.get_schemas_by_names(names)
    if category:
        return reg.get_schemas_by_category(category)
    if tags:
        return reg.get_schemas_by_tags(tags)
    return reg.get_all_schemas()


# ─── Stats ───────────────────────────────────────────────────────────────────

def get_tools_summary() -> str:
    return get_registry().get_summary()


def get_tool_stats(name: str | None = None) -> dict:
    return get_registry().get_stats(name)


# ─── Management ──────────────────────────────────────────────────────────────

def is_tool_registered(name: str) -> bool:
    return get_registry().is_registered(name)


def list_tools(category: str | None = None) -> list[ToolDefinition]:
    reg = get_registry()
    if category:
        return reg.list_by_category(category)
    return reg.list_all()


# ─── MCP ─────────────────────────────────────────────────────────────────────

async def load_mcp_tools(mcp_config: dict[str, Any] | None = None) -> None:
    """Load MCP tools from config dict.

    Args:
        mcp_config: MCP configuration dict (sourced from McpConfigManager).
    """
    if not mcp_config:
        return
    try:
        from tools.mcp.client import MCPToolLoader
    except ImportError:
        return
    loader = MCPToolLoader(get_registry())
    await loader.load_from_config(mcp_config)


# ─── Atomics ─────────────────────────────────────────────────────────────────

def load_atomics() -> None:
    """Register all atomic tools into the registry (Layer 1)."""
    from tools.atomics import (
        list_dir,
        read_file,
        file_create,
        edit_file_by_lines,
        grep,
        bash,
        python_repl,
        js_repl,
        search_web,
        fetch_webpage,
        glob,
        mcp_list_resources,
        mcp_read_resource,
    )

    # Inject bash timeout from config
    bash_timeout = getattr(g_config.skills.execution, "bash_timeout_sec", 300) or 300
    bash_timeout_ms = bash_timeout * 1000

    reg = get_registry()
    for name, handler, description, parameters in [
        ("list_dir", list_dir, list_dir.__doc__ or "", list_dir._schema),
        ("read_file", read_file, read_file.__doc__ or "", read_file._schema),
        ("file_create", file_create, file_create.__doc__ or "", file_create._schema),
        (
            "edit_file_by_lines",
            edit_file_by_lines,
            edit_file_by_lines.__doc__ or "",
            edit_file_by_lines._schema,
        ),
        ("grep", grep, grep.__doc__ or "", grep._schema),
        ("bash", bash, bash.__doc__ or "", bash._schema),
        ("python_repl", python_repl, python_repl.__doc__ or "", python_repl._schema),
        ("js_repl", js_repl, js_repl.__doc__ or "", js_repl._schema),
        ("search_web", search_web, search_web.__doc__ or "", search_web._schema),
        (
            "fetch_webpage",
            fetch_webpage,
            fetch_webpage.__doc__ or "",
            fetch_webpage._schema,
        ),
        ("glob", glob, glob.__doc__ or "", glob._schema),
        ("mcp_list_resources", mcp_list_resources, mcp_list_resources.__doc__ or "", mcp_list_resources._schema),
        ("mcp_read_resource", mcp_read_resource, mcp_read_resource.__doc__ or "", mcp_read_resource._schema),
    ]:
        reg.register(
            name=name,
            handler=handler,
            description=description,
            parameters=parameters,
            category="atomic",
            tags=["atomic"],
            timeout_ms=bash_timeout_ms if name == "bash" else None,
        )


# ─── Bootstrap ───────────────────────────────────────────────────────────────

async def bootstrap(
    mcp_config: dict[str, Any] | None = None,
) -> ToolRegistry:
    """Initialize the entire tools system.

    Call once during application bootstrap::

        from middleware.config.mcp_config_manager import g_mcp_config_manager
        registry = await tools.bootstrap(mcp_config=g_mcp_config_manager.get_mcp_config())

    Args:
        mcp_config: 完整的 mcp.json 配置 dict（包含顶层 enabled 和 mcp servers 字典）。
                    从 McpConfigManager.get_mcp_config() 获取。传入 None 则跳过 MCP tools。

    Returns:
        The initialized registry.
    """
    reg = init_registry()

    # Layer 1: atomic tools
    load_atomics()

    # Layer 2: MCP tools (dynamic)
    if mcp_config:
        await load_mcp_tools(mcp_config)

    return reg
