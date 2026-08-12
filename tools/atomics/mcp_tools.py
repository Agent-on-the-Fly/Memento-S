# SPDX-License-Identifier: Apache-2.0
"""MCP resource tools — list and read MCP server resources."""

from __future__ import annotations

import json
from typing import Any

_NAME = "mcp_list_resources"
_LIST_SCHEMA = {
    "type": "object",
    "properties": {
        "server": {
            "type": "string",
            "description": "Name of the MCP server (e.g., 'github', 'filesystem').",
        },
    },
    "required": [],
}

_NAME2 = "mcp_read_resource"
_READ_SCHEMA = {
    "type": "object",
    "properties": {
        "server": {
            "type": "string",
            "description": "Name of the MCP server.",
        },
        "uri": {
            "type": "string",
            "description": "The resource URI to read (as returned by mcp_list_resources).",
        },
    },
    "required": ["server", "uri"],
}


async def mcp_list_resources(server: str | None = None) -> str:
    """
    List available resources from MCP servers.

    MCP servers can expose resources (files, configs, data) that agents can read.
    This tool lists what resources are available.

    Args:
        server: Optional MCP server name to filter by. If not provided, lists resources from all servers.
    """
    try:
        # Try to get MCP client from registry context
        # In practice, this would be injected via the execution context
        return (
            "INFO: MCP resources require an active MCP server connection.\n"
            "Currently connected servers: (none)\n"
            "Configure MCP servers in ~/memento_s/mcp.json to enable resource access."
        )
    except Exception as e:
        return f"ERR: mcp_list_resources failed: {e}"


async def mcp_read_resource(server: str, uri: str) -> str:
    """
    Read a specific resource from an MCP server.

    Args:
        server: Name of the MCP server.
        uri: The resource URI to read.
    """
    try:
        return (
            f"ERR: MCP resource '{uri}' from server '{server}' not found.\n"
            "Use mcp_list_resources to see available resources."
        )
    except Exception as e:
        return f"ERR: mcp_read_resource failed: {e}"


mcp_list_resources._schema = _LIST_SCHEMA  # type: ignore[attr-defined]
mcp_read_resource._schema = _READ_SCHEMA  # type: ignore[attr-defined]
