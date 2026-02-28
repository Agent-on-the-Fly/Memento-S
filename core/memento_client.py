"""Memento tool manager — connects to FastMCP server with LangChain integration.

Uses ``fastmcp.Client`` for in-memory MCP transport and wraps discovered
tools as LangChain ``StructuredTool`` instances for use with LangGraph agents.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastmcp import Client
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from core.memento_server import mcp as _mcp_server, configure as _configure_server


class MementoToolManager:
    """Wraps the in-process FastMCP server, exposes LangChain tools."""

    def __init__(self) -> None:
        self._client: Client | None = None
        self._langchain_tools: list[StructuredTool] = []

    async def start(self, *, base_dir: Path | None = None) -> None:
        """Start the in-process MCP server and discover tools."""
        _configure_server(base_dir=base_dir)
        self._client = Client(_mcp_server)
        await self._client.__aenter__()
        raw_tools = await self._client.list_tools()
        self._langchain_tools = _build_langchain_tools(raw_tools, self._client)

    async def shutdown(self) -> None:
        if self._client is not None:
            await self._client.__aexit__(None, None, None)
            self._client = None

    def get_langchain_tools(self) -> list[StructuredTool]:
        """Return tools as LangChain ``StructuredTool`` instances."""
        return list(self._langchain_tools)


# ===================================================================
# Internal helpers
# ===================================================================

def _coerce_tool_args(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Coerce stringified JSON values back to native types.

    LLMs often serialize array/object parameters as JSON strings
    (e.g. ``"[1, 50]"`` instead of ``[1, 50]``). FastMCP's pydantic
    validation rejects these, so we parse them here.
    """
    out = dict(kwargs)
    for key, value in out.items():
        if not isinstance(value, str):
            continue
        stripped = value.strip()
        if stripped.startswith(("[", "{")):
            try:
                out[key] = json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                pass
    return out


def _extract_text(result: Any) -> str:
    """Extract text content from an MCP tool result."""
    if isinstance(result, list):
        parts: list[str] = []
        for block in result:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(result)


def _build_langchain_tools(
    tools: list,
    client: Client,
) -> list[StructuredTool]:
    """Convert MCP tools to LangChain ``StructuredTool`` instances."""
    lc_tools: list[StructuredTool] = []
    for t in tools:
        name = t.name if hasattr(t, "name") else str(t.get("name", ""))
        description = (
            t.description if hasattr(t, "description") else str(t.get("description", ""))
        )
        schema = (
            t.inputSchema if hasattr(t, "inputSchema") else t.get("inputSchema", {})
        )

        _tool_name = name  # capture in closure

        async def _call(
            _client_ref: Client = client,
            _name: str = _tool_name,
            **kwargs: Any,
        ) -> str:
            kwargs = _coerce_tool_args(kwargs)
            result = await _client_ref.call_tool(_name, kwargs)
            return _extract_text(result)

        lc_tools.append(
            StructuredTool(
                name=name,
                description=description,
                coroutine=_call,
                func=None,  # async-only
                args_schema=_json_schema_to_pydantic(name, schema),
            )
        )
    return lc_tools


def _json_schema_to_pydantic(tool_name: str, schema: dict) -> type[BaseModel]:
    """Convert a JSON schema dict to a Pydantic model class for ``args_schema``."""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}

    for prop_name, prop_schema in properties.items():
        prop_type_str = prop_schema.get("type", "string")
        python_type: type = str
        if prop_type_str == "integer":
            python_type = int
        elif prop_type_str == "number":
            python_type = float
        elif prop_type_str == "boolean":
            python_type = bool
        elif prop_type_str == "array":
            python_type = list
        elif prop_type_str == "object":
            python_type = dict

        field_desc = prop_schema.get("title", "") or prop_schema.get("description", "")
        default = prop_schema.get("default", ...)

        if prop_name in required:
            fields[prop_name] = (python_type, Field(description=field_desc))
        else:
            if default is ...:
                default = None
                python_type = python_type | None  # type: ignore[assignment]
            fields[prop_name] = (
                python_type,
                Field(default=default, description=field_desc),
            )

    model_name = f"{tool_name.title().replace('_', '')}Input"
    return create_model(model_name, **fields)
