# SPDX-License-Identifier: Apache-2.0
"""MCP tool loader — integrates MCP servers into the ToolRegistry.

Loads tools from configured MCP servers using the official `mcp` package,
converts them to async callables, and registers them in ToolRegistry.

Config is sourced from ~/memento_s/mcp.json (完整 mcp.json 格式)::

    {
      "enabled": true,
      "mcp": {
        "github": {
          "transport": "stdio",
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-github"],
          "environment": {},
          "enabled": true
        },
        "remote-api": {
          "transport": "streamable_http",
          "url": "https://mcp.example.com/mcp",
          "headers": { "Authorization": "Bearer ${MCP_TOKEN}" },
          "timeout": 5000,
          "authToken": "${MCP_TOKEN}",
          "enabled": true
        }
      }
    }

传输方式由 transport 字段显式指定：
- transport: "stdio"          → stdio_client (本地子进程)
- transport: "streamable_http" → streamable_http_client (HTTP 远程)
"""

from __future__ import annotations

import os
import re
from typing import Any

import httpx

from tools.registry import ToolRegistry


def _resolve_env_vars(value: Any) -> Any:
    """Resolve ${VAR_NAME} patterns in config values using os.environ."""
    if isinstance(value, str):
        return re.sub(
            r"\$\{([^}]+)\}",
            lambda m: os.environ.get(m.group(1), ""),
            value,
        )
    elif isinstance(value, dict):
        return {k: _resolve_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    return value


def _convert_json_schema(prop: dict[str, Any]) -> dict[str, Any]:
    """Convert a JSON Schema property definition to OpenAI function-calling format."""
    result = dict(prop)
    result.pop("jsonSchema", None)
    result.pop("$schema", None)
    return result


def _build_http_client(config: dict[str, Any]) -> httpx.AsyncClient:
    """根据 streamable_http 配置构建 httpx.AsyncClient。

    处理 headers、authToken、timeout 等字段，最终传给 streamable_http_client。
    """
    headers = dict(config.get("headers", {}))
    auth_token = config.get("authToken", None)

    if auth_token and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {auth_token}"

    timeout_ms = config.get("timeout", 5000)
    timeout_sec = timeout_ms / 1000.0
    timeout = httpx.Timeout(timeout_sec, read=timeout_sec * 2)

    return httpx.AsyncClient(
        headers=headers if headers else None,
        timeout=timeout,
        follow_redirects=True,
    )


class MCPToolLoader:
    """Loads tools from MCP servers and registers them to ToolRegistry.

    传输方式由 transport 字段指定：
    - "stdio"          → stdio_client (本地子进程)
    - "streamable_http" → streamable_http_client (HTTP 远程)

    Args:
        registry: The ToolRegistry instance to register tools into.
    """

    def __init__(self, registry: ToolRegistry):
        self._registry = registry
        self._sessions: dict[str, Any] = {}

    async def load_from_config(self, mcp_config: dict[str, Any]) -> None:
        """Load tools from all configured MCP servers.

        Args:
            mcp_config: 完整的 mcp.json 配置 dict，格式如下：
                {
                    "enabled": bool,
                    "mcp": {
                        "server_name": {
                            "transport": "stdio",
                            "command": "...", "args": [...], "enabled": true
                        },
                        "server_name": {
                            "transport": "streamable_http",
                            "url": "...", "headers": {}, "timeout": 5000, "authToken": "...", "enabled": true
                        }
                    }
                }
        """
        if not mcp_config.get("enabled", False):
            return

        servers = mcp_config.get("mcp", {})
        if not servers:
            return

        for name, cfg in servers.items():
            if not cfg.get("enabled", True):
                continue
            try:
                await self._load_server(name, _resolve_env_vars(cfg))
            except Exception as exc:
                import sys

                print(
                    f"[MCPToolLoader] Skipping server '{name}': {exc}",
                    file=sys.stderr,
                )

    async def _load_server(self, name: str, config: dict[str, Any]) -> None:
        """Load tools from a single MCP server.

        传输方式由 transport 字段指定：
        - "stdio"           → stdio 本地子进程
        - "streamable_http" → HTTP 远程服务
        """
        transport = config.get("transport", "").strip().lower() or "stdio"

        if transport == "stdio":
            await self._load_stdio_server(name, config)
        elif transport == "streamable_http":
            await self._load_http_server(name, config)
        else:
            raise ValueError(
                f"Server '{name}' transport 字段无效: '{transport}'，期望 'stdio' 或 'streamable_http'"
            )

    async def _load_stdio_server(self, name: str, config: dict[str, Any]) -> None:
        """通过 stdio 加载本地 MCP server（子进程模式）。"""
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        command = config["command"]
        args = config.get("args", [])
        env_extra = config.get("environment", {})

        env = {**os.environ}
        for k, v in env_extra.items():
            if v is not None:
                env[k] = v

        params = StdioServerParameters(command=command, args=args, env=env)

        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await self._register_tools(session, name)

    async def _load_http_server(self, name: str, config: dict[str, Any]) -> None:
        """通过 streamable_http 加载远程 MCP server。

        streamable_http_client 签名：
        streamable_http_client(url, *, http_client=None, terminate_on_close=True)
        → http_client 支持传入预配置好的 httpx.AsyncClient，可在其中设置 headers/timeout/auth
        """
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        url = config["url"]
        http_client = _build_http_client(config)

        async with http_client:
            async with streamable_http_client(
                url,
                http_client=http_client,
                terminate_on_close=True,
            ) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    await self._register_tools(session, name)

    async def _register_tools(self, session: Any, server_name: str) -> None:
        """列出 server 支持的工具并注册到 ToolRegistry。"""
        list_result = await session.list_tools()

        for mcp_tool in list_result.tools:
            tool_name = f"{server_name}_{mcp_tool.name}"

            async def make_caller(mcp_tool=mcp_tool, session=session):
                async def caller(**kwargs: Any) -> str:
                    result = await session.call_tool(mcp_tool.name, kwargs)
                    texts = []
                    for content in result.content:
                        if content.type == "text":
                            texts.append(content.text)
                        elif content.type == "image":
                            data = getattr(content, "data", "")
                            texts.append(f"[image: {data[:100]}...]")
                        elif content.type == "resource":
                            texts.append(f"[resource: {getattr(content, 'uri', '')}]")
                    return "\n".join(texts) if texts else "(no output)"

                return caller

            mcp_tool_caller = await make_caller()

            input_schema = getattr(mcp_tool, "inputSchema", None) or {}
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])

            schema = {
                "type": "object",
                "properties": {
                    prop_name: _convert_json_schema(prop)
                    for prop_name, prop in properties.items()
                },
                "required": required,
            }

            description = (
                getattr(mcp_tool, "description", "")
                or getattr(mcp_tool, "title", "")
                or f"MCP tool from server '{server_name}'"
            )

            self._registry.register(
                name=tool_name,
                handler=mcp_tool_caller,
                description=description,
                parameters=schema,
                category="mcp",
                tags=["mcp", f"server:{server_name}"],
            )

        self._sessions[server_name] = session

    async def close(self) -> None:
        """关闭所有 MCP server 连接。"""
        for name, session in list(self._sessions.items()):
            try:
                await session.close()
            except Exception:
                pass
        self._sessions.clear()