# SPDX-License-Identifier: Apache-2.0
"""
MCP Server Configuration Pydantic Models.

These models mirror the JSON Schema in mcp_config_schema.json but provide
Pydantic type-checking, validation, and IDE auto-complete.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class OAuthConfig(BaseModel):
    """OAuth authentication configuration (streamable_http transport only)."""

    model_config = ConfigDict(extra="ignore")

    clientId: str = ""
    clientSecret: str = ""
    scope: str = ""


class McpServerBase(BaseModel):
    """Base fields shared by all MCP server types."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    timeout: int = Field(default=5000, ge=1000, le=300000, description="Timeout in milliseconds")
    environment: dict[str, str | None] = Field(default_factory=dict)


class McpStdioServer(McpServerBase):
    """Local subprocess MCP server (stdio transport)."""

    transport: Literal["stdio"] = "stdio"
    command: str = Field(description="Executable path (e.g. npx, python, node)")
    args: list[str] = Field(default_factory=list)
    cwd: str | None = None


class McpHttpServer(McpServerBase):
    """Remote MCP server (streamable_http transport)."""

    transport: Literal["streamable_http"] = "streamable_http"
    url: HttpUrl = Field(description="Remote MCP server URL")
    headers: dict[str, str] = Field(default_factory=dict)
    authToken: str | None = None
    oauth: OAuthConfig | bool = False


# Discriminated union — McpServerConfig resolves to the correct subclass
# based on the transport field.
McpServerConfig = Annotated[
    McpStdioServer | McpHttpServer,
    Field(discriminator="transport"),
]


class McpConfig(BaseModel):
    """Top-level MCP configuration (mirrors mcp.json)."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    mcp: dict[str, McpServerConfig] = Field(default_factory=dict)

    def get_enabled_servers(self) -> dict[str, McpServerConfig]:
        """Return only servers that have enabled=True."""
        return {k: v for k, v in self.mcp.items() if v.enabled}


__all__ = [
    "McpConfig",
    "McpServerConfig",
    "McpServerBase",
    "McpStdioServer",
    "McpHttpServer",
    "OAuthConfig",
]
