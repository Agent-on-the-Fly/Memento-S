# SPDX-License-Identifier: Apache-2.0
"""
Config schemas for Memento-S.

统一存放所有 Pydantic BaseModel 配置类。
"""

from .config_models import (
    AgentConfig,
    AppConfig,
    AuthConfig,
    DingTalkConfig,
    ExecutionConfig,
    FeishuConfig,
    GatewayConfig,
    GlobalConfig,
    IMConfig,
    LLMConfig,
    LLMProfile,
    LoggingConfig,
    OTAConfig,
    PathsConfig,
    RetrievalConfig,
    SkillEvolutionConfig,
    SkillsConfig,
    WechatConfig,
    WecomConfig,
)
from .mcp_config_schemas import (
    McpConfig,
    McpServerConfig,
    McpServerBase,
    McpStdioServer,
    McpHttpServer,
    OAuthConfig,
)
from .skill_config_schemas import (
    SkillEntry,
    SkillIndex,
    SkillRegistryConfig,
)

__all__ = [
    # GlobalConfig 体系
    "AgentConfig",
    "AppConfig",
    "AuthConfig",
    "DingTalkConfig",
    "ExecutionConfig",
    "FeishuConfig",
    "GatewayConfig",
    "GlobalConfig",
    "IMConfig",
    "LLMConfig",
    "LLMProfile",
    "LoggingConfig",
    "OTAConfig",
    "PathsConfig",
    "RetrievalConfig",
    "SkillEvolutionConfig",
    "SkillsConfig",
    "WechatConfig",
    "WecomConfig",
    # MCP Config
    "McpConfig",
    "McpServerConfig",
    "McpServerBase",
    "McpStdioServer",
    "McpHttpServer",
    "OAuthConfig",
    # Skill Registry Config
    "SkillEntry",
    "SkillIndex",
    "SkillRegistryConfig",
]
