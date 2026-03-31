"""
Memento-S 配置模型模块
包含所有的 Pydantic BaseModel 配置类
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    theme: str = "system"
    language: str = " "
    theme_options: dict[str, dict[str, str]] | None = None
    language_options: dict[str, dict[str, str]] | None = None


class LLMProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    api_key: str | None = None
    base_url: str | None = None
    litellm_provider: str = ""
    extra_headers: dict[str, Any] = Field(default_factory=dict)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    context_window: int = 100000
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 120

    @property
    def input_budget(self) -> int:
        """context_window 减去 max_tokens 后可用于输入的 token 预算。

        当 max_tokens >= context_window 时（配置错误），返回 context_window 的 50%
        作为安全降级，避免产生负值导致 compact 逻辑异常。
        """
        budget = self.context_window - self.max_tokens
        if budget <= 0:
            return max(self.context_window // 2, 4096)
        return budget

    @property
    def provider(self) -> str:
        if not self.model or "/" not in self.model:
            return ""
        return self.model.split("/", 1)[0]

    @property
    def model_name(self) -> str:
        if not self.model:
            return ""
        if "/" not in self.model:
            return self.model
        return self.model.split("/", 1)[1]


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    active_profile: str = ""
    profiles: dict[str, LLMProfile] = Field(default_factory=dict)

    @property
    def current(self) -> LLMProfile | None:
        if not self.active_profile or self.active_profile not in self.profiles:
            return None
        return self.profiles[self.active_profile]

    @property
    def current_profile(self) -> LLMProfile | None:
        return self.current

    @model_validator(mode="after")
    def _validate_active_profile(self) -> "LLMConfig":
        # 只有当有 profiles 且设置了 active_profile 时才验证
        if self.profiles and self.active_profile:
            if self.active_profile not in self.profiles:
                raise ValueError(
                    f"Active profile '{self.active_profile}' not found in llm.profiles"
                )
        return self


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    top_k: int = 5
    min_score: float = 0.012
    embedding_model: str = "auto"
    embedding_dimension: int = 1536  # text-embedding-3-small 默认维度
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    reranker_enabled: bool = True
    reranker_min_score: float = 0.001


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    timeout_sec: int = 30
    sandbox_provider: Literal["local", "e2b", "modal", "uv"] = "uv"
    e2b_api_key: str | None = None
    # uv sandbox 配置
    uv_python_version: str = "3.12"
    # 执行超时配置
    bash_timeout_sec: int = 300
    pip_install_timeout_sec: int = 180
    cli_install_timeout_sec: int = 300
    # skill 执行恢复策略
    max_attempts: int = 3
    same_signature_limit: int = 2


class SkillsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    catalog_path: str
    cloud_catalog_url: str | None = None
    retrieval: RetrievalConfig
    execution: ExecutionConfig


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    data_dir: Path | None = None  # 数据根目录，其他路径的父目录
    workspace_dir: Path | None = None
    skills_dir: Path | None = None
    db_dir: Path | None = None
    logs_dir: Path | None = None
    venv_dir: Path | None = None  # uv venv 目录，默认 .venv
    context_dir: Path | None = None  # context 数据目录，默认 {workspace_dir}/context/
    path_validation_enabled: bool = False


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_iterations: int = 100


class OTAConfig(BaseModel):
    """OTA update configuration."""

    model_config = ConfigDict(extra="ignore")
    url: str | None = None
    auto_check: bool = True  # Check for updates on startup
    auto_download: bool = True  # Auto download updates
    check_interval_hours: int = 24  # Check interval (0 = check every startup)
    notify_on_complete: bool = True  # Show notification when download completes
    install_confirmation: bool = True  # Ask before installing


class FeishuConfig(BaseModel):
    """Feishu (Lark) IM platform configuration."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    app_id: str | None = None
    app_secret: str | None = None
    webhook_url: str | None = None
    encrypt_key: str | None = None
    verification_token: str | None = None
    base_url: str = "https://open.feishu.cn"


class DingTalkConfig(BaseModel):
    """DingTalk IM platform configuration."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    app_key: str | None = None
    app_secret: str | None = None
    webhook_url: str | None = None
    webhook_secret: str | None = None
    base_url: str = "https://api.dingtalk.com"


class WecomConfig(BaseModel):
    """WeCom (Enterprise WeChat) IM platform configuration."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    corp_id: str | None = None
    agent_id: str | None = None
    secret: str | None = None
    token: str | None = None
    encoding_aes_key: str | None = None
    bot_id: str | None = None
    bot_secret: str | None = None
    webhook_url: str | None = None


class WechatConfig(BaseModel):
    """WeChat (Personal) IM platform configuration."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    base_url: str = "https://ilinkai.weixin.qq.com"
    token: str | None = None


class IMConfig(BaseModel):
    """IM platform integration configuration."""

    model_config = ConfigDict(extra="ignore")

    platform: str = "feishu"
    feishu: FeishuConfig = Field(default_factory=FeishuConfig)
    dingtalk: DingTalkConfig = Field(default_factory=DingTalkConfig)
    wecom: WecomConfig = Field(default_factory=WecomConfig)
    wechat: WechatConfig = Field(default_factory=WechatConfig)


class GatewayConfig(BaseModel):
    """Gateway mode configuration for IM integration."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True  # 默认开启 Gateway
    mode: Literal["bridge", "gateway"] = "bridge"
    websocket_host: str = "127.0.0.1"
    websocket_port: int = 8765
    webhook_host: str = "127.0.0.1"
    webhook_port: int = 18080


class AuthConfig(BaseModel):
    """Authentication configuration (system-level, not user-editable)."""

    model_config = ConfigDict(extra="ignore")

    base_url: str = ""


class GlobalConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    config_schema: str | None = Field(default=None, alias="$schema")
    version: str = "1.0.0"
    app: AppConfig
    llm: LLMConfig
    skills: SkillsConfig
    paths: PathsConfig
    logging: LoggingConfig
    agent: AgentConfig
    env: dict[str, Any] | None = None
    ota: OTAConfig = Field(default_factory=OTAConfig)
    im: IMConfig = Field(default_factory=IMConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True, exclude_none=False)


__all__ = [
    "AppConfig",
    "LLMProfile",
    "LLMConfig",
    "RetrievalConfig",
    "ExecutionConfig",
    "SkillsConfig",
    "PathsConfig",
    "LoggingConfig",
    "AgentConfig",
    "OTAConfig",
    "FeishuConfig",
    "DingTalkConfig",
    "WecomConfig",
    "WechatConfig",
    "IMConfig",
    "GatewayConfig",
    "AuthConfig",
    "GlobalConfig",
]
