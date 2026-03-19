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
    extra_headers: dict[str, Any] = Field(default_factory=dict)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    context_window: int = 100000
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 120

    @property
    def input_budget(self) -> int:
        """context_window 减去 max_tokens 后可用于输入的 token 预算。"""
        return self.context_window - self.max_tokens

    @property
    def provider(self) -> str:
        if "/" not in self.model:
            return ""
        return self.model.split("/", 1)[0]

    @property
    def model_name(self) -> str:
        if "/" not in self.model:
            return self.model
        return self.model.split("/", 1)[1]


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    active_profile: str
    profiles: dict[str, LLMProfile]

    @property
    def current(self) -> LLMProfile:
        if self.active_profile not in self.profiles:
            raise ValueError(
                f"Active profile '{self.active_profile}' not found in llm.profiles"
            )
        return self.profiles[self.active_profile]

    @property
    def current_profile(self) -> LLMProfile:
        return self.current

    @model_validator(mode="after")
    def _validate_active_profile(self) -> "LLMConfig":
        _ = self.current
        return self


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    top_k: int = 5
    min_score: float = 0.012
    embedding_model: str = "auto"
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


class StrategyConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    resolve_strategy: Literal["local_only", "local_first", "always_search"] = (
        "local_first"
    )
    download_method: Literal["github_api", "npx", "auto"] = "auto"


class SkillsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    catalog_path: str
    cloud_catalog_url: str | None = None
    retrieval: RetrievalConfig
    execution: ExecutionConfig
    strategy: StrategyConfig


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

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

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True, exclude_none=False)


__all__ = [
    "AppConfig",
    "LLMProfile",
    "LLMConfig",
    "RetrievalConfig",
    "ExecutionConfig",
    "StrategyConfig",
    "SkillsConfig",
    "PathsConfig",
    "LoggingConfig",
    "AgentConfig",
    "OTAConfig",
    "GlobalConfig",
]
