
from __future__ import annotations

from pathlib import Path
from typing import Literal

from dotenv import find_dotenv, load_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv(find_dotenv())


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=find_dotenv(),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    app_name: str = Field(
        alias="APP_NAME",
        default="Memento-S",
        description="Application display name for TUI title bar",
    )
    context_compress_threshold: int = Field(
        alias="CONTEXT_COMPRESS_THRESHOLD",
        default=60000,
        description="Token count threshold that triggers automatic context compression",
    )
    context_max_tokens: int = Field(
        alias="CONTEXT_MAX_TOKENS",
        default=80000,
        description="Maximum total context tokens before refusing new input",
    )
    summary_max_tokens: int = Field(
        alias="SUMMARY_MAX_TOKENS",
        default=2000,
        description="Max tokens for compressed conversation summary",
    )

    llm_api: str = Field(
        alias="LLM_API",
        default="anthropic",
        description=" provider: anthropic, openai, google, openrouter, ollama, ...",
    )
    llm_model: str = Field(
        alias="LLM_MODEL",
        default="claude-3-5-sonnet-20241022",
        description=" gpt-4oclaude-3-5-sonnetgemini-1.5-flashopenrouter/...",
    )
    llm_api_key: str | None = Field(
        alias="LLM_API_KEY",
        default=None,
        description=" API Key LLM_API  KEY",
    )
    llm_base_url: str | None = Field(
        alias="LLM_BASE_URL",
        default=None,
        description=" Base URL LLM_API  BASE",
    )
    llm_max_tokens: int = Field(alias="LLM_MAX_TOKENS", default=4096)
    llm_temperature: float = Field(alias="LLM_TEMPERATURE", default=0.7)
    llm_timeout: int = Field(alias="LLM_TIMEOUT", default=120)

    agent_max_iterations: int = Field(
        alias="AGENT_MAX_ITERATIONS",
        default=100,
        description="ReAct  LLM+tool ",
    )
    log_level: str = Field(
        alias="LOG_LEVEL",
        default="ERROR",
        description=": DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )

    openai_api_key: str | None = Field(alias="OPENAI_API_KEY", default=None)
    openai_base_url: str | None = Field(alias="OPENAI_API_BASE", default=None)

    anthropic_api_key: str | None = Field(alias="ANTHROPIC_API_KEY", default=None)

    google_api_key: str | None = Field(alias="GOOGLE_API_KEY", default=None)

    openrouter_api_key: str | None = Field(alias="OPENROUTER_API_KEY", default=None)
    openrouter_base_url: str | None = Field(alias="OPENROUTER_BASE_URL", default=None)
    openrouter_site_url: str = Field(alias="OPENROUTER_SITE_URL", default="")
    openrouter_app_name: str = Field(alias="OPENROUTER_APP_NAME", default="")

    skills_catalog_path: str = Field(
        alias="SKILLS_CATALOG_PATH",
        default="router_data/skills_catalog.jsonl",
        description="skills_catalog.jsonl  project_root ",
    )
    github_token: str = Field(alias="GITHUB_TOKEN", default="")
    retrieval_top_k: int = Field(alias="RETRIEVAL_TOP_K", default=5)
    embedding_model: str = Field(alias="EMBEDDING_MODEL", default="auto")
    embedding_api_key: str = Field(alias="EMBEDDING_API_KEY", default="")
    embedding_base_url: str = Field(alias="EMBEDDING_BASE_URL", default="")
    retrieval_min_score: float = Field(alias="RETRIEVAL_MIN_SCORE", default=0.012)
    reranker_model: str = Field(alias="RERANKER_MODEL", default="auto")
    reranker_enabled: bool = Field(alias="RERANKER_ENABLED", default=True)
    reranker_min_score: float = Field(alias="RERANKER_MIN_SCORE", default=0.001)
    qwen3_tokenizer_path: str = Field(alias="QWEN3_TOKENIZER_PATH", default="")
    qwen3_model_path: str = Field(alias="QWEN3_MODEL_PATH", default="")
    execution_timeout_sec: int = Field(alias="EXECUTION_TIMEOUT_SEC", default=30)
    max_reflection_retries: int = Field(alias="MAX_REFLECTION_RETRIES", default=3)
    sandbox_provider: Literal["local", "e2b", "modal"] = Field(
        alias="SANDBOX_PROVIDER", default="local"
    )
    e2b_api_key: str = Field(alias="E2B_API_KEY", default="")
    resolve_strategy: Literal["local_only", "local_first", "always_search"] = Field(
        alias="RESOLVE_STRATEGY", default="local_first"
    )
    skill_download_method: Literal["github_api", "npx", "auto"] = Field(
        alias="SKILL_DOWNLOAD_METHOD", default="auto"
    )

    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent,
        description="memento_s ",
    )
    workspace_dir: Path = Field(
        alias="WORKSPACE_DIR",
        default=Path("workspace"),
        description=" project_rootSessionManager  workspace ",
    )
    conversations_dir: Path = Field(
        alias="CONVERSATIONS_DIR",
        default=Path("conversations"),
        description=" JSONL  workspace_dir",
    )
    workspace_root: str = Field(alias="WORKSPACE_ROOT", default="")

    def _resolve_path(self, path: str | Path) -> Path:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = self.project_root / p
        return p.resolve()

    @property
    def workspace_path(self) -> Path:
        if self.workspace_root:
            return self._resolve_path(self.workspace_root)
        return self._resolve_path(self.workspace_dir)

    @property
    def workspace(self) -> Path:
        return self.workspace_path

    def setup_workspace(self) -> None:
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        self.conversations_path.mkdir(parents=True, exist_ok=True)

    @property
    def conversations_path(self) -> Path:
        return self.workspace_path / self.conversations_dir

    @property
    def data_directory(self) -> Path:
        d = self.workspace_path / "data"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def skills_directory(self) -> Path:
        return self.workspace_path / "skills"

    @property
    def chroma_directory(self) -> Path:
        return self.data_directory / "chroma"

    @property
    def qwen3_tokenizer_path_resolved(self) -> Path | None:
        if not self.qwen3_tokenizer_path:
            return None
        return self._resolve_path(self.qwen3_tokenizer_path)

    @property
    def qwen3_model_path_resolved(self) -> Path | None:
        if not self.qwen3_model_path:
            return None
        return self._resolve_path(self.qwen3_model_path)

    @model_validator(mode="after")
    def _normalize(self) -> "Settings":
        self.llm_api = (self.llm_api or "anthropic").lower()
        return self

    def resolve_llm_api_key(self) -> str | None:
        if self.llm_api_key:
            return self.llm_api_key
        p = self.llm_api
        if p == "openai":
            return self.openai_api_key
        if p in ("anthropic", "claude"):
            return self.anthropic_api_key
        if p == "google":
            return self.google_api_key
        if p == "openrouter":
            return self.openrouter_api_key
        return None

    def resolve_llm_base_url(self) -> str | None:
        base = self.llm_base_url
        if not base:
            p = self.llm_api
            if p == "openai":
                base = self.openai_base_url
            elif p == "openrouter":
                base = self.openrouter_base_url
        if not base:
            return None
        if self.llm_api == "openrouter":
            base = base.rstrip("/")
            if base.endswith("/api") and not base.endswith("/api/v1"):
                base = base + "/v1"
        return base


g_settings = Settings()
