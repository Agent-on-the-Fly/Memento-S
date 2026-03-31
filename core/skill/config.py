"""Skill 模块配置定义

提供 Skill 模块所需的所有配置，解耦与 middleware.config 的直接依赖。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SkillConfig:
    """Skill 模块配置 - 不可变配置对象

    通过 from_global_config() 从全局配置创建，
    或手动构造用于测试。
    """

    # === 路径配置 ===
    skills_dir: Path
    builtin_skills_dir: Path
    workspace_dir: Path
    db_path: Path

    # === Embedding 配置 ===
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536  # text-embedding-3-small 默认维度

    # === 云端配置 ===
    cloud_catalog_url: str | None = None

    # === 召回配置 ===
    retrieval_top_k: int = 5

    # === 执行配置 ===
    pip_install_timeout: int = 120
    max_attempts: int = 3
    same_signature_limit: int = 2

    # === 安全配置 ===
    path_validation_enabled: bool = True

    @classmethod
    def from_global_config(cls) -> "SkillConfig":
        """从全局 g_config 创建配置

        这是生产环境的默认创建方式。
        """
        from middleware.config import g_config

        return cls(
            skills_dir=g_config.get_skills_path(),
            builtin_skills_dir=g_config.get_builtin_skills_path(),
            workspace_dir=Path(g_config.paths.workspace_dir),
            db_path=g_config.get_db_path(),
            embedding_base_url=g_config.skills.retrieval.embedding_base_url,
            embedding_api_key=g_config.skills.retrieval.embedding_api_key,
            embedding_model=g_config.skills.retrieval.embedding_model,
            embedding_dimension=g_config.skills.retrieval.embedding_dimension,
            cloud_catalog_url=g_config.skills.cloud_catalog_url,
            retrieval_top_k=g_config.skills.retrieval.top_k,
            pip_install_timeout=g_config.skills.execution.pip_install_timeout_sec,
            max_attempts=g_config.skills.execution.max_attempts,
            same_signature_limit=g_config.skills.execution.same_signature_limit,
            path_validation_enabled=g_config.paths.path_validation_enabled,
        )
