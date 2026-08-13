# SPDX-License-Identifier: Apache-2.0
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

    # === 云端配置 ===
    cloud_catalog_url: str | None = None

    # === 召回配置 ===
    retrieval_top_k: int = 5
    retrieval_embedding_model: str | None = None
    retrieval_router_backend: str = "keyword"
    retrieval_embedding_api_key: str | None = None
    retrieval_embedding_base_url: str | None = None
    qwen_tokenizer_path: str = "Qwen/Qwen3-Embedding-0.6B"
    qwen_model_path: str = ""
    qwen_device: str = "auto"
    qwen_max_length: int = 8192
    qwen_batch_size: int = 32
    qwen_timeout_sec: float = 30.0
    qwen_query_instruction: str = (
        "Instruct: Given a user query, retrieve relevant skill descriptions "
        "that match the query\nQuery:"
    )

    # === Read-Write skill evolution ===
    evolution_enabled: bool = False
    evolution_protected_skills: tuple[str, ...] = (
        "filesystem",
        "skill-creator",
        "uv-pip-install",
        "web-search",
    )
    evolution_min_attribution_confidence: float = 0.5
    evolution_max_updates_per_step: int = 1
    evolution_candidate_attempts: int = 1
    evolution_utility_discovery_enabled: bool = True
    evolution_utility_discovery_threshold: float = 0.2
    evolution_utility_min_samples: int = 3
    evolution_test_timeout_sec: int = 180
    evolution_synthetic_test_enabled: bool = True
    evolution_keep_failed_candidate: bool = True
    evolution_max_prompt_chars: int = 60000

    # === 执行配置 ===
    pip_install_timeout: int = 120
    max_attempts: int = 3
    same_signature_limit: int = 2

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
            cloud_catalog_url=g_config.skills.cloud_catalog_url,
            retrieval_top_k=g_config.skills.retrieval.top_k,
            retrieval_embedding_model=g_config.skills.retrieval.embedding_model,
            retrieval_router_backend=g_config.skills.retrieval.router_backend,
            retrieval_embedding_api_key=g_config.skills.retrieval.embedding_api_key,
            retrieval_embedding_base_url=g_config.skills.retrieval.embedding_base_url,
            qwen_tokenizer_path=g_config.skills.retrieval.qwen_tokenizer_path,
            qwen_model_path=g_config.skills.retrieval.qwen_model_path,
            qwen_device=g_config.skills.retrieval.qwen_device,
            qwen_max_length=g_config.skills.retrieval.qwen_max_length,
            qwen_batch_size=g_config.skills.retrieval.qwen_batch_size,
            qwen_timeout_sec=g_config.skills.retrieval.qwen_timeout_sec,
            qwen_query_instruction=g_config.skills.retrieval.qwen_query_instruction,
            evolution_enabled=g_config.skills.evolution.enabled,
            evolution_protected_skills=tuple(
                g_config.skills.evolution.protected_skills
            ),
            evolution_min_attribution_confidence=(
                g_config.skills.evolution.min_attribution_confidence
            ),
            evolution_max_updates_per_step=(
                g_config.skills.evolution.max_updates_per_step
            ),
            evolution_candidate_attempts=(g_config.skills.evolution.candidate_attempts),
            evolution_utility_discovery_enabled=(
                g_config.skills.evolution.utility_discovery_enabled
            ),
            evolution_utility_discovery_threshold=(
                g_config.skills.evolution.utility_discovery_threshold
            ),
            evolution_utility_min_samples=(
                g_config.skills.evolution.utility_min_samples
            ),
            evolution_test_timeout_sec=g_config.skills.evolution.test_timeout_sec,
            evolution_synthetic_test_enabled=(
                g_config.skills.evolution.synthetic_test_enabled
            ),
            evolution_keep_failed_candidate=(
                g_config.skills.evolution.keep_failed_candidate
            ),
            evolution_max_prompt_chars=g_config.skills.evolution.max_prompt_chars,
            pip_install_timeout=g_config.skills.execution.pip_install_timeout_sec,
            max_attempts=g_config.skills.execution.max_attempts,
            same_signature_limit=g_config.skills.execution.same_signature_limit,
        )
