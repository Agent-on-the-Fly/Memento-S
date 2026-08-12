# SPDX-License-Identifier: Apache-2.0
"""ContextFactory — ContextProvider 工厂函数。

无 core/ 依赖。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from infra.context.base import ContextProvider
from infra.context.impl.file_context import FileContextProvider
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ContextFactoryConfig:
    """ContextFactory 配置。"""

    # ── Session ─────────────────────────────────────────────────────
    session_id: str = ""
    """Session 标识。"""

    session_dir: Path | str | None = None
    """Session 目录路径（默认从 session_id 生成）。"""

    data_dir: Path | str | None = None
    """数据目录路径（默认 session_dir/../data）。"""

    context_dir: Path | str | None = None
    """Context 根目录路径（默认从 session_dir 推导，session_dir.parent.parent）。"""

    model: str = ""

    # ── File Context 配置 ───────────────────────────────────────────
    history_load_limit: int = 20

    recent_rounds_keep: int = 3

    history_budget_ratio: float = 0.5

    summary_ratio: float = 0.15

    embedding_enabled: bool = False

    context_max_tokens: int = 0

    # ── Artifact Store ─────────────────────────────────────────────
    persist_ratio: float = 0.15

    extract_ratio: float = 0.05

    # ── Pipeline ──────────────────────────────────────────────────
    preview_ratio: float = 0.005

    slim_ratio: float = 0.003

    # ── Microcompact ──────────────────────────────────────────────
    microcompact_keep_recent: int = 5

    microcompact_compactable_tools: list[str] = field(
        default_factory=lambda: ["execute_skill", "search_skill", "read_file", "bash"]
    )

    # ── Budget Guard ──────────────────────────────────────────────
    emergency_keep_tail: int = 6

    max_compact_failures: int = 3

    sm_compact_min_ratio: float = 0.02

    sm_compact_max_ratio: float = 0.08

    breaker_cooldown_s: float = 60.0

    # ── L1 Session Memory ─────────────────────────────────────────
    sm_llm_update_interval: int = 5

    # ── 可选依赖 ─────────────────────────────────────────────────
    skill_gateway: Any | None = None

    history_loader: Any | None = None

    @classmethod
    def from_core_context_config(cls, cfg: Any) -> "ContextFactoryConfig":
        """从 core/context/schemas.py 的 ContextConfig 转换。

        Args:
            cfg: core.context.schemas.ContextConfig 实例

        Returns:
            ContextFactoryConfig 实例
        """
        return cls(
            context_max_tokens=0,
            history_budget_ratio=cfg.history_budget_ratio,
            summary_ratio=cfg.summary_ratio,
            persist_ratio=cfg.persist_ratio,
            extract_ratio=cfg.extract_ratio,
            preview_ratio=cfg.preview_ratio,
            slim_ratio=cfg.slim_ratio,
            microcompact_keep_recent=cfg.microcompact_keep_recent,
            microcompact_compactable_tools=list(cfg.microcompact_compactable_tools),
            emergency_keep_tail=cfg.emergency_keep_tail,
            max_compact_failures=cfg.max_compact_failures,
            sm_compact_min_ratio=cfg.sm_compact_min_ratio,
            sm_compact_max_ratio=cfg.sm_compact_max_ratio,
            breaker_cooldown_s=cfg.breaker_cooldown_s,
            sm_llm_update_interval=cfg.sm_llm_update_interval,
            embedding_enabled=cfg.embedding_enabled,
            history_load_limit=cfg.history_load_limit,
            recent_rounds_keep=cfg.recent_rounds_keep,
        )


def create_context(config: ContextFactoryConfig) -> ContextProvider:
    """工厂函数：创建 FileContextProvider 实例。

    Args:
        config: ContextFactoryConfig 配置，必须显式传入 session_dir 和 data_dir

    Returns:
        FileContextProvider 实例

    Raises:
        ValueError: 配置无效时
    """
    if not config.session_id:
        raise ValueError("session_id is required")

    session_dir: Path
    data_dir: Path

    if config.session_dir is None:
        raise ValueError("session_dir is required")
    session_dir = Path(config.session_dir) if isinstance(config.session_dir, str) else config.session_dir

    if config.data_dir is None:
        raise ValueError("data_dir is required")
    data_dir = Path(config.data_dir) if isinstance(config.data_dir, str) else config.data_dir

    ctx_dir: Path | None = None
    if config.context_dir is not None:
        ctx_dir = Path(config.context_dir) if isinstance(config.context_dir, str) else config.context_dir

    return FileContextProvider(
        session_dir=session_dir,
        data_dir=data_dir,
        context_dir=ctx_dir,
        model=config.model,
        context_max_tokens=config.context_max_tokens,
        history_load_limit=config.history_load_limit,
        recent_rounds_keep=config.recent_rounds_keep,
        history_budget_ratio=config.history_budget_ratio,
        summary_ratio=config.summary_ratio,
        persist_ratio=config.persist_ratio,
        extract_ratio=config.extract_ratio,
        preview_ratio=config.preview_ratio,
        slim_ratio=config.slim_ratio,
        microcompact_keep_recent=config.microcompact_keep_recent,
        microcompact_compactable_tools=config.microcompact_compactable_tools,
        emergency_keep_tail=config.emergency_keep_tail,
        max_compact_failures=config.max_compact_failures,
        sm_compact_min_ratio=config.sm_compact_min_ratio,
        sm_compact_max_ratio=config.sm_compact_max_ratio,
        breaker_cooldown_s=config.breaker_cooldown_s,
        sm_llm_update_interval=config.sm_llm_update_interval,
        embedding_enabled=config.embedding_enabled,
        history_loader=config.history_loader,
        skill_gateway=config.skill_gateway,
    )


__all__ = [
    "ContextFactoryConfig",
    "create_context",
]
