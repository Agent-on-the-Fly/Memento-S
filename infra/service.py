# SPDX-License-Identifier: Apache-2.0
"""InfraService — core/ 上层对 infra/ 的唯一依赖入口。

集中了 ContextProvider, CompactorProvider, ArtifactProvider
的创建和访问，上层不需要知道 infra/ 的内部结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from infra.context.base import ContextProvider
from infra.context.factory import ContextFactoryConfig, create_context
from infra.context.providers.artifact import ArtifactProvider
from infra.context.providers.compactor import CompactorProvider
from infra.memory.impl.session_memory import SessionMemory
from infra.memory.impl.long_term_memory import LongTermMemory
from middleware.config import g_config
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class InfraContextConfig:
    """统一的上下文配置（对应 infra/context/factory.py 的参数子集）。"""

    context_max_tokens: int = 0
    history_budget_ratio: float = 0.5
    summary_ratio: float = 0.15
    persist_ratio: float = 0.15
    extract_ratio: float = 0.05
    preview_ratio: float = 0.005
    slim_ratio: float = 0.003
    microcompact_keep_recent: int = 5
    microcompact_compactable_tools: list[str] = field(
        default_factory=lambda: ["execute_skill", "search_skill", "read_file", "bash"]
    )
    emergency_keep_tail: int = 6
    max_compact_failures: int = 3
    sm_compact_min_ratio: float = 0.02
    sm_compact_max_ratio: float = 0.08
    breaker_cooldown_s: float = 60.0
    sm_llm_update_interval: int = 5
    embedding_enabled: bool = False
    history_load_limit: int = 20
    recent_rounds_keep: int = 3

    @classmethod
    def from_core_context_config(cls, cfg: Any) -> "InfraContextConfig":
        """从 core/context/schemas.py 的 ContextConfig 转换。

        Args:
            cfg: core.context.schemas.ContextConfig 实例

        Returns:
            InfraContextConfig 实例
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

    def to_factory_config(
        self,
        session_id: str,
        *,
        session_dir: Path | str | None = None,
        data_dir: Path | str | None = None,
        context_dir: Path | str | None = None,
        model: str = "",
        skill_gateway: Any | None = None,
        history_loader: Any | None = None,
    ) -> ContextFactoryConfig:
        """将 InfraContextConfig 转换为 ContextFactoryConfig。

        消除字段重复，避免两边配置不一致。
        """
        return ContextFactoryConfig(
            session_id=session_id,
            session_dir=session_dir,
            data_dir=data_dir,
            context_dir=context_dir,
            model=model,
            context_max_tokens=self.context_max_tokens,
            history_budget_ratio=self.history_budget_ratio,
            summary_ratio=self.summary_ratio,
            persist_ratio=self.persist_ratio,
            extract_ratio=self.extract_ratio,
            preview_ratio=self.preview_ratio,
            slim_ratio=self.slim_ratio,
            microcompact_keep_recent=self.microcompact_keep_recent,
            microcompact_compactable_tools=self.microcompact_compactable_tools,
            emergency_keep_tail=self.emergency_keep_tail,
            max_compact_failures=self.max_compact_failures,
            sm_compact_min_ratio=self.sm_compact_min_ratio,
            sm_compact_max_ratio=self.sm_compact_max_ratio,
            breaker_cooldown_s=self.breaker_cooldown_s,
            sm_llm_update_interval=self.sm_llm_update_interval,
            embedding_enabled=self.embedding_enabled,
            history_load_limit=self.history_load_limit,
            recent_rounds_keep=self.recent_rounds_keep,
            skill_gateway=skill_gateway,
            history_loader=history_loader,
        )


@dataclass
class InfraService:
    """core/ 上层对 infra/ 的唯一依赖入口。

    集中了 ContextProvider、CompactorProvider、ArtifactProvider
    的创建和访问，上层不需要知道 infra/ 的内部结构。

    使用方式::

        infra = InfraService(
            session_id="session_001",
            context_config=InfraContextConfig(),
        )
        session_bundle = SessionBundle(session_ctx, infra)

        # 通过 infra 访问所有能力
        infra.context.load_and_assemble(...)
        infra.session_memory.append_worklog_entry(...)
        infra.compactor.prepare_for_api(...)
        infra.artifact_provider.has_artifact(...)
    """

    def __init__(
        self,
        session_id: str,
        *,
        session_dir: Path | str | None = None,
        data_dir: Path | str | None = None,
        context_dir: Path | str | None = None,
        model: str = "",
        context_config: InfraContextConfig | None = None,
        sm_llm_update_interval: int = 5,
        skill_gateway: Any | None = None,
        history_loader: Any | None = None,
    ) -> None:
        if isinstance(session_dir, str):
            session_dir = Path(session_dir)
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        if isinstance(context_dir, str):
            context_dir = Path(context_dir)

        if not g_config.is_loaded():
            g_config.load()

        if session_dir is None:
            session_dir = g_config.paths.context_dir
        if data_dir is None:
            data_dir = g_config.paths.context_dir
        context_dir = g_config.paths.context_dir

        ctx_cfg = context_config or InfraContextConfig()

        ctx_factory = ctx_cfg.to_factory_config(
            session_id=session_id,
            session_dir=session_dir,
            data_dir=data_dir,
            context_dir=context_dir,
            model=model,
            skill_gateway=skill_gateway,
            history_loader=history_loader,
        )
        self._context = create_context(ctx_factory)

        # 路径保存（用于 RecallEngine）
        self._session_dir = session_dir
        self._context_dir = context_dir

        # L1 Session Memory（由 FileContextProvider 持有）
        self._session_memory = self._context.session_memory

        # L2 Long Memory（由 FileContextProvider 创建）
        self._long_memory = self._context.context_memory

        # Recall Engine（lazy，注入依赖和路径）
        self._recall_engine = None

        # Memory Consolidation Engine（由 InfraService 持有，供 Agent/Dream 触发）
        self._init_memory_engine()

        logger.info(
            "[InfraService] initialized: session_dir={}, data_dir={}, model={}, "
            "session_memory={}, long_memory={}, memory_engine={}",
            session_dir, data_dir, model,
            type(self._session_memory).__name__,
            type(self._long_memory).__name__,
            type(self._memory_engine).__name__ if self._memory_engine else "None",
        )

    @property
    def context(self) -> ContextProvider:
        """统一上下文接口。"""
        return self._context

    @property
    def session_memory(self) -> SessionMemory:
        """L1 Session Memory。始终存在。"""
        return self._session_memory

    @property
    def context_memory(self) -> LongTermMemory:
        """L2 Long Memory（跨 session）。始终存在。"""
        return self._long_memory

    @property
    def compactor(self) -> CompactorProvider:
        """预压缩流水线。"""
        return self._context.compactor

    @property
    def artifact_provider(self) -> ArtifactProvider:
        """Tool Result 持久化。"""
        return self._context.artifact_provider

    @property
    def context_dir(self) -> Path:
        """Context 根目录（sessions/ 和 memory/ 的父目录）。"""
        return self._context_dir

    @property
    def session_dir(self) -> Path:
        """当前 session 的根目录。"""
        return self._session_dir

    @property
    def recall_engine(self) -> "RecallEngine":
        """统一 Recall Engine（lazy 初始化）。"""
        if self._recall_engine is None:
            from .memory.recall_engine import RecallEngine
            self._recall_engine = RecallEngine(
                session_memory=self._session_memory,
                long_term_memory=self._long_memory,
                artifact_provider=self._context.artifact_provider,
                context_dir=self._context_dir,
                current_session_dir=self._session_dir,
            )
        return self._recall_engine

    @property
    def memory_engine(self) -> "MemoryConsolidationEngine | None":
        """Memory Consolidation Engine（跨 session 自动整合）。"""
        return self._memory_engine

    def _init_memory_engine(self) -> None:
        """初始化 MemoryConsolidationEngine（如果启用）。"""
        from pathlib import Path

        cfg = g_config.load()
        mem_cfg = cfg.memory

        if not mem_cfg.enabled:
            self._memory_engine = None
            return

        context_dir: Path | None = g_config.paths.context_dir
        if context_dir is None:
            self._memory_engine = None
            return

        memory_dir = context_dir / "memory"

        from .memory.consolidation import MemoryConsolidationEngine

        self._memory_engine = MemoryConsolidationEngine(
            memory=memory_dir,
            config=mem_cfg,
        )
