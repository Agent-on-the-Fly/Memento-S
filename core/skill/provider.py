"""SkillProvider — SkillGateway 的默认实现。"""

from __future__ import annotations

import asyncio
from typing import Any

from core.skill.execution.analyzer.dependencies import check_missing_dependencies
from core.skill.gateway import (
    DEFAULT_SKILL_PARAMS,
    SkillErrorCode,
    SkillExecOptions,
    SkillExecutionResponse,
    SkillGateway,
    SkillGovernanceMeta,
    SkillManifest,
    SkillStatus,
)
from core.skill.retrieval.multi_recall import RecallCandidate
from core.skill.schema import Skill, ExecutionMode
from middleware.config import g_config
from utils.logger import get_logger

logger = get_logger(__name__)


class SkillProvider(SkillGateway):
    """Skill 契约实现：目录层、运行时层、治理层。

    使用工厂方法 create_default() 创建生产实例。
    依赖在首次使用时懒加载，也可通过参数注入（用于测试）。
    """

    def __init__(
        self,
        store,
        indexer=None,
        multi_recall=None,
        cloud_catalog=None,
        executor=None,
        llm=None,
    ):
        """初始化 Provider。

        Args:
            store: SkillStore 实例（必需）
            indexer: 可选的 SkillIndexer
            multi_recall: 可选的 MultiRecall
            cloud_catalog: 可选的 RemoteCloudCatalog
            executor: 可选的 SkillExecutor
            llm: 可选的 LLM 客户端

        注意：生产环境使用 create_default() 工厂方法，
              测试环境可手动传入 mock 依赖。
        """
        self._store = store
        self._indexer = indexer
        self._multi_recall = multi_recall
        self._cloud_catalog = cloud_catalog
        self._executor = executor
        self._llm = llm
        self._download_locks: dict[str, asyncio.Lock] = {}

    @classmethod
    async def create_default(cls):
        """工厂方法：使用默认配置创建 SkillProvider。

        前提：系统已初始化（DB、配置等），且 skill 同步已由 bootstrap 完成

        Returns:
            SkillProvider 实例
        """
        from middleware.storage.services.skill_service import SkillService
        from middleware.llm import LLMClient
        from core.skill.store import SkillStore
        from core.skill.retrieval.embedding_recall import EmbeddingRecall
        from core.skill.retrieval.indexer import SkillIndexer
        from core.skill.retrieval.multi_recall import MultiRecall
        from core.skill.retrieval.remote_catalog import RemoteCloudCatalog
        from core.skill.execution import SkillExecutor
        from core.skill.embedding import EmbeddingClient, EmbeddingClientConfig

        # 1. 创建基础客户端
        llm = LLMClient()

        # 2. 创建 embedding client（如果有配置）
        embedding_client = None
        embedding_url = g_config.skills.retrieval.embedding_base_url
        if embedding_url:
            embedding_client = EmbeddingClient(
                config=EmbeddingClientConfig(
                    base_url=embedding_url,
                    api_key=g_config.skills.retrieval.embedding_api_key or "",
                    model=g_config.skills.retrieval.embedding_model,
                )
            )

        # 3. 创建基础设施（DB 已初始化，skill 同步已由 bootstrap 完成）
        skill_service = SkillService()
        store = SkillStore(
            skill_service=skill_service,
            embedding_client=embedding_client,
        )

        # 4. 创建 indexer
        embedding_recall = EmbeddingRecall(
            db_path=g_config.get_db_path(),
            embedding_client=embedding_client,
        )
        indexer = SkillIndexer(embedding_recall=embedding_recall)

        # 5. 创建 cloud catalog
        cloud_catalog = None
        cloud_url = g_config.skills.cloud_catalog_url
        if cloud_url:
            cloud_catalog = RemoteCloudCatalog(base_url=cloud_url)

        # 6. 创建 multi_recall 和 executor
        multi_recall = MultiRecall(cloud_catalog=cloud_catalog)
        executor = SkillExecutor(llm=llm)

        # 7. 创建 Provider
        provider = cls(
            store=store,
            indexer=indexer,
            multi_recall=multi_recall,
            cloud_catalog=cloud_catalog,
            executor=executor,
            llm=llm,
        )

        logger.info(
            "SkillProvider created with {} local skills",
            len(store.local_cache),
        )

        return provider

    # ---------------- Catalog ----------------

    def discover(self) -> list[SkillManifest]:
        """Discover all available skills.

        Exceptions are caught and logged, returning empty list for consistency.
        """
        try:
            return [
                self._to_manifest(s, source="local")
                for s in self._store.local_cache.values()
            ]
        except Exception as e:
            logger.warning("Skill discover failed: {}", e)
            return []

    async def search(
        self, query: str, k: int = 5, cloud_only: bool = False
    ) -> list[SkillManifest]:
        """Search skills by query.

        Args:
            query: Search query
            k: Number of results to return
            cloud_only: If True, skip local embedding search and only return cloud results
        """
        try:
            # cloud_only 模式：跳过本地 embedding 搜索
            if cloud_only:
                if not self._cloud_catalog:
                    return []
                try:
                    cloud_results = self._cloud_catalog.search(query, k=k)
                    return [
                        SkillManifest(
                            name=info.name,
                            description=info.description or "",
                            parameters=DEFAULT_SKILL_PARAMS,
                            execution_mode=ExecutionMode.KNOWLEDGE,
                            dependencies=[],
                            governance=SkillGovernanceMeta(source="cloud"),
                        )
                        for info in cloud_results
                    ]
                except Exception as e:
                    logger.debug("Cloud search failed: {}", e)
                    return []

            cloud_top_k = g_config.skills.retrieval.top_k
            if self._multi_recall is None:
                return []
            candidates = await self._multi_recall.recall(
                query,
                local_cache=self._store.local_cache,
                cloud_k=max(cloud_top_k, k),
            )

            reranked = self._rerank_candidates(candidates)

            result: list[SkillManifest] = []
            for c in reranked:
                if c.source == "local" and c.skill:
                    result.append(self._to_manifest(c.skill, source="local"))
                else:
                    result.append(
                        SkillManifest(
                            name=c.name,
                            description=c.description or "",
                            parameters=DEFAULT_SKILL_PARAMS,
                            execution_mode=ExecutionMode.KNOWLEDGE,
                            dependencies=[],
                            governance=SkillGovernanceMeta(source="cloud"),
                        )
                    )

            return result
        except Exception as e:
            logger.warning("Skill search failed for query '{}': {}", query, e)
            return []

    # ---------------- Runtime ----------------

    async def execute(
        self,
        skill_name: Skill | str,
        params: dict[str, Any],
        options: SkillExecOptions | None = None,
    ) -> SkillExecutionResponse:
        resolved_skill_name = (
            skill_name.name if isinstance(skill_name, Skill) else skill_name
        )

        skill = await self._ensure_local_skill(resolved_skill_name)
        if skill is None:
            return SkillExecutionResponse(
                ok=False,
                status=SkillStatus.FAILED,
                error_code=SkillErrorCode.SKILL_NOT_FOUND,
                summary=f"Skill '{resolved_skill_name}' not found",
                skill_name=resolved_skill_name,
            )

        missing_deps = check_missing_dependencies(skill.dependencies)
        if missing_deps:
            logger.info(
                "Auto-installing missing dependencies for '{}': {}",
                resolved_skill_name,
                missing_deps,
            )
            try:
                from core.skill.execution.sandbox import get_sandbox

                sandbox = get_sandbox()
                pip_timeout = g_config.skills.execution.pip_install_timeout_sec
                success, error_msg = sandbox.install_python_deps(
                    missing_deps, timeout=pip_timeout
                )
                if not success:
                    install_hint = "uv pip install " + " ".join(missing_deps)
                    return SkillExecutionResponse(
                        ok=False,
                        status=SkillStatus.BLOCKED,
                        error_code=SkillErrorCode.DEPENDENCY_MISSING,
                        summary=f"Failed to install dependencies: {error_msg}. Run: {install_hint}",
                        skill_name=resolved_skill_name,
                        diagnostics={
                            "missing_dependencies": missing_deps,
                            "install_hint": install_hint,
                            "error_type": "dependency_error",
                        },
                    )
                logger.info("Dependencies installed successfully: {}", missing_deps)
            except Exception as e:
                install_hint = "uv pip install " + " ".join(missing_deps)
                return SkillExecutionResponse(
                    ok=False,
                    status=SkillStatus.BLOCKED,
                    error_code=SkillErrorCode.DEPENDENCY_MISSING,
                    summary=f"Failed to install dependencies: {e}. Run: {install_hint}",
                    skill_name=resolved_skill_name,
                    diagnostics={
                        "missing_dependencies": missing_deps,
                        "install_hint": install_hint,
                        "error_type": "dependency_error",
                    },
                )

        from core.skill.execution.utils.skill_keys_checker import check_skill_keys

        keys_ok, missing_keys = check_skill_keys(skill)
        if not keys_ok:
            return SkillExecutionResponse(
                ok=False,
                status=SkillStatus.BLOCKED,
                error_code=SkillErrorCode.KEY_MISSING,
                summary=f"Missing API keys: {', '.join(missing_keys)}",
                skill_name=resolved_skill_name,
            )

        try:
            # 从 params 中提取 query（如果存在），否则转为字符串
            query = params.get("request", str(params))
            if self._executor is None:
                from core.skill.execution import SkillExecutor

                self._executor = SkillExecutor(llm=self._llm)
            exec_result, generated_code = await self._executor.execute(
                skill,
                query=query,
                params=params,
            )
            if exec_result.success:
                return SkillExecutionResponse(
                    ok=True,
                    status=SkillStatus.SUCCESS,
                    summary="skill executed",
                    output=exec_result.result,
                    outputs={
                        "generated_code": generated_code or "",
                        "operation_results": exec_result.operation_results or [],
                    },
                    artifacts=exec_result.artifacts or [],
                    diagnostics={"track": self._execution_mode(skill)},
                    skill_name=skill.name,
                )

            diagnostics = {
                "error_type": exec_result.error_type.value
                if exec_result.error_type
                else None,
                "error_detail": exec_result.error_detail or None,
            }
            return SkillExecutionResponse(
                ok=False,
                status=SkillStatus.FAILED,
                error_code=SkillErrorCode.RUNTIME_ERROR,
                summary=exec_result.error or "Skill execution failed",
                output=exec_result.result,
                outputs={"operation_results": exec_result.operation_results or []},
                artifacts=exec_result.artifacts or [],
                diagnostics=diagnostics,
                skill_name=skill.name,
            )
        except Exception as e:
            logger.warning("Skill execution failed for '{}': {}", skill_name, e)
            return SkillExecutionResponse(
                ok=False,
                status=SkillStatus.FAILED,
                error_code=SkillErrorCode.INTERNAL_ERROR,
                summary=str(e),
                skill_name=str(skill_name),
            )

    # ---------------- Internal ----------------

    def _rerank_candidates(
        self,
        candidates: list[RecallCandidate],
    ) -> list[RecallCandidate]:
        """本地优先，云端按 score 降序"""

        def rank_key(c: RecallCandidate) -> tuple[int, float]:
            tier = 0 if c.source == "local" else 1
            return (tier, -float(c.score or 0.0))

        return sorted(candidates, key=rank_key)

    async def _ensure_local_skill(self, skill_name: str) -> Skill | None:
        skill = self._find_local_skill(skill_name)
        if skill is not None:
            return skill

        if not self._cloud_catalog:
            return None

        lock = self._download_locks.setdefault(skill_name, asyncio.Lock())
        async with lock:
            skill = self._find_local_skill(skill_name)
            if skill is not None:
                return skill

            downloaded = await self._download_cloud_skill(skill_name)
            if downloaded is None:
                return None
            try:
                await self._add_skill(downloaded)
            except Exception as e:
                logger.warning(
                    "Cloud skill '{}' downloaded but failed to add into store: {}",
                    downloaded.name,
                    e,
                )
                return None

            logger.info(
                "Cloud skill '{}' downloaded and added to library", downloaded.name
            )
            return downloaded

    def _find_local_skill(self, skill_name: str) -> Skill | None:
        """查找本地 skill（支持多种命名格式）。"""
        return self._store.find_by_name(skill_name)

    async def _download_cloud_skill(self, skill_name: str) -> Skill | None:
        """下载云端 skill 并加载到本地存储。"""
        if not self._cloud_catalog:
            return None

        try:
            from pathlib import Path

            local_path = self._cloud_catalog.download(
                skill_name,
                g_config.get_skills_path(),
            )
            if not local_path:
                return None

            skill = await self._store.load_from_path(Path(local_path))
            return skill
        except Exception as e:
            logger.warning("Failed to download cloud skill '{}': {}", skill_name, e)
            return None

    @staticmethod
    def _execution_mode(skill: Skill) -> ExecutionMode:
        if skill.execution_mode:
            return skill.execution_mode
        return ExecutionMode.PLAYBOOK if skill.is_playbook else ExecutionMode.KNOWLEDGE

    @staticmethod
    def _to_manifest(skill: Skill, source: str = "local") -> SkillManifest:
        exec_mode_str = SkillProvider._execution_mode(skill)
        # 转换为 ExecutionMode enum
        exec_mode = (
            exec_mode_str
            if isinstance(exec_mode_str, ExecutionMode)
            else ExecutionMode(exec_mode_str)
        )
        # 使用 skill 自描述的 parameters，或使用默认值
        parameters = skill.parameters if skill.parameters else DEFAULT_SKILL_PARAMS
        return SkillManifest(
            name=skill.name,
            description=skill.description or "",
            parameters=parameters,
            execution_mode=exec_mode,
            dependencies=skill.dependencies or [],
            governance=SkillGovernanceMeta(
                source="cloud" if source == "cloud" else "local",
            ),
        )

    # ---------------- Storage Management (from SkillManager) ----------------

    def list_skills(self) -> list[Skill]:
        """List all local skills."""
        return list(self._store.local_cache.values())

    async def _add_skill(self, skill: Skill) -> None:
        """Add skill to store and index."""
        await self._store.add_skill(skill)
        if self._indexer and await self._indexer.ensure_ready():
            await self._indexer.index(skill)

    async def remove_skill(self, skill_name: str) -> bool:
        """Remove skill from store and index."""
        removed = await self._store.remove_skill(skill_name)
        if removed and self._indexer:
            self._indexer.delete(skill_name)
        return removed

    async def refresh_from_disk(self) -> int:
        """Refresh skills from disk and re-index."""
        added = await self._store.refresh_from_disk()
        if added and self._indexer and await self._indexer.ensure_ready():
            await self._indexer.index_batch(self._store.local_cache.values())
        return added

    async def sync_all_to_db(self) -> None:
        """Sync all skills to DB."""
        await self._store.sync_all_to_db()
