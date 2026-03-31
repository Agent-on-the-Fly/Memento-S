"""Agent-Skill 契约层：SkillGateway 实现。

DTO 定义在 schema.py 中，通过 core.skill 包导入。
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from middleware.llm import LLMClient
from utils.logger import get_logger

from .config import SkillConfig
from .downloader.factory import create_default_download_manager
from .execution import SkillExecutor
from .execution.policy.pre_execute import run_pre_execute_gate
from .retrieval import MultiRecall, RemoteRecall
from .retrieval.local_file_recall import LocalFileRecall
from .retrieval.schema import RecallCandidate
from .schema import (
    DEFAULT_SKILL_PARAMS,
    DiscoverStrategy,
    ExecutionMode,
    Skill,
    SkillErrorCode,
    SkillExecOptions,
    SkillExecutionResponse,
    SkillGovernanceMeta,
    SkillManifest,
    SkillStatus,
)
from .store import SkillStore

logger = get_logger(__name__)


class SkillGateway:
    """Skill 契约实现：目录层、运行时层、治理层。

    这是唯一的实现类，外部通过此接口与 Skill 系统交互。
    内部管理 SkillStore，生产环境通过 core.skill.init_skill_system() 创建。
    """

    def __init__(
        self,
        config: "SkillConfig",
        store: "SkillStore",
        multi_recall: MultiRecall | None = None,
        executor: SkillExecutor | None = None,
        llm: "LLMClient" | None = None,
    ):
        """初始化 SkillGateway。

        Args:
            config: SkillConfig 配置对象（必需）
            store: SkillStore 实例（必需，使用 SkillStore.from_config() 创建）
            multi_recall: 可选的 MultiRecall（内部包含 RemoteRecall 等策略）
            executor: 可选的 SkillExecutor
            llm: 可选的 LLM 客户端

        注意：生产环境使用 init_skill_system() 或 from_config() 工厂方法。
        """
        self._config = config
        self._store = store
        self._multi_recall = multi_recall
        self._executor = executor
        self._llm = llm
        self._download_locks: dict[str, asyncio.Lock] = {}

    @classmethod
    async def from_config(
        cls,
        config: "SkillConfig | None" = None,
    ) -> "SkillGateway":
        """异步工厂方法创建 SkillGateway。

        内部自动创建所有依赖（Store, MultiRecall, RemoteRecall, Executor, LLM）。

        Args:
            config: SkillConfig 配置，为 None 时自动从全局配置创建

        Returns:
            初始化好的 SkillGateway 实例
        """
        # 1. 创建/获取配置
        if config is None:
            config = SkillConfig.from_global_config()

        # 2. 异步创建 Store（内部自动处理 DB、embedding 等依赖）
        store = await SkillStore.from_config(config)

        # 2. 创建 LLM 客户端
        llm = LLMClient()

        # 3. 创建 Executor
        executor = SkillExecutor(config=config, llm=llm)

        # 4. 创建 MultiRecall（包含所有召回策略）
        multi_recall = MultiRecall.from_config(config)

        return cls(
            config=config,
            store=store,
            multi_recall=multi_recall,
            executor=executor,
            llm=llm,
        )

    @property
    def skill_store(self):
        return self._store

    async def discover(
        self,
        strategy: DiscoverStrategy | str = DiscoverStrategy.LOCAL_ONLY,
        query: str = "",
        k: int = 10,
    ) -> list[SkillManifest]:
        """Discover skills by strategy.

        Args:
            strategy: Discover strategy
                - DiscoverStrategy.LOCAL_ONLY: return all local skills from file store (default)
                - DiscoverStrategy.MULTI_RECALL: use multi-recall search path
            query: search query for multi_recall strategy
            k: max candidates for multi_recall strategy

        Returns:
            Skill manifest list. Returns [] on errors.
        """
        try:
            normalized_strategy = DiscoverStrategy(strategy)

            if normalized_strategy == DiscoverStrategy.LOCAL_ONLY:
                skills = await self._store.list_all_skills()
                manifests = [
                    self._to_manifest(skill, source="local")
                    for skill in skills.values()
                ]
                manifests.sort(key=lambda m: m.name)
                return manifests

            if normalized_strategy == DiscoverStrategy.MULTI_RECALL:
                if self._multi_recall is None:
                    logger.warning(
                        "discover(strategy=multi_recall) called but multi_recall is not initialized"
                    )
                    return []
                candidates = await self._multi_recall.search(query, k=max(1, int(k)))
                return [self._candidate_to_manifest(c) for c in candidates]

            return []
        except ValueError:
            logger.warning("discover got unknown strategy: {}", strategy)
            return []
        except Exception as e:
            logger.warning("Skill discover failed(strategy={}): {}", strategy, e)
            return []

    async def search(
        self, query: str, k: int = 5, cloud_only: bool = False
    ) -> list[SkillManifest]:
        """Search skills by query.

        Args:
            query: Search query
            k: Number of results to return
            cloud_only: If True, skip local embedding search and only return remote results
        """
        try:
            if self._multi_recall is None:
                return []

            # 统一使用 multi_recall，通过 source_filter 控制是否只搜索云端
            source_filter = "remote" if cloud_only else None
            candidates = await self._multi_recall.search(
                query,
                k=k,
                source_filter=source_filter,
            )

            reranked = self._rerank_candidates(candidates)

            return [self._candidate_to_manifest(c) for c in reranked]
        except Exception as e:
            logger.warning("Skill search failed for query '{}': {}", query, e)
            return []

    # ---------------- Runtime ----------------

    async def execute(
        self,
        skill_name: Skill | str,
        params: dict[str, Any],
        options: SkillExecOptions | None = None,
        session_id: str | None = None,
        on_step: Any | None = None,
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

        # 执行前准入检查
        pre_execute = run_pre_execute_gate(skill, params=params)
        if not pre_execute.allowed:
            detail = pre_execute.detail or {}
            error_type = detail.get("error_type")
            category = detail.get("category")
            return SkillExecutionResponse(
                ok=False,
                status=(
                    SkillStatus.BLOCKED
                    if error_type
                    in {"environment_error", "permission_denied", "policy_blocked"}
                    else SkillStatus.FAILED
                ),
                error_code=(
                    SkillErrorCode.KEY_MISSING
                    if detail.get("missing_keys")
                    else (
                        SkillErrorCode.POLICY_DENIED
                        if pre_execute.reason
                        else SkillErrorCode.INVALID_INPUT
                    )
                ),
                summary=pre_execute.reason,
                diagnostics={
                    "error_type": error_type,
                    "error_detail": {
                        **detail,
                        "category": category or "pre_execute",
                        "stage": "pre_execute",
                    },
                },
                skill_name=skill.name,
            )

        try:
            # 从 params 中提取 query（如果存在），否则转为字符串
            query = params.get("request", str(params))
            if self._executor is None:
                self._executor = SkillExecutor(config=self._config, llm=self._llm)
            run_dir = self._build_run_dir(session_id)
            run_dir.mkdir(parents=True, exist_ok=True)

            exec_result, generated_code = await self._executor.execute(
                skill,
                query=query,
                params=params,
                run_dir=run_dir,
                session_id=session_id,
                on_step=on_step,
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
                    diagnostics={
                        "track": (
                            skill.execution_mode
                            if skill.execution_mode
                            else (
                                ExecutionMode.PLAYBOOK
                                if skill.is_playbook
                                else ExecutionMode.KNOWLEDGE
                            )
                        )
                    },
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

    @staticmethod
    def _sanitize_session_id(session_id: str | None) -> str:
        raw = (session_id or "").strip()
        if not raw:
            return "default"
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", raw)
        return sanitized[:128] or "default"

    def _build_run_dir(self, session_id: str | None) -> Path:
        """Build run directory for skill execution.

        Uses date-grouped format: workspace/YYYY-MM-DD/{short_id}
        Groups sessions by date for easy navigation and cleanup.
        """
        from datetime import datetime

        date_str = datetime.now().strftime("%Y-%m-%d")

        # Use short hash (8 chars) for compact path
        raw_id = (session_id or "").strip()
        if raw_id:
            import hashlib

            short_id = hashlib.md5(raw_id.encode()).hexdigest()[:8]
        else:
            short_id = "default"

        return self._config.workspace_dir / date_str / short_id

    async def install(self, skill_name: str) -> Skill | None:
        """Download and install a cloud skill to local storage.

        Args:
            skill_name: Name of the cloud skill to install

        Returns:
            Installed Skill object on success, None on failure
        """
        return await self._ensure_local_skill(skill_name)

    # ---------------- Internal ----------------

    def _rerank_candidates(
        self,
        candidates: list,
    ) -> list:
        """本地优先，云端按 score 降序"""

        def rank_key(c: RecallCandidate) -> tuple[int, float]:
            tier = 0 if c.source == "local" else 1
            return (tier, -float(c.score or 0.0))

        return sorted(candidates, key=rank_key)

    async def _ensure_local_skill(self, skill_name: str) -> Skill | None:
        skill = await LocalFileRecall.load_full_skill(
            self._config.skills_dir, skill_name
        )
        if skill is not None:
            return skill

        remote_recall = (
            self._multi_recall.get_recall_by_type(RemoteRecall)
            if self._multi_recall
            else None
        )
        if not remote_recall:
            return None

        lock = self._download_locks.setdefault(skill_name, asyncio.Lock())
        async with lock:
            skill = await LocalFileRecall.load_full_skill(
                self._config.skills_dir, skill_name
            )
            if skill is not None:
                return skill

            downloaded = await self._download_cloud_skill(skill_name)
            if downloaded is None:
                return None
            try:
                await self._store.add_skill(downloaded)
            except Exception as e:
                logger.warning(
                    "Cloud skill '{}' downloaded but failed to add into store: {}",
                    downloaded.name,
                    e,
                )
                return None

            # 下载成功后重新加载（确保从文件系统获取完整数据）
            skill = await LocalFileRecall.load_full_skill(
                self._config.skills_dir, skill_name
            )
            return skill

    async def _download_cloud_skill(self, skill_name: str) -> Skill | None:
        """下载云端 skill 并加载到本地存储。"""
        remote_recall = (
            self._multi_recall.get_recall_by_type(RemoteRecall)
            if self._multi_recall
            else None
        )
        if not remote_recall:
            return None

        try:
            # 1. 获取云端 skill 的 github_url
            github_url = await self._get_cloud_skill_url(skill_name)
            if not github_url:
                return None

            # 2. 使用 download_manager 下载
            download_manager = create_default_download_manager()
            local_path = download_manager.download(
                github_url,
                self._config.skills_dir,
                skill_name,
            )
            if not local_path:
                return None

            skill = await self._store.load_from_path(Path(local_path))
            return skill
        except Exception as e:
            logger.warning("Failed to download cloud skill '{}': {}", skill_name, e)
            return None

    async def _get_cloud_skill_url(self, skill_name: str) -> str | None:
        """从云端检索服务获取 skill 的 github_url。"""
        remote_recall = (
            self._multi_recall.get_recall_by_type(RemoteRecall)
            if self._multi_recall
            else None
        )
        if not remote_recall:
            return None

        try:
            base_url = remote_recall._base_url
            with httpx.Client() as client:
                resp = client.post(
                    f"{base_url}/api/v1/download",
                    json={"skill_name": skill_name},
                )
                if resp.status_code == 200:
                    return resp.json().get("github_url", "")
        except Exception as e:
            logger.warning("Failed to get cloud skill URL for '{}': {}", skill_name, e)

        return None

    @staticmethod
    def _to_manifest(skill: Skill, source: str = "local") -> SkillManifest:
        # 确定执行模式
        exec_mode = skill.execution_mode or (
            ExecutionMode.PLAYBOOK if skill.is_playbook else ExecutionMode.KNOWLEDGE
        )

        return SkillManifest(
            name=skill.name,
            description=skill.description or "",
            parameters=skill.parameters or DEFAULT_SKILL_PARAMS,
            execution_mode=exec_mode,
            dependencies=skill.dependencies or [],
            governance=SkillGovernanceMeta(
                source="cloud" if source == "cloud" else "local",
            ),
        )

    def _candidate_to_manifest(self, candidate) -> SkillManifest:
        if candidate.source == "local" and candidate.skill:
            return self._to_manifest(candidate.skill, source="local")

        return SkillManifest(
            name=candidate.name,
            description=candidate.description or "",
            parameters=DEFAULT_SKILL_PARAMS,
            execution_mode=ExecutionMode.KNOWLEDGE,
            dependencies=[],
            governance=SkillGovernanceMeta(source="cloud"),
        )
