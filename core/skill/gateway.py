# SPDX-License-Identifier: Apache-2.0
"""Agent-Skill 契约层：SkillGateway 实现。

DTO 定义在 schema.py 中，通过 core.skill 包导入。
"""

from __future__ import annotations

import asyncio
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from middleware.llm import LLMClient
from utils.log_config import log_preview_long
from utils.logger import get_logger

from shared.schema import SkillConfig
from .downloader.factory import create_default_download_manager
from .execution import SkillAgent
from .execution.policy.pre_execute import run_pre_execute_gate
from .retrieval import MultiRecall, RemoteRecall
from .retrieval.local_recall import load_full_skill
from .store import SkillStorage
from shared.schema import (
    ExecutionMode,
    SkillErrorCode,
    SkillExecutionResponse,
    SkillGovernanceMeta,
    SkillManifest,
    SkillSearchResult,
    SkillStatus,
)
from .schema import DEFAULT_SKILL_PARAMS, DiscoverStrategy, Skill

logger = get_logger(__name__)


@dataclass(frozen=True)
class SkillPreflightResult:
    """Backward-compatible result for callers that preflight separately."""

    ready: bool
    status: SkillStatus
    message: str = ""
    error_code: SkillErrorCode | None = None


class SkillGateway:
    """Skill 契约实现：目录层、运行时层、治理层。

    这是唯一的实现类，外部通过此接口与 Skill 系统交互。
    内部管理 SkillStore，生产环境通过 core.skill.init_skill_system() 创建。
    """

    def __init__(
        self,
        config: "SkillConfig",
        store: "SkillStorage",
        multi_recall: MultiRecall | None = None,
        agent: SkillAgent | None = None,
        llm: "LLMClient" | None = None,
    ):
        self._config = config
        self._store = store
        self._multi_recall = multi_recall
        self._agent = agent
        self._llm = llm
        self._download_locks: dict[str, asyncio.Lock] = {}
        self._evolution_engine = None

    @classmethod
    async def from_config(
        cls,
        config: "SkillConfig | None" = None,
    ) -> "SkillGateway":
        """异步工厂方法创建 SkillGateway。

        内部自动创建所有依赖（Store, MultiRecall, Executor, LLM）。

        Args:
            config: SkillConfig 配置，为 None 时自动从全局配置创建

        Returns:
            初始化好的 SkillGateway 实例
        """
        if config is None:
            config = SkillConfig.from_global_config()

        store = await SkillStorage.from_config(config)
        llm = LLMClient()
        agent = SkillAgent(config=config, llm=llm)
        multi_recall = await MultiRecall.from_config(config)

        return cls(
            config=config,
            store=store,
            multi_recall=multi_recall,
            agent=agent,
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
            strategy: DiscoverStrategy.LOCAL_ONLY or MULTI_RECALL
            query: search query for multi_recall strategy
            k: max candidates for multi_recall strategy
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
            cloud_only: If True, skip local recall and only return remote results
        """
        try:
            if self._multi_recall is None:
                return []

            source_filter = "cloud" if cloud_only else None
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

    async def route(self, query: str, k: int | None = None) -> list[SkillManifest]:
        """Route a goal to a compact local skill set for planning.

        This is the paper's Read phase.  MultiRecall may use keyword, Qwen, or
        their score-aware fusion; cloud entries are excluded because a plan can
        only directly execute installed skills.
        """
        if self._multi_recall is None:
            return []
        route_k = max(1, int(k or self._config.retrieval_top_k))
        try:
            candidates = await self._multi_recall.search(
                query,
                k=route_k,
                per_recall_k=max(route_k * 3, route_k),
                source_filter="local",
            )
            return [self._candidate_to_manifest(c) for c in candidates]
        except Exception as exc:
            logger.warning("Skill route failed for query '{}': {}", query, exc)
            return []

    async def get_manifest(self, skill_name: str) -> SkillManifest | None:
        """Return metadata for one installed skill."""
        skill = await self._store.get_skill(skill_name)
        return self._to_manifest(skill) if skill is not None else None

    async def read(self, skill_name: str) -> str | None:
        """Read the full SKILL.md content for one installed skill."""
        skill = await self._store.get_skill(skill_name)
        if skill is None:
            return None
        if skill.source_dir:
            skill_md = Path(skill.source_dir) / "SKILL.md"
            if skill_md.is_file():
                return skill_md.read_text(encoding="utf-8")
        return skill.content

    async def preflight(
        self, skill_name: str, params: dict[str, Any] | None = None
    ) -> SkillPreflightResult:
        """Validate that an installed skill can start without executing it."""
        skill = await self._store.get_skill(skill_name)
        if skill is None:
            return SkillPreflightResult(
                ready=False,
                status=SkillStatus.FAILED,
                message=f"Skill '{skill_name}' not found",
                error_code=SkillErrorCode.SKILL_NOT_FOUND,
            )
        result = run_pre_execute_gate(skill, params=params or {})
        if not result.allowed:
            return SkillPreflightResult(
                ready=False,
                status=SkillStatus.BLOCKED,
                message=result.reason,
                error_code=SkillErrorCode.POLICY_DENIED,
            )
        return SkillPreflightResult(
            ready=True,
            status=SkillStatus.SUCCESS,
            message="ready",
        )

    # ── Runtime ──────────────────────────────────────────────────────────

    async def execute(
        self,
        skill_name: Skill | str,
        params: dict[str, Any],
        options: Any = None,
        session_id: str | None = None,
        on_step: Any | None = None,
    ) -> SkillExecutionResponse:
        resolved_skill_name = (
            skill_name.name if isinstance(skill_name, Skill) else skill_name
        )

        logger.debug(
            f"[SkillGateway.execute] ENTRY: "
            f"skill_name='{resolved_skill_name}', "
            f"params_keys={list(params.keys())}, "
            f"session_id={session_id!r}"
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

        # ── P2-2: BEFORE_SKILL_EXEC Hook ──────────────────────────────────
        # SkillPolicyHook 在全局 HookExecutor 中注册，通过 BEFORE_SKILL_EXEC 事件触发
        # 注意：run_pre_execute_gate 的结果仍直接使用（保持向后兼容）
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

        # ── Dependency Pre-check (warning-only) ─────────────────────────────────
        # 检查缺失的 Python 依赖，记录警告但不阻断执行。
        # 实际安装在 UvLocalSandbox 层统一处理（python_repl 调用时自动安装）。
        # Gateway 层只做检查/警告，不做安装，保持职责单一。
        if skill.dependencies:
            from core.skill.execution.policy.pre_execute import check_missing_dependencies

            missing = check_missing_dependencies(skill.dependencies)
            if missing:
                logger.warning(
                    f"[SkillGateway.execute] Skill '{skill.name}' is missing dependencies: {missing}. "
                    f"Dependencies will be installed automatically when python_repl is invoked."
                )

        try:
            query = params.get("request", str(params))
            if self._agent is None:
                self._agent = SkillAgent(config=self._config, llm=self._llm)
            run_dir = self._build_run_dir(session_id)
            run_dir.mkdir(parents=True, exist_ok=True)

            logger.debug(
                f"[SkillGateway.execute] CALLING SkillAgent.run: "
                f"skill={skill.name}, "
                f"query='{log_preview_long(query)}', "
                f"run_dir={run_dir}, "
                f"session_id={session_id}, "
                f"allowed_tools={skill.allowed_tools}"
            )

            exec_result, generated_code = await self._agent.run(
                skill,
                query=query,
                params=params,
                run_dir=run_dir,
                session_id=session_id or "",
                on_step=on_step,
            )

            logger.debug(
                f"[SkillGateway.execute] SkillAgent.run RETURNED: "
                f"skill={skill.name}, "
                f"success={exec_result.success}, "
                f"error='{str(exec_result.error)[:80] if exec_result.error else 'none'}', "
                f"created_files={exec_result.artifacts}, "
                f"generated_code_len={len(generated_code) if generated_code else 0}"
            )

            # 根据产物有无决定 PARTIAL vs FAILED
            artifacts = exec_result.artifacts or []
            has_artifacts = bool(artifacts)

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
                    artifacts=artifacts,
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

            # 失败时：根据是否有产物决定 PARTIAL 还是 FAILED
            # PARTIAL = 有产物但未完全成功（可能是 loop、超时等），后续步骤仍可继续
            status = SkillStatus.PARTIAL if has_artifacts else SkillStatus.FAILED
            diagnostics = {
                "error_type": exec_result.error_type.value
                if exec_result.error_type
                else None,
                "error_detail": exec_result.error_detail or None,
                "has_artifacts": has_artifacts,
            }
            return SkillExecutionResponse(
                ok=False,
                status=status,
                error_code=SkillErrorCode.RUNTIME_ERROR,
                summary=exec_result.error or ("Skill partially completed with artifacts" if has_artifacts else "Skill execution failed"),
                output=exec_result.result,
                outputs={"operation_results": exec_result.operation_results or []},
                artifacts=artifacts,
                diagnostics=diagnostics,
                skill_name=skill.name,
            )
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            logger.warning(
                "Skill execution failed for '{}': {}\nCall stack:\n{}",
                skill_name,
                e,
                tb,
            )
            return SkillExecutionResponse(
                ok=False,
                status=SkillStatus.FAILED,
                error_code=SkillErrorCode.INTERNAL_ERROR,
                summary=str(e),
                skill_name=str(skill_name),
            )

    async def install(self, skill_name: str) -> Skill | None:
        """Download and install a cloud skill to local storage."""
        return await self._ensure_local_skill(skill_name)

    # ── Read-Write reflective evolution ─────────────────────────────────

    def _get_evolution_engine(self):
        if self._evolution_engine is None:
            from .evolution import SkillEvolutionEngine

            if self._llm is None:
                self._llm = LLMClient()
            self._evolution_engine = SkillEvolutionEngine(
                config=self._config,
                llm=self._llm,
                candidate_runner=self._run_evolution_candidate,
            )
        return self._evolution_engine

    async def record_skill_outcome(self, skill_name: str, success: bool) -> dict[str, Any]:
        """Update the empirical utility table for an executed skill."""
        try:
            return await self._get_evolution_engine().record_outcome(skill_name, success)
        except Exception as exc:
            logger.warning("Failed to record utility for '{}': {}", skill_name, exc)
            return {}

    async def evolve_failure(
        self,
        *,
        task: str,
        used_skills: list[str],
        trace: list[dict[str, Any]],
        rationale: str,
        session_id: str = "",
    ):
        """Run automatic attribution, rewrite, tests, and guarded deployment."""
        return await self._get_evolution_engine().evolve_failure(
            task=task,
            used_skills=used_skills,
            trace=trace,
            rationale=rationale,
            session_id=session_id,
        )

    async def rollback_skill(self, skill_name: str):
        """Restore the most recent successful pre-evolution snapshot."""
        return await self._get_evolution_engine().rollback_last(skill_name)

    async def _run_evolution_candidate(
        self, skill: Skill, request: str
    ) -> dict[str, Any]:
        """Execute an isolated candidate through the normal SkillAgent runtime."""
        if self._agent is None:
            self._agent = SkillAgent(config=self._config, llm=self._llm)
        test_root = self._config.workspace_dir / ".evolution-tests"
        test_root.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="candidate-", dir=test_root))
        try:
            outcome, _generated_code = await self._agent.run(
                skill,
                query=request,
                params={"request": request},
                run_dir=run_dir,
                session_id="skill-evolution-test",
                on_step=None,
            )
            return {
                "success": outcome.success,
                "result": outcome.result,
                "error": outcome.error,
                "error_type": outcome.error_type.value if outcome.error_type else None,
                "artifacts": outcome.artifacts,
                "operation_results": outcome.operation_results,
            }
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    # ── Internal ────────────────────────────────────────────────────────

    @staticmethod
    def _sanitize_session_id(session_id: str | None) -> str:
        raw = (session_id or "").strip()
        if not raw:
            return "default"
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", raw)
        return sanitized[:128] or "default"

    def _build_run_dir(self, session_id: str | None) -> Path:
        import hashlib
        from datetime import datetime

        date_str = datetime.now().strftime("%Y-%m-%d")
        raw_id = (session_id or "").strip()
        if raw_id:
            short_id = hashlib.md5(raw_id.encode()).hexdigest()[:8]
        else:
            short_id = "default"

        return self._config.workspace_dir / date_str / short_id

    def _rerank_candidates(
        self,
        candidates: list,
    ) -> list:
        """local 优先，remote 按 score 降序"""

        def rank_key(c: SkillSearchResult) -> tuple[int, float]:
            tier = 0 if c.source == "local" else 1
            return (tier, -float(c.score or 0.0))

        return sorted(candidates, key=rank_key)

    async def _ensure_local_skill(self, skill_name: str) -> Skill | None:
        """确保 skill 在本地可用，必要时从云端下载"""
        skill = load_full_skill(self._config.skills_dir, skill_name)
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
            skill = load_full_skill(self._config.skills_dir, skill_name)
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

            skill = load_full_skill(self._config.skills_dir, skill_name)
            return skill

    async def _download_cloud_skill(self, skill_name: str) -> Skill | None:
        remote_recall = (
            self._multi_recall.get_recall_by_type(RemoteRecall)
            if self._multi_recall
            else None
        )
        if not remote_recall:
            return None

        try:
            github_url = await self._get_cloud_skill_url(skill_name)
            if not github_url:
                return None

            download_manager = create_default_download_manager()
            local_path = download_manager.download(
                github_url,
                self._config.skills_dir,
                skill_name,
            )
            if not local_path:
                return None

            skill = await self._store.get_skill(skill_name)
            if skill:
                return skill
            # load from disk
            return load_full_skill(self._config.skills_dir, skill_name)
        except Exception as e:
            logger.warning("Failed to download cloud skill '{}': {}", skill_name, e)
            return None

    async def _get_cloud_skill_url(self, skill_name: str) -> str | None:
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

    def _candidate_to_manifest(
        self, result: SkillSearchResult, *, load_skill: bool = True
    ) -> SkillManifest:
        """将 SkillSearchResult 转换为 SkillManifest。

        Args:
            result: 检索结果
            load_skill: 是否尝试从本地加载完整 Skill 对象来构建完整 manifest
        """
        if result.source == "local" and load_skill:
            skill = load_full_skill(self._config.skills_dir, result.name)
            if skill is not None:
                return self._to_manifest(skill, source="local")

        return SkillManifest(
            name=result.name,
            description=result.description or "",
            parameters=DEFAULT_SKILL_PARAMS,
            execution_mode=ExecutionMode.KNOWLEDGE,
            dependencies=[],
            governance=SkillGovernanceMeta(source=result.source),
        )
