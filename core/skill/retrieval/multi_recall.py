# SPDX-License-Identifier: Apache-2.0
"""multi_recall — 多路召回合并器

管理多个召回策略，执行并行召回并合并去重。
只使用 LocalRecall 和 RemoteRecall，不依赖 DB/Vector。

用法示例：
    from core.skill.retrieval import MultiRecall, LocalRecall, RemoteRecall

    multi = MultiRecall(recalls=[
        LocalRecall(skills_dir),
        RemoteRecall(base_url),
    ])

    candidates = await multi.recall("数据分析", k=10)
"""

from __future__ import annotations

import asyncio

from shared.schema import SkillConfig, SkillSearchResult
from utils.logger import get_logger

from .base import BaseRecall
from .local_recall import LocalRecall
from .qwen_recall import QwenRecall
from .remote_recall import RemoteRecall

logger = get_logger(__name__)


class MultiRecall:
    """多路召回合并器

    管理多个召回策略，执行并行召回并合并去重（local 优先）。

    Args:
        recalls: 召回策略列表
    """

    def __init__(self, recalls: list[BaseRecall] | None = None):
        self._recalls = recalls or []

    @classmethod
    async def from_config(cls, config: SkillConfig) -> MultiRecall:
        """从配置异步创建 MultiRecall 实例

        自动创建 LocalRecall 和 RemoteRecall（如果配置了 cloud_catalog_url）。
        不再创建 LocalDbRecall。
        """
        recalls: list[BaseRecall] = []

        backend = (config.retrieval_router_backend or "keyword").lower()

        # Keyword recall remains the safe fallback and one half of hybrid recall.
        if backend in {"keyword", "hybrid"}:
            local = LocalRecall.from_config(config)
            if local.is_available():
                recalls.append(local)

        if backend in {"qwen", "hybrid"}:
            qwen = QwenRecall.from_config(config)
            if qwen.is_available():
                recalls.append(qwen)
            else:
                logger.warning(
                    "[MULTI_RECALL] Qwen router selected but no local model or usable "
                    "embedding endpoint is configured; using available fallbacks"
                )
                # The reference router falls back to its sparse route when the
                # embedding runtime is unavailable.  qwen-only mode needs the
                # same guarantee; hybrid already added LocalRecall above.
                if backend == "qwen":
                    local = LocalRecall.from_config(config)
                    if local.is_available():
                        recalls.append(local)

        # RemoteRecall（如果配置了 cloud_catalog_url）
        remote = await RemoteRecall.from_config(config)
        if remote:
            recalls.append(remote)

        return cls(recalls)

    def add_recall(self, recall: BaseRecall) -> None:
        self._recalls.append(recall)

    def remove_recall(self, name: str) -> bool:
        for i, r in enumerate(self._recalls):
            if r.name == name:
                self._recalls.pop(i)
                return True
        return False

    def get_available_recalls(self) -> list[BaseRecall]:
        return [r for r in self._recalls if r.is_available()]

    def get_recall_by_type(self, recall_type: type) -> BaseRecall | None:
        for recall in self._recalls:
            if isinstance(recall, recall_type):
                return recall
        return None

    async def recall(
        self,
        query: str,
        k: int = 10,
        per_recall_k: int | None = None,
        source_filter: str | None = None,
        **kwargs,
    ) -> list[SkillSearchResult]:
        """执行多路召回并合并结果"""
        per_k = per_recall_k or k
        available = self.get_available_recalls()

        if not available:
            logger.warning("[MULTI_RECALL] No available recall strategies")
            return []

        # 并行执行所有召回
        tasks = [self._safe_search(r, query, per_k, **kwargs) for r in available]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Score-aware reciprocal rank fusion.  This lets keyword and Qwen recall
        # vote on the same local skill while keeping remote discovery available.
        grouped: dict[str, list[tuple[str, int, SkillSearchResult]]] = {}
        for recall, result in zip(available, results, strict=True):
            if isinstance(result, Exception):
                logger.warning("[MULTI_RECALL] '{}' failed: {}", recall.name, result)
                continue
            for rank, candidate in enumerate(result, start=1):
                if source_filter and candidate.source != source_filter:
                    continue
                grouped.setdefault(candidate.name, []).append(
                    (recall.name, rank, candidate)
                )

        fused: list[SkillSearchResult] = []
        for name, hits in grouped.items():
            best = max(hits, key=lambda item: item[2].score)[2]
            # Score-aware RRF: each channel contributes both rank evidence and
            # its calibrated score.  Dividing by the active channel count keeps
            # the result in [0, 1] while rewarding cross-router agreement.
            score = sum(
                float(hit.score) * 61.0 / (60.0 + rank)
                for _recall_name, rank, hit in hits
            ) / len(available)
            match_types = sorted(
                {hit.match_type for _, _, hit in hits if hit.match_type}
            )
            metadata = dict(best.metadata)
            metadata["fused_match_types"] = match_types
            metadata["router_scores"] = {
                recall_name: {"rank": rank, "score": hit.score}
                for recall_name, rank, hit in hits
            }
            fused.append(
                SkillSearchResult(
                    name=name,
                    description=best.description,
                    source=(
                        "local"
                        if any(h.source == "local" for _, _, h in hits)
                        else best.source
                    ),
                    score=min(1.0, score),
                    match_type="hybrid" if len(match_types) > 1 else best.match_type,
                    metadata=metadata,
                )
            )

        candidates = sorted(fused, key=lambda c: (-c.score, c.name))[:k]

        logger.info(
            "[MULTI_RECALL] query='{}' → {} candidates (local={}, remote={})",
            query,
            len(candidates),
            sum(1 for c in candidates if c.source == "local"),
            len(candidates) - sum(1 for c in candidates if c.source == "local"),
        )
        return candidates

    async def search(
        self,
        query: str,
        k: int = 10,
        per_recall_k: int | None = None,
        **kwargs,
    ) -> list[SkillSearchResult]:
        """兼容旧接口，转调 recall"""
        return await self.recall(query, k=k, per_recall_k=per_recall_k, **kwargs)

    async def _safe_search(
        self,
        recall: BaseRecall,
        query: str,
        k: int,
        **kwargs,
    ) -> list[SkillSearchResult] | Exception:
        try:
            return await recall.search(query, k=k, **kwargs)
        except Exception as e:
            logger.warning("Recall '{}' failed: {}", recall.name, e)
            return e

    def get_stats(self) -> dict:
        return {
            "total_strategies": len(self._recalls),
            "available_strategies": len(self.get_available_recalls()),
            "strategies": [r.get_stats() for r in self._recalls],
        }

    async def close(self) -> None:
        for recall in self._recalls:
            if hasattr(recall, "close"):
                close_method = recall.close
                if callable(close_method):
                    try:
                        if asyncio.iscoroutinefunction(close_method):
                            await close_method()
                        else:
                            close_method()
                    except Exception as e:
                        logger.warning(
                            "Failed to close recall '{}': {}", recall.name, e
                        )
