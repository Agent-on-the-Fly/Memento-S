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

from utils.logger import get_logger
from .base import BaseRecall
from .local_recall import LocalRecall
from .remote_recall import RemoteRecall
from shared.schema import SkillSearchResult

logger = get_logger(__name__)


class MultiRecall:
    """多路召回合并器

    管理多个召回策略，执行并行召回并合并去重（local 优先）。

    Args:
        recalls: 召回策略列表
    """

    def __init__(self, recalls: "list[BaseRecall] | None" = None):
        self._recalls = recalls or []

    @classmethod
    async def from_config(cls, config: "SkillConfig") -> "MultiRecall":
        """从配置异步创建 MultiRecall 实例

        自动创建 LocalRecall 和 RemoteRecall（如果配置了 cloud_catalog_url）。
        不再创建 LocalDbRecall。
        """
        recalls: list[BaseRecall] = []

        # LocalRecall（总是可用）
        local = LocalRecall.from_config(config)
        if local.is_available():
            recalls.append(local)

        # RemoteRecall（如果配置了 cloud_catalog_url）
        remote = await RemoteRecall.from_config(config)
        if remote:
            recalls.append(remote)

        return cls(recalls)

    def add_recall(self, recall: "BaseRecall") -> None:
        self._recalls.append(recall)

    def remove_recall(self, name: str) -> bool:
        for i, r in enumerate(self._recalls):
            if r.name == name:
                self._recalls.pop(i)
                return True
        return False

    def get_available_recalls(self) -> list["BaseRecall"]:
        return [r for r in self._recalls if r.is_available()]

    def get_recall_by_type(self, recall_type: type) -> "BaseRecall | None":
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

        # 合并去重（local 优先）
        seen: dict[str, SkillSearchResult] = {}
        for recall, result in zip(available, results):
            if isinstance(result, Exception):
                logger.warning("[MULTI_RECALL] '{}' failed: {}", recall.name, result)
                continue
            for candidate in result:
                if source_filter and candidate.source != source_filter:
                    continue
                existing = seen.get(candidate.name)
                if existing is None:
                    seen[candidate.name] = candidate
                elif candidate.source == "local" and existing.source != "local":
                    seen[candidate.name] = candidate

        candidates = sorted(seen.values(), key=lambda c: c.score, reverse=True)[:k]

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
        recall: "BaseRecall",
        query: str,
        k: int,
        **kwargs,
    ) -> "list[SkillSearchResult] | Exception":
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
                close_method = getattr(recall, "close")
                if callable(close_method):
                    try:
                        if asyncio.iscoroutinefunction(close_method):
                            await close_method()
                        else:
                            close_method()
                    except Exception as e:
                        logger.warning("Failed to close recall '{}': {}", recall.name, e)
