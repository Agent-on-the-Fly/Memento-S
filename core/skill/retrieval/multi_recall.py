"""multi_recall — 多路召回合并器

管理多个召回策略，执行并行召回并合并去重。

用法示例：
    from core.skill.retrieval import MultiRecall, LocalFileRecall, LocalDbRecall, RemoteRecall

    recalls = [
        LocalFileRecall(skills_dir),
        LocalDbRecall(db_path, embedding_client),
        RemoteRecall(base_url),
    ]

    multi = MultiRecall(recalls)
    candidates = await multi.recall("数据分析", k=10)
"""

from __future__ import annotations

import asyncio

from utils.logger import get_logger
from .schema import RecallCandidate

from .base import BaseRecall

logger = get_logger(__name__)


class MultiRecall:
    """多路召回合并器

    管理多个召回策略，执行并行召回并合并去重（local 优先）。

    Args:
        recalls: 召回策略列表，按优先级排序（高优先级在前）
    """

    def __init__(self, recalls: "list[BaseRecall] | None" = None):
        self._recalls = recalls or []

    @classmethod
    def from_config(cls, config: "SkillConfig") -> "MultiRecall":
        """从配置创建 MultiRecall 实例

        自动创建所有可用的召回策略（LocalFileRecall、LocalDbRecall、RemoteRecall）。

        Args:
            config: SkillConfig 配置

        Returns:
            配置好的 MultiRecall 实例
        """
        recalls = []

        # 导入具体的 recall 类
        from .local_file_recall import LocalFileRecall
        from .local_db_recall import LocalDbRecall
        from .remote_recall import RemoteRecall

        # 添加 LocalFileRecall（总是可用）
        file_recall = LocalFileRecall.from_config(config)
        if file_recall.is_available():
            recalls.append(file_recall)

        # 添加 LocalDbRecall（如果 embedding 可用）
        try:
            db_recall = LocalDbRecall.from_config(config)
            if db_recall.is_available():
                recalls.append(db_recall)
        except Exception:
            pass  # Embedding 不可用，跳过

        # 添加 RemoteRecall（如果配置了云端 catalog URL）
        remote_recall = RemoteRecall.from_config(config)
        if remote_recall:
            recalls.append(remote_recall)

        return cls(recalls)

    def add_recall(self, recall: "BaseRecall") -> None:
        """添加召回策略"""
        self._recalls.append(recall)

    def remove_recall(self, name: str) -> bool:
        """按名称移除召回策略

        Args:
            name: 召回策略名称

        Returns:
            True 如果成功移除，False 如果未找到
        """
        for i, r in enumerate(self._recalls):
            if r.name == name:
                self._recalls.pop(i)
                return True
        return False

    def get_available_recalls(self) -> list["BaseRecall"]:
        """获取所有可用的召回策略"""
        return [r for r in self._recalls if r.is_available()]

    def get_recall_by_type(self, recall_type: type) -> "BaseRecall | None":
        """获取指定类型的召回策略

        Args:
            recall_type: 召回策略类型（如 RemoteRecall）

        Returns:
            找到的召回策略实例，如果未找到则返回 None
        """
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
    ) -> list[RecallCandidate]:
        """执行多路召回并合并结果

        并行调用所有可用的召回策略，合并结果并按 source 优先级去重
        （local 优先于 remote）。

        Args:
            query: 搜索查询
            k: 返回的最大结果总数
            per_recall_k: 每个召回策略返回的最大结果数，None 表示使用 k
            source_filter: 可选的源过滤器，"local" 或 "remote"，None 表示不过滤
            **kwargs: 传递给底层召回策略的额外参数

        Returns:
            合并去重后的 RecallCandidate 列表
        """
        per_k = per_recall_k or k
        available_recalls = self.get_available_recalls()

        if not available_recalls:
            logger.warning("[MULTI_RECALL] No available recall strategies")
            return []

        # 并行执行所有召回
        tasks = [
            self._safe_search(r, query, per_k, **kwargs) for r in available_recalls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 合并结果，按 source 优先级去重（local 优先）
        seen: dict[str, RecallCandidate] = {}

        for recall, result in zip(available_recalls, results):
            if isinstance(result, Exception):
                logger.warning(
                    "[MULTI_RECALL] Recall '{}' failed: {}", recall.name, result
                )
                continue

            logger.debug(
                "[MULTI_RECALL] '{}' returned {} results", recall.name, len(result)
            )

            for candidate in result:
                # 应用 source_filter（在结果层面过滤）
                if source_filter and candidate.source != source_filter:
                    continue

                existing = seen.get(candidate.name)
                if existing is None:
                    # 新技能，直接添加
                    seen[candidate.name] = candidate
                elif candidate.source == "local" and existing.source == "remote":
                    # local 优先于 remote，替换
                    seen[candidate.name] = candidate
                # 其他情况保持现有（local 优先于 local 时保留先出现的）

        # 按分数降序排序，取前 k 个
        candidates = sorted(seen.values(), key=lambda c: c.score, reverse=True)[:k]

        # 统计
        local_count = sum(1 for c in candidates if c.source == "local")
        remote_count = len(candidates) - local_count

        logger.info(
            "[MULTI_RECALL] query='{}' → {} candidates "
            "(local={}, remote={}, strategies={})",
            query,
            len(candidates),
            local_count,
            remote_count,
            len(available_recalls),
        )

        return candidates

    async def search(
        self,
        query: str,
        k: int = 10,
        per_recall_k: int | None = None,
        **kwargs,
    ) -> list[RecallCandidate]:
        """兼容旧接口，转调 recall。"""
        return await self.recall(query, k=k, per_recall_k=per_recall_k, **kwargs)

    async def _safe_search(
        self,
        recall: "BaseRecall",
        query: str,
        k: int,
        **kwargs,
    ) -> "list[RecallCandidate] | Exception":
        """安全地执行单个召回策略，捕获异常"""
        try:
            return await recall.search(query, k=k, **kwargs)
        except Exception as e:
            logger.warning("Recall '{}' failed: {}", recall.name, e)
            return e  # 返回异常而不是抛出

    def get_stats(self) -> dict:
        """获取所有召回策略的统计信息"""
        return {
            "total_strategies": len(self._recalls),
            "available_strategies": len(self.get_available_recalls()),
            "strategies": [r.get_stats() for r in self._recalls],
        }

    async def close(self) -> None:
        """关闭所有召回策略的资源"""
        for recall in self._recalls:
            if hasattr(recall, "close") and callable(getattr(recall, "close")):
                try:
                    recall.close()
                except Exception as e:
                    logger.warning("Failed to close recall '{}': {}", recall.name, e)
