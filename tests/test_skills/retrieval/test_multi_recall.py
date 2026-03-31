"""test_multi_recall.py — MultiRecall 单元测试

测试多路召回合并功能。

用法：
    pytest tests/test_skills/retrieval/test_multi_recall.py -v
"""

from __future__ import annotations

import pytest
import time

from core.skill.retrieval import MultiRecall, LocalFileRecall, RecallCandidate


class TestMultiRecall:
    """MultiRecall 测试类"""

    def test_initialization_with_strategies(self, multi_recall: MultiRecall):
        """测试使用策略列表初始化"""
        assert isinstance(multi_recall._recalls, list)
        assert len(multi_recall._recalls) >= 1  # 至少应该有 LocalFileRecall

    def test_get_available_recalls(self, multi_recall: MultiRecall):
        """测试获取可用策略"""
        available = multi_recall.get_available_recalls()

        assert isinstance(available, list)
        assert len(available) >= 1  # 至少有一个可用

        # 所有可用策略的 is_available 应该返回 True
        for recall in available:
            assert recall.is_available() is True

        print(f"\nAvailable recalls: {[r.name for r in available]}")

    def test_get_stats(self, multi_recall: MultiRecall):
        """测试获取统计信息"""
        stats = multi_recall.get_stats()

        assert "total_strategies" in stats
        assert "available_strategies" in stats
        assert "strategies" in stats

        assert isinstance(stats["total_strategies"], int)
        assert isinstance(stats["available_strategies"], int)
        assert stats["available_strategies"] <= stats["total_strategies"]

        print(
            f"\nStats: {stats['total_strategies']} total, {stats['available_strategies']} available"
        )

    @pytest.mark.asyncio
    async def test_search_returns_candidates(self, multi_recall: MultiRecall):
        """测试搜索返回候选列表"""
        if not multi_recall.get_available_recalls():
            pytest.skip("No available recall strategies")

        query = "test"
        start = time.time()
        candidates = await multi_recall.search(query, k=10)
        elapsed = (time.time() - start) * 1000

        print(
            f"\nQuery: '{query}', Time: {elapsed:.1f}ms, Candidates: {len(candidates)}"
        )

        assert isinstance(candidates, list)
        assert all(isinstance(c, RecallCandidate) for c in candidates)
        assert len(candidates) <= 10  # 不超过 k

    @pytest.mark.asyncio
    async def test_search_local_priority(self, multi_recall: MultiRecall):
        """测试 local 优先策略"""
        available = multi_recall.get_available_recalls()
        if not available:
            pytest.skip("No available recall strategies")

        # 如果同时有 local 和 remote，验证 local 优先
        has_local = any(r.name == "local_file" for r in available)
        has_remote = any(r.name == "remote" for r in available)

        if not (has_local and has_remote):
            pytest.skip("Need both local and remote recalls to test priority")

        query = "web"
        candidates = await multi_recall.search(query, k=20)

        # 统计来源
        local_count = sum(1 for c in candidates if c.source == "local")
        remote_count = sum(1 for c in candidates if c.source == "remote")

        print(f"\nLocal: {local_count}, Remote: {remote_count}")

        # 如果同一个 skill 在 local 和 remote 都存在，应该优先使用 local
        # 这里只是验证搜索成功，不验证具体逻辑

    @pytest.mark.asyncio
    async def test_search_respects_k_parameter(self, multi_recall: MultiRecall):
        """测试 k 参数限制返回数量"""
        if not multi_recall.get_available_recalls():
            pytest.skip("No available recall strategies")

        query = "test"

        candidates_3 = await multi_recall.search(query, k=3)
        candidates_5 = await multi_recall.search(query, k=5)
        candidates_10 = await multi_recall.search(query, k=10)

        assert len(candidates_3) <= 3
        assert len(candidates_5) <= 5
        assert len(candidates_10) <= 10

        print(
            f"\nk=3: {len(candidates_3)}, k=5: {len(candidates_5)}, k=10: {len(candidates_10)}"
        )

    @pytest.mark.asyncio
    async def test_search_sorted_by_score(self, multi_recall: MultiRecall):
        """测试结果按分数降序排序"""
        if not multi_recall.get_available_recalls():
            pytest.skip("No available recall strategies")

        query = "test"
        candidates = await multi_recall.search(query, k=10)

        if len(candidates) >= 2:
            # 验证分数是降序
            for i in range(len(candidates) - 1):
                assert candidates[i].score >= candidates[i + 1].score

    @pytest.mark.asyncio
    async def test_search_with_per_recall_k(self, multi_recall: MultiRecall):
        """测试 per_recall_k 参数"""
        if not multi_recall.get_available_recalls():
            pytest.skip("No available recall strategies")

        query = "test"

        # 使用 per_recall_k 限制每个策略的返回数
        candidates = await multi_recall.search(query, k=10, per_recall_k=5)

        assert isinstance(candidates, list)
        assert len(candidates) <= 10

    def test_add_recall(self, skills_dir):
        """测试动态添加策略"""
        multi = MultiRecall()

        assert len(multi._recalls) == 0

        # 添加策略
        recall = LocalFileRecall(skills_dir)
        multi.add_recall(recall)

        assert len(multi._recalls) == 1
        assert multi._recalls[0] == recall

    def test_remove_recall(self, skills_dir):
        """测试动态移除策略"""
        recall = LocalFileRecall(skills_dir)
        multi = MultiRecall([recall])

        assert len(multi._recalls) == 1

        # 移除策略
        removed = multi.remove_recall("local_file")

        assert removed is True
        assert len(multi._recalls) == 0

        # 移除不存在的策略
        removed = multi.remove_recall("non_existent")
        assert removed is False

    @pytest.mark.asyncio
    async def test_close(self, multi_recall: MultiRecall):
        """测试关闭资源"""
        # 只是验证不抛出异常
        await multi_recall.close()


class TestMultiRecallEmpty:
    """MultiRecall 空状态测试"""

    def test_empty_recalls(self):
        """测试空策略列表"""
        multi = MultiRecall([])

        assert multi.get_available_recalls() == []
        assert multi.get_stats()["total_strategies"] == 0

    @pytest.mark.asyncio
    async def test_search_with_no_available_recalls(self):
        """测试无可用策略时搜索"""
        multi = MultiRecall([])

        candidates = await multi.search("test", k=10)

        assert candidates == []
