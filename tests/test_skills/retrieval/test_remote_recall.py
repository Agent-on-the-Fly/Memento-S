"""test_remote_recall.py — RemoteRecall 单元测试

测试远程 API 召回功能。
标记为 integration 测试，需要网络连接。

用法：
    pytest tests/test_skills/retrieval/test_remote_recall.py -v
    pytest tests/test_skills/retrieval/test_remote_recall.py -v -m integration
"""

from __future__ import annotations

import pytest
import time

from core.skill.retrieval import RemoteRecall, RecallCandidate


@pytest.mark.integration
class TestRemoteRecall:
    """RemoteRecall 测试类"""

    def test_initialization(self, remote_recall: RemoteRecall, cloud_url):
        """测试初始化"""
        assert remote_recall.name == "remote"
        assert remote_recall._base_url == cloud_url.rstrip("/")

    def test_is_available(self, remote_recall: RemoteRecall):
        """测试可用性检查"""
        available = remote_recall.is_available()
        assert available is True  # fixture 已经确保可用

    def test_get_stats(self, remote_recall: RemoteRecall):
        """测试获取统计信息"""
        stats = remote_recall.get_stats()

        assert "name" in stats
        assert "available" in stats
        assert "base_url" in stats
        assert "catalog_size" in stats
        assert stats["name"] == "remote"
        assert isinstance(stats["available"], bool)
        assert isinstance(stats["catalog_size"], int)

        print(f"\nRemote catalog size: {stats['catalog_size']}")

    @pytest.mark.asyncio
    async def test_search_returns_candidates(self, remote_recall: RemoteRecall):
        """测试搜索返回候选列表"""
        query = "data analysis"
        start = time.time()
        candidates = await remote_recall.search(query, k=5)
        elapsed = (time.time() - start) * 1000

        print(f"\nQuery: '{query}', Time: {elapsed:.1f}ms")

        assert isinstance(candidates, list)
        assert all(isinstance(c, RecallCandidate) for c in candidates)

        # 验证所有候选的 source 都是 remote
        for candidate in candidates:
            assert candidate.source == "remote"
            assert candidate.match_type == "remote"
            assert candidate.score > 0
            assert candidate.score <= 1.0

        print(f"Found {len(candidates)} candidates")
        if candidates:
            for i, c in enumerate(candidates[:3], 1):
                print(f"  {i}. {c.name} (score: {c.score:.2f})")

    @pytest.mark.asyncio
    async def test_search_with_different_queries(self, remote_recall: RemoteRecall):
        """测试不同 query 返回不同结果"""
        queries = ["web", "data", "analysis", "python"]

        results = {}
        for query in queries:
            candidates = await remote_recall.search(query, k=5)
            results[query] = len(candidates)

        print(f"\nQuery results: {results}")

        # 验证每个查询都返回了结果
        for query, count in results.items():
            assert count > 0, f"Query '{query}' returned no results"

    @pytest.mark.asyncio
    async def test_search_respects_k_parameter(self, remote_recall: RemoteRecall):
        """测试 k 参数限制返回数量"""
        query = "test"

        candidates_3 = await remote_recall.search(query, k=3)
        candidates_5 = await remote_recall.search(query, k=5)
        candidates_10 = await remote_recall.search(query, k=10)

        # 验证数量不超过 k
        assert len(candidates_3) <= 3
        assert len(candidates_5) <= 5
        assert len(candidates_10) <= 10

        print(
            f"\nk=3: {len(candidates_3)}, k=5: {len(candidates_5)}, k=10: {len(candidates_10)}"
        )

    def test_embedding_ready_property(self, remote_recall: RemoteRecall):
        """测试 embedding_ready 属性"""
        # 这是一个属性，不是方法
        ready = remote_recall.embedding_ready
        assert isinstance(ready, bool)
        print(f"\nEmbedding ready: {ready}")

    def test_size_property(self, remote_recall: RemoteRecall):
        """测试 size 属性"""
        size = remote_recall.size
        assert isinstance(size, int)
        assert size >= 0
        print(f"\nCatalog size: {size}")
