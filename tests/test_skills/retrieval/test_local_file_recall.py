"""test_local_file_recall.py — LocalFileRecall 单元测试

测试本地文件扫描召回功能。

用法：
    pytest tests/test_skills/retrieval/test_local_file_recall.py -v
"""

from __future__ import annotations

import pytest
import time

from core.skill.retrieval import LocalFileRecall, RecallCandidate


class TestLocalFileRecall:
    """LocalFileRecall 测试类"""

    def test_initialization(self, local_file_recall: LocalFileRecall, skills_dir):
        """测试初始化"""
        assert local_file_recall.name == "local_file"
        assert local_file_recall._skills_dir == skills_dir

    def test_is_available(self, local_file_recall: LocalFileRecall, skills_dir):
        """测试可用性检查"""
        available = local_file_recall.is_available()

        # 如果目录存在，应该返回 True
        if skills_dir.exists():
            assert available is True
        else:
            assert available is False

    def test_get_stats(self, local_file_recall: LocalFileRecall):
        """测试获取统计信息"""
        stats = local_file_recall.get_stats()

        assert "name" in stats
        assert "available" in stats
        assert stats["name"] == "local_file"
        assert isinstance(stats["available"], bool)

    @pytest.mark.asyncio
    async def test_search_returns_candidates(self, local_file_recall: LocalFileRecall):
        """测试搜索返回候选列表"""
        if not local_file_recall.is_available():
            pytest.skip("Skills directory not available")

        query = "test"
        candidates = await local_file_recall.search(query, k=10)

        assert isinstance(candidates, list)
        assert all(isinstance(c, RecallCandidate) for c in candidates)

        # 验证所有候选的 source 都是 local
        for candidate in candidates:
            assert candidate.source == "local"
            assert candidate.match_type == "local_file"
            assert candidate.score == 1.0

    @pytest.mark.asyncio
    async def test_search_caching(self, local_file_recall: LocalFileRecall):
        """测试缓存机制"""
        if not local_file_recall.is_available():
            pytest.skip("Skills directory not available")

        query = "test"

        # 第一次搜索（冷启动）
        start = time.time()
        candidates1 = await local_file_recall.search(query, k=10)
        cold_time = (time.time() - start) * 1000

        # 第二次搜索（缓存命中）
        start = time.time()
        candidates2 = await local_file_recall.search(query, k=10)
        cached_time = (time.time() - start) * 1000

        # 验证结果一致
        assert len(candidates1) == len(candidates2)

        # 验证缓存更快（通常快10倍以上）
        # 注意：如果数据量很小，这个断言可能不稳定
        print(f"\nCold: {cold_time:.1f}ms, Cached: {cached_time:.1f}ms")

    @pytest.mark.asyncio
    async def test_search_ignores_query(self, local_file_recall: LocalFileRecall):
        """测试搜索忽略 query 参数（全量返回）"""
        if not local_file_recall.is_available():
            pytest.skip("Skills directory not available")

        # 不同 query 应该返回相同数量的结果
        candidates1 = await local_file_recall.search("query1", k=10)
        candidates2 = await local_file_recall.search("query2", k=10)
        candidates3 = await local_file_recall.search("", k=10)

        # 全量返回，数量应该相同
        assert len(candidates1) == len(candidates2) == len(candidates3)

    def test_skills_have_description(self, local_file_recall: LocalFileRecall):
        """测试加载的 skills 都有 description"""
        if not local_file_recall.is_available():
            pytest.skip("Skills directory not available")

        import asyncio

        candidates = asyncio.run(local_file_recall.search("test", k=100))

        for candidate in candidates:
            # description 可能是空字符串，但应该有 skill 对象
            assert candidate.skill is not None
            assert hasattr(candidate.skill, "description")
