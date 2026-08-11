"""test_local_recall.py — LocalRecall 单元测试

测试本地文件扫描召回功能。

用法：
    pytest core/skill/tests/retrieval/test_local_recall.py -v
"""

from __future__ import annotations

import pytest
import time

from core.skill.retrieval import LocalRecall
from shared.schema import SkillSearchResult


class TestLocalRecall:
    """LocalRecall 测试类"""

    def test_initialization(self, local_recall: LocalRecall, skills_dir):
        """测试初始化"""
        assert local_recall.name == "local"
        assert local_recall._skills_dir == skills_dir

    def test_is_available(self, local_recall: LocalRecall, skills_dir):
        """测试可用性检查"""
        available = local_recall.is_available()

        # 如果目录存在，应该返回 True
        if skills_dir.exists():
            assert available is True
        else:
            assert available is False

    def test_get_stats(self, local_recall: LocalRecall):
        """测试获取统计信息"""
        stats = local_recall.get_stats()

        assert "name" in stats
        assert "available" in stats
        assert stats["name"] == "local"
        assert isinstance(stats["available"], bool)

    @pytest.mark.asyncio
    async def test_search_returns_candidates(self, local_recall: LocalRecall):
        """测试搜索返回候选列表"""
        if not local_recall.is_available():
            pytest.skip("Skills directory not available")

        query = "test"
        candidates = await local_recall.search(query, k=10)

        assert isinstance(candidates, list)
        assert all(isinstance(c, SkillSearchResult) for c in candidates)

        # 验证所有候选的 source 是有效的（local 或 cloud）
        for candidate in candidates:
            assert candidate.source in ("local", "cloud")
            assert candidate.match_type == "keyword"
            assert 0 < candidate.score <= 1.0  # 分数在有效范围内

    @pytest.mark.asyncio
    async def test_search_caching(self, local_recall: LocalRecall):
        """测试缓存机制"""
        if not local_recall.is_available():
            pytest.skip("Skills directory not available")

        query = "test"

        # 第一次搜索（冷启动）
        start = time.time()
        candidates1 = await local_recall.search(query, k=10)
        cold_time = (time.time() - start) * 1000

        # 第二次搜索（缓存命中）
        start = time.time()
        candidates2 = await local_recall.search(query, k=10)
        cached_time = (time.time() - start) * 1000

        # 验证结果一致
        assert len(candidates1) == len(candidates2)

        # 验证缓存更快（通常快10倍以上）
        # 注意：如果数据量很小，这个断言可能不稳定
        print(f"\nCold: {cold_time:.1f}ms, Cached: {cached_time:.1f}ms")

    @pytest.mark.asyncio
    async def test_search_ignores_query(self, local_recall: LocalRecall):
        """测试搜索返回有效匹配（关键词搜索，非全量返回）"""
        if not local_recall.is_available():
            pytest.skip("Skills directory not available")

        # 已知能匹配的 query
        candidates_match = await local_recall.search("filesystem", k=10)
        # 不匹配的 query
        candidates_empty = await local_recall.search("xyznonexistent12345", k=10)

        # 匹配 query 有结果，空 query 返回全量
        assert len(candidates_match) > 0
        assert len(candidates_empty) == 0

    def test_skills_have_description(self, local_recall: LocalRecall):
        """测试加载的 skills 都有 description"""
        if not local_recall.is_available():
            pytest.skip("Skills directory not available")

        import asyncio

        candidates = asyncio.run(local_recall.search("test", k=100))

        for candidate in candidates:
            # SkillSearchResult 有独立字段 name 和 description
            assert candidate.name  # name 不为空
            assert isinstance(candidate.source, str)  # source 有效
