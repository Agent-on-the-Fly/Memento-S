"""test_local_db_recall.py — LocalDbRecall 单元测试

测试本地向量数据库召回功能。
依赖：sqlite-vec

用法：
    pytest tests/test_skills/retrieval/test_local_db_recall.py -v
"""

from __future__ import annotations

import pytest

from core.skill.retrieval import RecallCandidate


# 检查 sqlite-vec 是否可用，如果不可用则跳过整个模块
sqlite_vec = pytest.importorskip("sqlite_vec", reason="sqlite-vec not installed")

# 现在可以安全导入 LocalDbRecall
from core.skill.retrieval import LocalDbRecall


class TestLocalDbRecall:
    """LocalDbRecall 测试类"""

    def test_initialization(self, local_db_recall: LocalDbRecall, db_path):
        """测试初始化"""
        assert local_db_recall.name == "local_db"
        assert local_db_recall._db_path == db_path
        assert local_db_recall._embedding_client is None

    def test_is_available(self, local_db_recall: LocalDbRecall, db_path):
        """测试可用性检查"""
        available = local_db_recall.is_available()

        # 如果数据库文件存在，应该返回 True
        if db_path.exists():
            assert available is True
        else:
            # 数据库不存在时，连接会失败
            assert available is False

    def test_get_stats(self, local_db_recall: LocalDbRecall):
        """测试获取统计信息"""
        stats = local_db_recall.get_stats()

        assert "name" in stats
        assert "available" in stats
        assert "db_path" in stats
        assert stats["name"] == "local_db"
        assert isinstance(stats["available"], bool)

    @pytest.mark.asyncio
    async def test_search_without_embedding_client(
        self, local_db_recall: LocalDbRecall
    ):
        """测试无 embedding_client 时返回空列表"""
        if not local_db_recall.is_available():
            pytest.skip("Database not available")

        query = "test"
        candidates = await local_db_recall.search(query, k=5)

        # 没有 embedding_client 时应该返回空列表
        assert candidates == []

    @pytest.mark.asyncio
    async def test_search_returns_candidates_with_mock_embedding(
        self, local_db_recall: LocalDbRecall
    ):
        """测试使用 mock embedding client 搜索（需要数据库中有数据）"""
        if not local_db_recall.is_available():
            pytest.skip("Database not available")

        # 获取索引数量
        indexed_count = len(local_db_recall.get_indexed_names())

        # 如果没有索引数据，跳过
        if indexed_count == 0:
            pytest.skip("No indexed skills in database")

        # 创建一个 mock embedding client
        class MockEmbeddingClient:
            dimension = 384

            async def embed_query(self, query: str) -> list[float]:
                # 返回一个随机向量（实际测试应该用真实向量）
                import random

                return [random.random() for _ in range(self.dimension)]

        # 使用 mock client 重新创建实例
        import copy

        recall_with_mock = copy.copy(local_db_recall)
        recall_with_mock._embedding_client = MockEmbeddingClient()

        query = "test"
        candidates = await recall_with_mock.search(query, k=5)

        # 验证返回的是 RecallCandidate 列表
        assert isinstance(candidates, list)
        # 不验证具体数量，因为随机向量匹配结果不确定

    def test_get_indexed_names(self, local_db_recall: LocalDbRecall):
        """测试获取已索引的 skill names"""
        if not local_db_recall.is_available():
            pytest.skip("Database not available")

        names = local_db_recall.get_indexed_names()

        assert isinstance(names, set)
        # 不验证具体数量，取决于数据库状态
        print(f"\nIndexed skills: {len(names)}")
        if names:
            print(f"First few: {list(names)[:5]}")

    def test_close(self, local_db_recall: LocalDbRecall):
        """测试关闭连接"""
        if not local_db_recall.is_available():
            pytest.skip("Database not available")

        # 关闭前连接应该存在
        assert local_db_recall._conn is not None

        # 关闭
        local_db_recall.close()

        # 关闭后连接应该为 None
        assert local_db_recall._conn is None
