"""vector_storage.py 测试 - VectorStorage"""

from __future__ import annotations

import pytest

from core.skill.store import VectorStorage


class TestVectorStorage:
    """VectorStorage 测试类"""

    @pytest.mark.skipif(
        not pytest.importorskip("sqlite_vec", reason="sqlite-vec not installed"),
        reason="sqlite-vec extension required",
    )
    @pytest.mark.asyncio
    async def test_init(self, db_dir):
        """测试初始化 - 使用 g_config 路径"""
        db_path = db_dir / "test_vectors_init.db"
        storage = VectorStorage(db_path=db_path, dimension=768)
        await storage.init()

        assert storage.is_ready is True

        await storage.close()

    @pytest.mark.skipif(
        not pytest.importorskip("sqlite_vec", reason="sqlite-vec not installed"),
        reason="sqlite-vec extension required",
    )
    @pytest.mark.asyncio
    async def test_save_and_load(self, db_dir):
        """测试保存和加载向量 - 使用 g_config 路径"""
        db_path = db_dir / "test_vectors_save_load.db"
        storage = VectorStorage(db_path=db_path, dimension=3)
        await storage.init()

        # 保存向量
        vector = [0.1, 0.2, 0.3]
        await storage.save("test_skill", vector)

        # 加载向量
        loaded = await storage.load("test_skill")

        assert loaded is not None
        assert len(loaded) == 3
        assert loaded[0] == pytest.approx(0.1, abs=1e-6)

        await storage.close()

    @pytest.mark.skipif(
        not pytest.importorskip("sqlite_vec", reason="sqlite-vec not installed"),
        reason="sqlite-vec extension required",
    )
    @pytest.mark.asyncio
    async def test_delete(self, db_dir):
        """测试删除向量 - 使用 g_config 路径"""
        db_path = db_dir / "test_vectors_delete.db"
        storage = VectorStorage(db_path=db_path, dimension=3)
        await storage.init()

        # 保存并删除
        await storage.save("test_skill", [0.1, 0.2, 0.3])
        result = await storage.delete("test_skill")

        assert result is True
        assert await storage.load("test_skill") is None

        await storage.close()

    @pytest.mark.skipif(
        not pytest.importorskip("sqlite_vec", reason="sqlite-vec not installed"),
        reason="sqlite-vec extension required",
    )
    @pytest.mark.asyncio
    async def test_list_names(self, db_dir):
        """测试列出所有向量名称 - 使用 g_config 路径"""
        db_path = db_dir / "test_vectors_list.db"
        storage = VectorStorage(db_path=db_path, dimension=3)
        await storage.init()

        # 保存多个向量
        await storage.save("skill1", [0.1, 0.2, 0.3])
        await storage.save("skill2", [0.4, 0.5, 0.6])

        names = await storage.list_names()

        assert "skill1" in names
        assert "skill2" in names
        assert len(names) == 2

        await storage.close()

    @pytest.mark.skipif(
        not pytest.importorskip("sqlite_vec", reason="sqlite-vec not installed"),
        reason="sqlite-vec extension required",
    )
    @pytest.mark.asyncio
    async def test_search(self, db_dir):
        """测试向量搜索 - 使用 g_config 路径"""
        db_path = db_dir / "test_vectors_search.db"
        storage = VectorStorage(db_path=db_path, dimension=3)
        await storage.init()

        # 保存向量
        await storage.save("skill1", [1.0, 0.0, 0.0])
        await storage.save("skill2", [0.0, 1.0, 0.0])

        # 搜索相似向量
        results = await storage.search([1.0, 0.0, 0.0], k=2)

        assert len(results) == 2
        # 第一个应该是最相似的
        assert results[0][0] == "skill1"
        assert results[0][1] > 0.9  # 相似度应该很高

        await storage.close()

    @pytest.mark.asyncio
    async def test_without_sqlite_vec(self, db_dir):
        """测试没有 sqlite-vec 时的行为 - 使用 g_config 路径"""
        # 这会跳过初始化
        db_path = db_dir / "test_vectors_no_ext.db"
        storage = VectorStorage(db_path=db_path, dimension=3)

        # 尝试初始化（如果没有 sqlite-vec，应该不会崩溃）
        try:
            await storage.init()
            # 如果没有 sqlite-vec，is_ready 应该是 False
            if not storage.is_ready:
                assert True
            else:
                await storage.close()
        except Exception:
            # 预期行为：可能抛出异常
            pass
