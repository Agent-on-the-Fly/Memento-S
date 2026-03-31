"""db_storage.py 测试 - DBStorage"""

from __future__ import annotations

import pytest

from core.skill.store import DBStorage
from core.skill.schema import Skill


class TestDBStorage:
    """DBStorage 测试类"""

    @pytest.mark.asyncio
    async def test_init_with_service(self, skill_service):
        """测试有 service 时的初始化"""
        if not skill_service:
            pytest.skip("Skill service not available")

        storage = DBStorage(skill_service)
        await storage.init()

        assert storage.is_ready is True

        await storage.close()

    @pytest.mark.asyncio
    async def test_init_without_service(self):
        """测试无 service 时的初始化"""
        storage = DBStorage(None)
        await storage.init()

        assert storage.is_ready is False

        await storage.close()

    @pytest.mark.asyncio
    async def test_save_and_load(self, db_storage, sample_skill):
        """测试保存和加载"""
        if not db_storage.is_ready:
            pytest.skip("DB storage not ready")

        # 保存
        await db_storage.save(sample_skill.name, sample_skill)

        # 加载
        loaded = await db_storage.load(sample_skill.name)

        assert loaded is not None
        assert loaded.name == sample_skill.name

    @pytest.mark.asyncio
    async def test_save_with_embedding(self, db_storage, sample_skill):
        """测试带 embedding 的保存"""
        if not db_storage.is_ready:
            pytest.skip("DB storage not ready")

        # 创建模拟 embedding
        embedding = b"\x00\x01\x02\x03"

        # 保存
        await db_storage.save(sample_skill.name, sample_skill, embedding=embedding)

        # 验证保存成功（通过加载）
        loaded = await db_storage.load(sample_skill.name)
        assert loaded is not None

    @pytest.mark.asyncio
    async def test_delete(self, db_storage, sample_skill):
        """测试删除"""
        if not db_storage.is_ready:
            pytest.skip("DB storage not ready")

        # 保存并删除
        await db_storage.save(sample_skill.name, sample_skill)
        result = await db_storage.delete(sample_skill.name)

        assert result is True
        assert await db_storage.load(sample_skill.name) is None

    @pytest.mark.asyncio
    async def test_list_names(self, db_storage, sample_skill, sample_skill2):
        """测试列出所有名称"""
        if not db_storage.is_ready:
            pytest.skip("DB storage not ready")

        # 保存两个
        await db_storage.save(sample_skill.name, sample_skill)
        await db_storage.save(sample_skill2.name, sample_skill2)

        names = await db_storage.list_names()

        assert sample_skill.name in names
        assert sample_skill2.name in names

    @pytest.mark.asyncio
    async def test_close(self, skill_service):
        """测试关闭"""
        if not skill_service:
            pytest.skip("Skill service not available")

        storage = DBStorage(skill_service)
        await storage.init()
        await storage.close()
        # 不应抛出异常


class TestDBStorageWithoutDB:
    """无数据库时的测试"""

    @pytest.mark.asyncio
    async def test_operations_without_db(self):
        """测试无数据库时的操作"""
        storage = DBStorage(None)
        await storage.init()

        skill = Skill(name="test", description="Test", content="# Test")

        # 所有操作应静默失败
        await storage.save("test", skill)
        assert await storage.load("test") is None
        assert await storage.delete("test") is False
        assert await storage.list_names() == set()

        await storage.close()
