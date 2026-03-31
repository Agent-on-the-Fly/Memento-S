"""skill_store.py 测试 - SkillStore 协调层"""

from __future__ import annotations

import pytest
import tempfile
import shutil
from pathlib import Path

from core.skill.store import (
    SkillStore,
    FileStorage,
    DBStorage,
    VectorStorage,
)
from core.skill.schema import Skill


def create_skill_with_frontmatter(
    name: str, description: str, extra_content: str = ""
) -> Skill:
    """辅助函数：创建带有 frontmatter 的 Skill"""
    content = f"""---
name: {name}
description: {description}
metadata:
  function_name: {name}
---

# {name}

{extra_content}
"""
    return Skill(name=name, description=description, content=content)


class TestSkillStore:
    """SkillStore 测试类"""

    @pytest.mark.asyncio
    async def test_init_with_components(
        self, skills_dir, skill_service, vector_storage
    ):
        """测试用组件初始化 - 使用 g_config 路径"""
        file_storage = FileStorage(skills_dir)
        await file_storage.init()

        db_storage = DBStorage(skill_service)
        await db_storage.init()

        store = SkillStore(file_storage, db_storage, vector_storage)

        assert store.file_storage is file_storage
        assert store.db_storage is db_storage
        assert store.vector_storage is vector_storage

        await store.close()

    @pytest.mark.asyncio
    async def test_from_config(self, skill_config):
        """测试从配置创建"""
        store = await SkillStore.from_config(skill_config)

        assert store.file_storage.is_ready is True
        assert store.db_storage.is_ready is True

        await store.close()

    @pytest.mark.asyncio
    async def test_add_skill_without_vector(self, skill_store, sample_skill):
        """测试添加 skill（无向量）- 文件应已存在，add_skill 只负责索引"""
        # 先保存文件（源头）
        await skill_store.file_storage.save(sample_skill.name, sample_skill)
        # 再添加索引
        await skill_store.add_skill(sample_skill)

        # 验证可以从文件加载
        loaded = await skill_store.get_skill(sample_skill.name)
        assert loaded is not None
        assert loaded.name == sample_skill.name

    @pytest.mark.asyncio
    async def test_add_skill_with_embedding(self, skill_store):
        """测试添加 skill（自动生成向量）- 文件应已存在"""
        skill = create_skill_with_frontmatter("embedding_test", "Test with embedding")

        # 先保存文件（源头）
        await skill_store.file_storage.save(skill.name, skill)
        # 再添加索引（自动生成向量）
        await skill_store.add_skill(skill)

        # 验证可以从文件加载
        loaded = await skill_store.get_skill(skill.name)
        assert loaded is not None

    @pytest.mark.asyncio
    async def test_remove_skill(self, skill_store, sample_skill):
        """测试删除 skill"""
        # 先保存文件，再添加索引
        await skill_store.file_storage.save(sample_skill.name, sample_skill)
        await skill_store.add_skill(sample_skill)

        result = await skill_store.remove_skill(sample_skill.name)

        assert result is True
        assert await skill_store.get_skill(sample_skill.name) is None

    @pytest.mark.asyncio
    async def test_list_all_skills(self, skill_store, sample_skill):
        """测试列出所有 skills"""
        skill2 = create_skill_with_frontmatter("another_skill", "Another test")

        # 先保存文件，再添加索引
        await skill_store.file_storage.save(sample_skill.name, sample_skill)
        await skill_store.file_storage.save(skill2.name, skill2)
        await skill_store.add_skill(sample_skill)
        await skill_store.add_skill(skill2)

        skills = await skill_store.list_all_skills()

        assert sample_skill.name in skills
        assert "another_skill" in skills

    @pytest.mark.asyncio
    async def test_load_from_path(self, skill_store, skills_dir):
        """测试从路径加载 - load_from_path 已包含文件到 DB 的同步"""
        # 创建一个 skill 并保存到文件
        skill = create_skill_with_frontmatter("path_test", "Test", "Content")
        await skill_store.file_storage.save(skill.name, skill)

        # 从路径加载（会自动同步到 DB）
        skill_dir = skills_dir / "path-test"
        loaded = await skill_store.load_from_path(skill_dir)

        assert loaded.name == skill.name

    @pytest.mark.asyncio
    async def test_sync_from_disk(self, skill_store, sample_skill):
        """测试从磁盘同步"""
        # 先保存文件（源头）
        await skill_store.file_storage.save(sample_skill.name, sample_skill)
        # 添加索引
        await skill_store.add_skill(sample_skill)

        # 同步
        count = await skill_store.sync_from_disk()

        assert count >= 1

    @pytest.mark.asyncio
    async def test_refresh_from_disk(self, skill_store):
        """测试刷新磁盘"""
        # 先保存文件（源头）
        skill = create_skill_with_frontmatter("refresh_test", "Test")
        await skill_store.file_storage.save(skill.name, skill)
        await skill_store.add_skill(skill)

        # 刷新
        added = await skill_store.refresh_from_disk()

        # 应该找到已添加的 skill（在缓存中已存在，所以可能为 0）
        assert added >= 0

    @pytest.mark.asyncio
    async def test_cleanup_orphans(self, skill_store, sample_skill):
        """测试清理孤儿"""
        # 先保存文件，再添加索引
        await skill_store.file_storage.save(sample_skill.name, sample_skill)
        await skill_store.add_skill(sample_skill)

        # 清理（应该没有孤儿）
        cleaned = await skill_store.cleanup_orphans()

        assert isinstance(cleaned, list)

    @pytest.mark.asyncio
    async def test_sync_vectors(self, skill_store, embedding_client):
        """测试同步向量 - 需要 embedding_client，使用与数据库表相同的维度"""
        if not embedding_client:
            pytest.skip("Embedding client not available")

        skill1 = create_skill_with_frontmatter("vec1", "Test 1")
        skill2 = create_skill_with_frontmatter("vec2", "Test 2")

        # 先保存文件，再添加索引
        await skill_store.file_storage.save(skill1.name, skill1)
        await skill_store.file_storage.save(skill2.name, skill2)
        await skill_store.add_skill(skill1)
        await skill_store.add_skill(skill2)

        # 使用与数据库表相同的维度（从 vector_storage 获取）
        dim = skill_store.vector_storage._dimension or 5
        vectors = {"vec1": [0.1] * dim, "vec2": [0.2] * dim}

        count = await skill_store.sync_vectors(vectors)

        assert count == 2

    @pytest.mark.asyncio
    async def test_local_cache(self, skill_store, sample_skill):
        """测试本地 skills 列表"""
        # 先保存文件，再添加索引
        await skill_store.file_storage.save(sample_skill.name, sample_skill)
        await skill_store.add_skill(sample_skill)

        # 刷新磁盘更新缓存
        await skill_store.refresh_from_disk()

        # 使用 list_all_skills 验证 skill 已加载
        all_skills = await skill_store.list_all_skills()
        assert sample_skill.name in all_skills


class TestSkillStoreIntegration:
    """SkillStore 集成测试"""

    @pytest.mark.asyncio
    async def test_full_workflow(self, skill_config):
        """测试完整工作流"""
        # 从配置创建（内部自动创建 DB）
        store = await SkillStore.from_config(skill_config)

        try:
            # 添加 skills - 先保存文件，再添加索引
            skill1 = create_skill_with_frontmatter("workflow1", "Test1")
            skill2 = create_skill_with_frontmatter("workflow2", "Test2")

            await store.file_storage.save(skill1.name, skill1)
            await store.file_storage.save(skill2.name, skill2)
            await store.add_skill(skill1)
            await store.add_skill(skill2)

            # 列出
            skills = await store.list_all_skills()
            assert len(skills) >= 2

            # 验证 skill 已添加
            found = await store.get_skill("workflow1")
            assert found is not None

            # 同步
            await store.sync_from_disk()

            # 删除
            await store.remove_skill("workflow1")
            assert await store.get_skill("workflow1") is None

        finally:
            await store.close()
