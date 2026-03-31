"""SkillMarket uninstall 接口测试

测试使用 skill.name 卸载 skill，并验证所有存储都被清理。
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from pathlib import Path

from core.skill.config import SkillConfig
from core.skill.market import SkillMarket
from middleware.config import ConfigManager, g_config


@pytest.fixture(scope="session")
def test_config():
    """加载测试配置"""
    if not g_config._config:
        config_manager = ConfigManager()
        config_manager.load()
        g_config._config = config_manager._config
    return SkillConfig.from_global_config()


@pytest_asyncio.fixture
async def skill_market(test_config):
    """创建 SkillMarket 实例"""
    market = await SkillMarket.from_config(test_config)
    yield market
    await market._store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_uninstall_skill(skill_market, test_config):
    """
    测试卸载 skill

    验证：
    1. 使用 skill.name 调用 uninstall
    2. 磁盘文件被删除
    3. Store 中无法查询
    4. DB 中记录被删除
    5. Vector 中向量被删除
    """
    skill_name = "feishu-doc"
    market = skill_market

    # 先确保 skill 已安装
    print(f"\n>>> 准备卸载测试：先安装 skill: {skill_name}")
    skill = await market.install(skill_name)
    if skill is None:
        print(f"⚠️  安装失败，尝试从磁盘加载已存在的 skill")
        skill_dir = test_config.skills_dir / skill_name
        if skill_dir.exists():
            # load_from_path 已经同步到 DB，不需要再调用 add_skill
            skill = await market._store.load_from_path(skill_dir)
        else:
            pytest.skip(f"无法安装或加载 skill: {skill_name}")

    normalized_name = skill.name  # "AgentDB Learning Plugins"
    skill_dir = (
        Path(skill.source_dir)
        if skill.source_dir
        else test_config.skills_dir / skill_name
    )
    storage_name = skill_dir.name  # "feishu-doc"

    print(f"   Skill 名称: {normalized_name}")
    print(f"   目录名: {storage_name}")
    print(f"   目录路径: {skill_dir}")

    # 验证安装成功
    assert skill_dir.exists(), f"Skill 目录不存在: {skill_dir}"
    assert await market._store.db_storage.load(normalized_name) is not None
    assert await market._store.vector_storage.load(storage_name) is not None
    print(f"✅ Skill 已准备好，可以开始卸载测试")

    # ===== 步骤 1: 使用 skill.name 卸载 =====
    print(f"\n>>> 使用 skill.name 调用 uninstall: {normalized_name}")
    result = await market.uninstall(normalized_name)
    assert result is True, f"卸载失败 (skill.name: {normalized_name})"
    print(f"✅ 卸载调用成功")

    # ===== 步骤 2: 验证磁盘文件已删除 =====
    print(f"\n>>> 验证磁盘文件已删除")
    assert not skill_dir.exists(), f"卸载后文件仍然存在: {skill_dir}"
    print(f"✅ 磁盘文件已删除")

    # ===== 步骤 3: 验证 Store 中已删除 =====
    print(f"\n>>> 验证 Store 中已删除")
    cached_skill = await market._store.get_skill(storage_name)
    assert cached_skill is None, f"卸载后仍然在 Store 中 (name: {storage_name})"
    print(f"✅ Store 已清理")

    # ===== 步骤 4: 验证 DB 中已删除 =====
    print(f"\n>>> 验证 DB 中已删除")
    db_skill = await market._store.db_storage.load(normalized_name)
    assert db_skill is None, f"卸载后 DB 中仍然存在 (name: {normalized_name})"
    print(f"✅ DB 已清理")

    # ===== 步骤 5: 验证 Vector 中已删除 =====
    print(f"\n>>> 验证 Vector 中已删除")
    vector = await market._store.vector_storage.load(storage_name)
    assert vector is None, f"卸载后 Vector 中仍然存在 (id: {storage_name})"
    print(f"✅ Vector 已清理")

    # ===== 卸载完成 =====
    print(f"\n{'=' * 50}")
    print(f"✅ 卸载测试通过")
    print(f"{'=' * 50}")
    print(f"\n📋 卸载信息:")
    print(f"   传入参数: skill.name = {normalized_name}")
    print(f"   实际删除: {storage_name}")
    print(f"   目录路径: {skill_dir}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
