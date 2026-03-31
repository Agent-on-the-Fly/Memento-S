"""SkillMarket install 接口集成测试

测试从云端安装 skill 的完整流程。
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from pathlib import Path

from core.skill.config import SkillConfig
from core.skill.market import SkillMarket
from core.skill.store import SkillStore
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
    """创建 SkillMarket 实例（使用真实 DB 和 Vector 存储）"""
    market = await SkillMarket.from_config(test_config)
    yield market
    await market._store.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_install_skill_full_flow(skill_market, test_config):
    """
    测试安装 skill 的完整流程

    验证：
    1. 成功从云端下载并安装 skill
    2. 磁盘文件存在且可访问
    3. Store 中可查询到 skill
    4. DB 中存在 skill 记录
    5. Vector 存储正常工作（如有 embedding 配置则生成向量）

    注意：卸载步骤需手动验证
    """
    skill_name = "feishu-doc"
    market = skill_market

    # ===== 步骤 1: 安装 skill =====
    print(f"\n>>> 开始安装 skill: {skill_name}")
    skill = await market.install(skill_name)
    assert skill is not None, f"安装 {skill_name} 失败"
    print(f"✅ Skill 安装成功: {skill.name}")

    normalized_name = skill.name
    skill_dir = (
        Path(skill.source_dir)
        if skill.source_dir
        else test_config.skills_dir / skill_name
    )
    storage_name = skill_dir.name

    # ===== 步骤 2: 验证磁盘文件 =====
    print(f"\n>>> 验证磁盘文件")
    print(f"   目录路径: {skill_dir}")
    assert skill_dir.exists(), f"Skill 目录不存在: {skill_dir}"
    assert (skill_dir / "SKILL.md").exists(), "SKILL.md 文件不存在"
    print(f"✅ 磁盘文件验证通过")

    # ===== 步骤 3: 验证 Store =====
    print(f"\n>>> 验证 Store")
    print(f"   存储名称: {storage_name}")
    cached_skill = await market._store.get_skill(storage_name)
    assert cached_skill is not None, f"Skill 不在 Store 中 (name={storage_name})"
    print(f"✅ Store 验证通过")

    # ===== 步骤 4: 验证 DB =====
    print(f"\n>>> 验证 DB")
    print(f"   规范化名称: {normalized_name}")
    db_skill = await market._store.db_storage.load(normalized_name)
    assert db_skill is not None, f"DB 中不存在 skill 记录 (name={normalized_name})"
    assert db_skill.name == normalized_name
    print(f"✅ DB 验证通过")

    # ===== 步骤 5: 验证 Vector =====
    print(f"\n>>> 验证 Vector")
    print(f"   Embedding 模型: {test_config.embedding_model}")
    print(f"   Vector ID (目录名): {storage_name}")
    vector = await market._store.vector_storage.load(storage_name)
    assert vector is not None, "Vector 未生成"
    assert len(vector) > 0, "Vector 为空"
    print(f"✅ Vector 验证通过，维度: {len(vector)}")

    # ===== 安装完成 =====
    print(f"\n{'=' * 50}")
    print(f"✅ 安装流程测试通过")
    print(f"{'=' * 50}")
    print(f"\n📋 安装信息:")
    print(f"   Skill 名称: {skill.name}")
    print(f"   目录路径: {skill_dir}")
    print(f"   统一 ID (目录名): {storage_name}")
    print(f"   DB 名称: {normalized_name}")
    print(f"   Vector ID: {storage_name}")
    print(f"\n⚠️  卸载步骤需手动验证")
    print(f"   如需卸载，请运行: await market.uninstall('{skill_name}')")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
