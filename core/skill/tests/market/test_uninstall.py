# SPDX-License-Identifier: Apache-2.0
"""SkillMarket uninstall 接口测试

测试使用 skill.name 卸载 skill，并验证所有存储都被清理。
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from pathlib import Path

from shared.schema import SkillConfig
from core.skill.market import SkillMarket
from middleware.config import ConfigManager
from middleware.config.skill_config_manager import SkillConfigManager
from core.skill.registry import SkillRegistry
from core.skill.store import SkillStorage
from core.skill.retrieval import RemoteRecall


@pytest.fixture
def test_config(tmp_path):
    """Build an isolated config so uninstall never touches user skills."""
    manager = ConfigManager()
    runtime = manager.load()
    return SkillConfig(
        skills_dir=tmp_path / "skills",
        builtin_skills_dir=tmp_path / "builtin-skills",
        workspace_dir=tmp_path / "workspace",
        cloud_catalog_url=runtime.skills.cloud_catalog_url,
    )


@pytest_asyncio.fixture
async def skill_market(test_config):
    """创建 SkillMarket 实例"""
    registry_manager = SkillConfigManager(
        user_path=test_config.workspace_dir / "skill.json"
    )
    store = SkillStorage(test_config.skills_dir, SkillRegistry(registry_manager))
    await store.init()
    remote = await RemoteRecall.from_config(test_config)
    market = SkillMarket(config=test_config, store=store, remote_recall=remote)
    yield market
    await market._store.close()
    if remote is not None:
        await remote.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_uninstall_skill(skill_market, test_config):
    """
    测试卸载 skill

    验证：
    1. 使用 skill.name 调用 uninstall
    2. 磁盘文件被删除
    3. Store 中无法查询
    """
    skill_name = "feishu-doc"
    market = skill_market

    # 先确保 skill 已安装
    print(f"\n>>> 准备卸载测试：先安装 skill: {skill_name}")
    skill = await market.install(skill_name)
    if skill is None:
        print(f"安装失败，尝试从磁盘加载已存在的 skill")
        skill_dir = test_config.skills_dir / skill_name
        if skill_dir.exists():
            from core.skill.loader import load_from_dir
            skill = load_from_dir(skill_dir)
        else:
            pytest.skip(f"无法安装或加载 skill: {skill_name}")

    normalized_name = skill.name
    skill_dir = (
        Path(skill.source_dir)
        if skill.source_dir
        else test_config.skills_dir / skill_name
    )
    storage_name = skill_dir.name

    print(f"   Skill 名称: {normalized_name}")
    print(f"   目录名: {storage_name}")
    print(f"   目录路径: {skill_dir}")

    # 验证安装成功
    assert skill_dir.exists(), f"Skill 目录不存在: {skill_dir}"
    print(f"Skill 已准备好，可以开始卸载测试")

    # ===== 步骤 1: 使用 skill.name 卸载 =====
    print(f"\n>>> 使用 skill.name 调用 uninstall: {normalized_name}")
    result = await market.uninstall(normalized_name)
    assert result is True, f"卸载失败 (skill.name: {normalized_name})"
    print(f"卸载调用成功")

    # ===== 步骤 2: 验证磁盘文件已删除 =====
    print(f"\n>>> 验证磁盘文件已删除")
    assert not skill_dir.exists(), f"卸载后文件仍然存在: {skill_dir}"
    print(f"磁盘文件已删除")

    # ===== 步骤 3: 验证 Store 中已删除 =====
    print(f"\n>>> 验证 Store 中已删除")
    cached_skill = await market._store.get_skill(storage_name)
    assert cached_skill is None, f"卸载后仍然在 Store 中 (name: {storage_name})"
    print(f"Store 已清理")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
