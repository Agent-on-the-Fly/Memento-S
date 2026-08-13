# SPDX-License-Identifier: Apache-2.0
"""SkillMarket install 接口集成测试

测试从云端安装 skill 的完整流程。
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from dataclasses import replace
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
    """Build an isolated config so installation never touches user skills."""
    manager = ConfigManager()
    runtime = manager.load()
    base = SkillConfig(
        skills_dir=tmp_path / "skills",
        builtin_skills_dir=tmp_path / "builtin-skills",
        workspace_dir=tmp_path / "workspace",
        cloud_catalog_url=runtime.skills.cloud_catalog_url,
    )
    return base


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
async def test_install_skill_full_flow(skill_market, test_config):
    """
    测试安装 skill 的完整流程

    验证：
    1. 成功从云端下载并安装 skill
    2. 磁盘文件存在且可访问
    3. Store 中可查询到 skill

    注意：卸载步骤需手动验证
    """
    skill_name = "feishu-doc"
    market = skill_market

    # ===== 步骤 1: 安装 skill =====
    print(f"\n>>> 开始安装 skill: {skill_name}")
    skill = await market.install(skill_name)
    assert skill is not None, f"安装 {skill_name} 失败"
    print(f"Skill 安装成功: {skill.name}")

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
    print(f"磁盘文件验证通过")

    # ===== 步骤 3: 验证 Store =====
    print(f"\n>>> 验证 Store")
    print(f"   存储名称: {storage_name}")
    cached_skill = await market._store.get_skill(storage_name)
    assert cached_skill is not None, f"Skill 不在 Store 中 (name={storage_name})"
    print(f"Store 验证通过")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
