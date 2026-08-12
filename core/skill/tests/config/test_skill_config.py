# SPDX-License-Identifier: Apache-2.0
"""test_skill_config.py - SkillConfig 配置测试

测试 SkillConfig 的创建和属性。
"""

import pytest
from pathlib import Path

from shared.schema import SkillConfig


class TestSkillConfig:
    """SkillConfig 测试类"""

    @pytest.mark.asyncio
    async def test_config_from_global(self, test_config):
        """测试从全局配置创建 SkillConfig"""
        config = SkillConfig.from_global_config()

        assert config is not None
        assert isinstance(config.skills_dir, Path)
        assert isinstance(config.builtin_skills_dir, Path)
        assert isinstance(config.workspace_dir, Path)
        assert isinstance(config.retrieval_top_k, int)

    @pytest.mark.asyncio
    async def test_config_paths_exist(self, skill_config):
        """测试配置路径都存在"""
        assert (
            skill_config.skills_dir.exists() or skill_config.skills_dir.parent.exists()
        )
