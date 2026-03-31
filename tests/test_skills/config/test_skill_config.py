"""test_skill_config.py - SkillConfig 配置测试

测试 SkillConfig 的创建和属性。
"""

import pytest
from pathlib import Path

from core.skill.config import SkillConfig
from middleware.config import g_config


class TestSkillConfig:
    """SkillConfig 测试类"""

    @pytest.mark.asyncio
    async def test_config_from_global(self, test_config):
        """测试从全局配置创建 SkillConfig"""
        # 确保 g_config 已加载
        from middleware.config import g_config

        if not g_config._config:
            g_config._config = test_config._config

        config = SkillConfig.from_global_config()

        assert config is not None
        assert isinstance(config.skills_dir, Path)
        assert isinstance(config.builtin_skills_dir, Path)
        assert isinstance(config.workspace_dir, Path)
        assert isinstance(config.db_path, Path)

    @pytest.mark.asyncio
    async def test_config_paths_exist(self, skill_config):
        """测试配置路径都存在"""
        assert (
            skill_config.skills_dir.exists() or skill_config.skills_dir.parent.exists()
        )
        assert (
            skill_config.builtin_skills_dir.exists()
            or skill_config.builtin_skills_dir.parent.exists()
        )

    @pytest.mark.asyncio
    async def test_config_cloud_catalog_url(self, skill_config):
        """测试云端 catalog URL 配置"""
        # cloud_catalog_url 可以是 str 或 None
        assert skill_config.cloud_catalog_url is None or isinstance(
            skill_config.cloud_catalog_url, str
        )

    @pytest.mark.asyncio
    async def test_config_retrieval_top_k(self, skill_config):
        """测试检索配置"""
        assert isinstance(skill_config.retrieval_top_k, int)
        assert skill_config.retrieval_top_k > 0

    @pytest.mark.asyncio
    async def test_config_pip_timeout(self, skill_config):
        """测试 pip 安装超时配置"""
        assert isinstance(skill_config.pip_install_timeout, int)
        assert skill_config.pip_install_timeout > 0

    @pytest.mark.asyncio
    async def test_config_path_validation(self, skill_config):
        """测试路径验证配置"""
        assert isinstance(skill_config.path_validation_enabled, bool)

    @pytest.mark.asyncio
    async def test_config_embedding_model(self, skill_config):
        """测试 embedding 模型配置"""
        assert isinstance(skill_config.embedding_model, str)
        assert len(skill_config.embedding_model) > 0

    @pytest.mark.asyncio
    async def test_config_is_frozen(self, skill_config):
        """测试配置是不可变的"""
        with pytest.raises(AttributeError):
            skill_config.skills_dir = Path("/tmp/test")

    @pytest.mark.asyncio
    async def test_config_manual_creation(self):
        """测试手动创建配置（用于测试）"""
        config = SkillConfig(
            skills_dir=Path("/tmp/skills"),
            builtin_skills_dir=Path("/tmp/builtin"),
            workspace_dir=Path("/tmp/workspace"),
            db_path=Path("/tmp/test.db"),
            embedding_base_url="http://localhost:8000",
            embedding_api_key="test-key",
            embedding_model="text-embedding-3-small",
        )

        assert config.skills_dir == Path("/tmp/skills")
        assert config.embedding_base_url == "http://localhost:8000"
