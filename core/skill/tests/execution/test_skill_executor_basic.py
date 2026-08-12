# SPDX-License-Identifier: Apache-2.0
"""test_skill_executor_basic.py - SkillAgent 基础测试

测试 SkillAgent 的基本功能和初始化。
"""

import pytest
from pathlib import Path

from core.skill.execution import SkillAgent
from shared.schema import SkillConfig
from core.skill.schema import Skill


class TestSkillAgentBasic:
    """SkillAgent 基础测试类"""

    @pytest.fixture
    def agent(self, skill_config):
        """创建 SkillAgent 实例"""
        return SkillAgent(skill_config)

    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """测试执行器初始化"""
        assert agent is not None
        assert agent._config is not None
        assert agent._llm is not None

    @pytest.mark.asyncio
    async def test_agent_with_config(self, skill_config):
        """测试使用配置创建执行器"""
        agent = SkillAgent(skill_config)
        assert agent._config == skill_config

    @pytest.mark.asyncio
    async def test_agent_policy_manager(self, agent):
        """测试执行器可接受 policy_manager"""
        agent_with_policy = SkillAgent(agent._config, policy_manager=agent._policy_manager)
        assert agent_with_policy._policy_manager is agent._policy_manager

    @pytest.mark.asyncio
    async def test_agent_skill_env_cache(self, agent):
        """测试 env var 缓存初始化"""
        assert agent._skill_env_cache is None


class TestSkillAgentToolSchemas:
    """SkillAgent 工具 schema 测试"""

    @pytest.fixture
    def agent(self, skill_config):
        """创建 SkillAgent 实例"""
        return SkillAgent(skill_config)

    @pytest.mark.asyncio
    async def test_get_tool_schemas_returns_list(self, agent):
        """测试 _get_tool_schemas 返回列表"""
        skill = Skill(name="test", description="")
        schemas = agent._get_tool_schemas(skill)
        assert isinstance(schemas, list)

    @pytest.mark.asyncio
    async def test_get_tool_schemas_empty_allowed(self, agent):
        """测试空 allowed_tools 不过滤"""
        skill = Skill(name="test", description="", allowed_tools=[])
        schemas = agent._get_tool_schemas(skill)
        assert isinstance(schemas, list)

    @pytest.mark.asyncio
    async def test_get_tool_schemas_with_allowed(self, agent):
        """测试带 allowed_tools 的过滤"""
        skill = Skill(name="test", description="", allowed_tools=["read_file"])
        schemas = agent._get_tool_schemas(skill)
        # Should return only read_file schema
        tool_names = [s.get("function", {}).get("name", "") for s in schemas]
        assert "read_file" in tool_names
