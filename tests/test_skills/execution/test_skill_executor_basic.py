"""test_skill_executor_basic.py - SkillExecutor 基础测试

测试 SkillExecutor 的基本功能和初始化。
"""

import pytest
from pathlib import Path

from core.skill.execution import SkillExecutor
from core.skill.config import SkillConfig
from core.skill.schema import Skill


class TestSkillExecutorBasic:
    """SkillExecutor 基础测试类"""

    @pytest.fixture
    def executor(self, skill_config):
        """创建 SkillExecutor 实例"""
        return SkillExecutor(skill_config)

    @pytest.mark.asyncio
    async def test_executor_initialization(self, executor):
        """测试执行器初始化"""
        assert executor is not None
        assert executor._config is not None
        assert executor._llm is not None

    @pytest.mark.asyncio
    async def test_executor_with_config(self, skill_config):
        """测试使用配置创建执行器"""
        executor = SkillExecutor(skill_config)
        assert executor._config == skill_config

    @pytest.mark.asyncio
    async def test_executor_sandbox(self, executor):
        """测试执行器有 sandbox"""
        assert executor._sandbox is not None

    @pytest.mark.asyncio
    async def test_executor_policy_manager(self, executor):
        """测试执行器有策略管理器"""
        assert executor._policy_manager is not None


class TestSkillExecutorFilterTools:
    """SkillExecutor 工具过滤测试"""

    @pytest.fixture
    def executor(self, skill_config):
        """创建 SkillExecutor 实例"""
        return SkillExecutor(skill_config)

    @pytest.mark.asyncio
    async def test_filter_tools_empty_allowed_list(self, executor):
        """测试空 allowed_tools 列表不过滤"""
        from builtin.tools.registry import BUILTIN_TOOL_SCHEMAS

        filtered = executor._filter_tools_by_allowed_list(BUILTIN_TOOL_SCHEMAS, None)
        # None 或空列表不过滤
        assert len(filtered) == len(BUILTIN_TOOL_SCHEMAS)

    @pytest.mark.asyncio
    async def test_filter_tools_with_allowed_list(self, executor):
        """测试带允许列表的工具过滤"""
        # 使用 OpenAI 风格的工具格式
        tools = [
            {
                "type": "function",
                "function": {"name": "tool1", "description": "Tool 1"},
            },
            {
                "type": "function",
                "function": {"name": "tool2", "description": "Tool 2"},
            },
            {
                "type": "function",
                "function": {"name": "tool3", "description": "Tool 3"},
            },
        ]

        filtered = executor._filter_tools_by_allowed_list(tools, ["tool1", "tool2"])

        assert len(filtered) == 2
        assert all(t["function"]["name"] in ["tool1", "tool2"] for t in filtered)

    @pytest.mark.asyncio
    async def test_filter_tools_all_allowed(self, executor):
        """测试所有工具都被允许"""
        tools = [
            {
                "type": "function",
                "function": {"name": "tool1", "description": "Tool 1"},
            },
        ]

        filtered = executor._filter_tools_by_allowed_list(tools, ["tool1"])

        assert len(filtered) == 1
        assert filtered[0]["function"]["name"] == "tool1"
