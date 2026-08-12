# SPDX-License-Identifier: Apache-2.0
"""test_skill_executor_prompt.py - SkillAgent Prompt 构建测试

测试 SkillAgent 的 prompt 构建功能。
"""

import pytest
from pathlib import Path

from core.skill.execution import SkillAgent
from shared.schema import SkillConfig
from core.skill.schema import Skill


class TestSkillAgentPrompt:
    """SkillAgent Prompt 构建测试类"""

    @pytest.fixture
    def agent(self, skill_config):
        """创建 SkillAgent 实例"""
        return SkillAgent(skill_config)

    @pytest.fixture
    def sample_skill(self):
        """创建示例 skill"""
        return Skill(
            name="test_skill",
            description="A test skill",
            content="# Test Skill\n\nThis is test content.",
        )

    def test_build_messages_returns_list(self, agent, sample_skill):
        """测试 _build_messages 返回消息列表"""
        from core.skill.execution.state import ReActState

        state = ReActState(query="Test query", params={}, max_turns=30)
        workspace = Path("/tmp")

        messages = agent._build_messages(sample_skill, state, workspace)
        assert isinstance(messages, list)
        assert len(messages) > 0

    def test_platform_info_returns_string(self, agent):
        """测试 _platform_info 返回平台信息"""
        info = agent._platform_info()
        assert isinstance(info, str)
        assert len(info) > 0

    def test_get_skill_content_prefers_md(self, agent):
        """测试 _get_skill_content 优先读 SKILL.md"""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "SKILL.md").write_text("# Doc\nHello")
            skill = Skill(name="t", description="", content="fallback", source_dir=d)
            result = agent._get_skill_content(skill)
            assert "Doc" in result and "fallback" not in result

    def test_get_skill_content_fallback(self, agent):
        """测试 _get_skill_content fallback 到 content"""
        skill = Skill(name="t", description="", content="fallback text")
        assert agent._get_skill_content(skill) == "fallback text"

    def test_list_existing_scripts_with_scripts(self, agent):
        """测试 _list_existing_scripts 列出脚本"""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            scripts_dir = Path(d) / "scripts"
            scripts_dir.mkdir()
            (scripts_dir / "demo.py").write_text("print(1)")
            skill = Skill(name="t", description="", content="", source_dir=d)
            result = agent._list_existing_scripts(skill)
            assert "demo.py" in result

    def test_list_existing_scripts_no_dir(self, agent):
        """测试 _list_existing_scripts 无目录时返回 <none>"""
        skill = Skill(name="t", description="", content="")
        result = agent._list_existing_scripts(skill)
        assert "<none>" in result

    def test_get_real_file_tree_limited(self, agent):
        """测试 _get_real_file_tree_limited 返回文件树"""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "file.txt").write_text("hello")
            result = agent._get_real_file_tree_limited(Path(d))
            assert isinstance(result, str)
            assert "file.txt" in result

    def test_tool_category_mapping(self, agent):
        """测试 _tool_category 分类映射"""
        assert agent._tool_category("python_repl") == "code"
        assert agent._tool_category("bash") == "code"
        assert agent._tool_category("file_create") == "write"
        assert agent._tool_category("read_file") == "read"
        assert agent._tool_category("search_web") == "web"
        assert agent._tool_category("unknown") == "other"

    def test_extract_tool_call_parts_dict(self, agent):
        """测试 _extract_tool_call_parts 解析 dict 格式"""
        tool_call = {
            "id": "call_123",
            "function": {
                "name": "read_file",
                "arguments": '{"path": "/test.txt"}',
            },
        }
        name, args, call_id = agent._extract_tool_call_parts(tool_call)
        assert name == "read_file"
        assert args["path"] == "/test.txt"
        assert call_id == "call_123"

    def test_evaluate_task_signal_error(self, agent):
        """测试 _evaluate_task_signal 对错误的判断"""
        assert agent._evaluate_task_signal("bash", "ERR: something failed") == "none"
        assert agent._evaluate_task_signal("read_file", "error occurred") == "none"

    def test_evaluate_task_signal_strong(self, agent):
        """测试 _evaluate_task_signal 对强信号的判断"""
        assert agent._evaluate_task_signal("file_create", "file created") == "strong"
        assert agent._evaluate_task_signal("bash", "command succeeded") == "strong"

    def test_evaluate_task_signal_medium(self, agent):
        """测试 _evaluate_task_signal 对中等信号的判断"""
        assert agent._evaluate_task_signal("search_web", "found 5 results") == "medium"
        assert agent._evaluate_task_signal("read_file", "file content here") == "medium"
