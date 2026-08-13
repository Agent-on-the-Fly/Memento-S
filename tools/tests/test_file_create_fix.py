# SPDX-License-Identifier: Apache-2.0
"""验证 file_create state_delta 修复的测试脚本。"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
import tempfile
import shutil

import pytest

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent))

from core.skill.execution.agent import SkillAgent
from core.skill.execution.state import ReActState
from core.skill.execution.artifact_registry import ArtifactRegistry
from shared.schema import SkillConfig


@pytest.mark.asyncio
async def test_file_create_state_delta():
    """测试 file_create 工具执行后 state_delta 和 state.created_files 的填充。"""
    
    # 创建临时 workspace
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        test_file = workspace / "test_notes.md"
        
        # 模拟 LLM response（file_create 成功）
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.has_tool_calls = False
        mock_response.text = "已完成！"
        mock_response.finish_reason = "stop"
        mock_llm.async_chat = AsyncMock(return_value=mock_response)
        
        # SkillAgent 配置
        config = MagicMock(spec=SkillConfig)
        config.workspace_dir = workspace
        config.llm_api_base = "http://test"
        config.llm_model = "test"
        config.llm_api_key = "test"
        
        agent = SkillAgent(config=config, llm=mock_llm)
        
        # 模拟 skill
        skill = MagicMock()
        skill.name = "test-skill"
        skill.content = "Test skill"
        skill.allowed_tools = None
        skill.is_playbook = False
        skill.execution_mode = None
        skill.source_dir = None
        
        # 执行 skill（模拟一个会触发 file_create 的场景）
        # 注意：由于 mock LLM 不返回 tool_calls，我们用另一种方式测试 _register_artifacts
        
        # 直接测试 _register_artifacts
        state = ReActState(query="test", params={}, max_turns=1)
        state.turn_count = 1
        state.artifact_registry = ArtifactRegistry()
        
        # 模拟 file_create 的 observation（state_delta 为空，符合 bug 场景）
        observation = {
            "tool": "file_create",
            "tool_call_id": "test-id",
            "summary": f"SUCCESS: Created file {test_file}",
            "exec_status": "success",
            "state_delta": {},  # ← 这是 bug：应该填充但为空
            "task_signal": "weak",
            "raw": {},
        }
        
        # tool_args（包含路径信息）
        tool_args = {"path": str(test_file), "content": "# Test", "overwrite": False}
        
        # 执行修复前的检查
        print("=== 修复前状态 ===")
        print(f"observation['state_delta']: {observation.get('state_delta')}")
        print(f"state.created_files: {list(state.created_files)}")
        
        # 执行 _register_artifacts（修复后）
        agent._register_artifacts(
            tool_name="file_create",
            observation=observation,
            state=state,
            tool_args=tool_args,
        )
        
        # 验证修复
        print("\n=== 修复后状态 ===")
        print(f"observation['state_delta']: {observation.get('state_delta')}")
        print(f"state.created_files: {list(state.created_files)}")
        
        # 断言
        assert observation.get("state_delta", {}) != {}, \
            "FAIL: observation['state_delta'] 仍然为空"
        assert "created_files" in observation.get("state_delta", {}), \
            "FAIL: state_delta 中没有 created_files 键"
        assert str(test_file) in observation["state_delta"].get("created_files", []), \
            "FAIL: created_files 中没有包含目标文件"
        assert str(test_file) in state.created_files, \
            "FAIL: state.created_files 中没有包含目标文件"
        
        # 验证 ArtifactRegistry 确实注册了
        assert state.artifact_registry.is_registered(str(test_file)), \
            "FAIL: ArtifactRegistry 没有注册该文件"
        
        print("\n✅ 所有断言通过！修复验证成功。")
        return True


if __name__ == "__main__":
    try:
        result = asyncio.run(test_file_create_state_delta())
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
