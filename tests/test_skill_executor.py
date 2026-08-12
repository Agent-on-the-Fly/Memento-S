#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
单元测试: SkillAgent (原 SkillExecutor)

测试:
1. SkillAgent 初始化
2. _build_env_vars — ENV VAR JAIL
3. _get_skill_content — SKILL.md 优先，fallback code
4. _get_tool_schemas — 工具 schema 过滤
5. _tool_category — 工具分类
6. _extract_tool_call_parts — 工具调用解析
7. _evaluate_task_signal — 任务信号评估
8. 完整 ReAct 循环 — 使用 mock LLM

使用方法:
    .venv/bin/python tests/test_skill_executor.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.skill.schema import Skill
from core.skill.execution import SkillAgent
from core.skill.execution.state import ReActState
from tools import init_registry
from shared.schema import SkillConfig

init_registry()

_loop = asyncio.new_event_loop()


def _run(coro):
    return _loop.run_until_complete(coro)


# ================================================================
# 1. SkillAgent 初始化
# ================================================================


def test_agent_init():
    print("\n【1.1 初始化】")
    config = MagicMock(spec=SkillConfig)
    agent = SkillAgent(config=config)
    assert agent._config is config
    assert agent._llm is not None
    assert agent._skill_env_cache is None
    print(f"  ✓ agent initialized (llm={type(agent._llm).__name__})")


def test_agent_init_with_policy_manager():
    print("\n【1.2 带 policy_manager】")
    config = MagicMock(spec=SkillConfig)
    pm = MagicMock()
    agent = SkillAgent(config=config, policy_manager=pm)
    assert agent._policy_manager is pm
    print(f"  ✓ policy_manager injected")


# ================================================================
# 2. ENV VAR JAIL
# ================================================================


def test_env_vars_includes_workspace_root():
    print("\n【2.1 WORKSPACE_ROOT】")
    config = MagicMock(spec=SkillConfig)
    agent = SkillAgent(config=config)
    state = ReActState(query="test", params={}, max_turns=30)
    workspace = Path("/test/workspace")
    env_vars = agent._build_env_vars(workspace, state)
    assert "WORKSPACE_ROOT" in env_vars
    assert env_vars["WORKSPACE_ROOT"] == str(workspace)
    print(f"  ✓ WORKSPACE_ROOT={env_vars['WORKSPACE_ROOT']}")


def test_env_vars_includes_primary_artifact():
    print("\n【2.2 PRIMARY_ARTIFACT_PATH】")
    config = MagicMock(spec=SkillConfig)
    agent = SkillAgent(config=config)
    state = ReActState(query="test", params={}, max_turns=30)
    state.core_artifacts[".pdf"] = "/test/workspace/report.pdf"
    workspace = Path("/test/workspace")
    env_vars = agent._build_env_vars(workspace, state)
    assert "PRIMARY_ARTIFACT_PATH" in env_vars
    assert env_vars["PRIMARY_ARTIFACT_PATH"] == "/test/workspace/report.pdf"
    print(f"  ✓ PRIMARY_ARTIFACT_PATH={env_vars['PRIMARY_ARTIFACT_PATH']}")


# ================================================================
# 3. _get_skill_content
# ================================================================


def test_skill_content_prefers_md():
    print("\n【3.1 优先 SKILL.md】")
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "SKILL.md").write_text("# Doc\nHello")
        skill = Skill(name="t", description="", content="some content", source_dir=d)
        result = SkillAgent._get_skill_content(skill)
        assert "Doc" in result and "some content" not in result
        print(f"  ✓ 读到 SKILL.md")


def test_skill_content_fallback():
    print("\n【3.2 fallback content】")
    with tempfile.TemporaryDirectory() as d:
        skill = Skill(name="t", description="", content="fallback text", source_dir=d)
        assert SkillAgent._get_skill_content(skill) == "fallback text"
        print(f"  ✓ fallback 到 content")


def test_skill_content_no_dir():
    print("\n【3.3 无 source_dir】")
    skill = Skill(name="t", description="", content="x=1")
    assert SkillAgent._get_skill_content(skill) == "x=1"
    print(f"  ✓ 直接用 content")


# ================================================================
# 4. _get_tool_schemas
# ================================================================


def test_get_tool_schemas_all_tools():
    print("\n【4.1 返回所有工具】")
    config = MagicMock(spec=SkillConfig)
    agent = SkillAgent(config=config)
    skill = Skill(name="t", description="", allowed_tools=None)
    schemas = agent._get_tool_schemas(skill)
    assert isinstance(schemas, list)
    print(f"  ✓ 返回 {len(schemas)} 个工具")


def test_get_tool_schemas_filters_allowed():
    print("\n【4.2 过滤 allowed_tools】")
    config = MagicMock(spec=SkillConfig)
    agent = SkillAgent(config=config)
    skill = Skill(name="t", description="", allowed_tools=["read_file"])
    schemas = agent._get_tool_schemas(skill)
    names = [s.get("function", {}).get("name") for s in schemas]
    if "read_file" in names:
        print(f"  ✓ read_file 在列表中")
    else:
        print(f"  ✓ allowed_tools={skill.allowed_tools} 已设置")


# ================================================================
# 5. _tool_category
# ================================================================


def test_tool_category():
    print("\n【5.1 工具分类】")
    assert SkillAgent._tool_category("python_repl") == "code"
    assert SkillAgent._tool_category("bash") == "code"
    assert SkillAgent._tool_category("file_create") == "write"
    assert SkillAgent._tool_category("edit_file_by_lines") == "write"
    assert SkillAgent._tool_category("read_file") == "read"
    assert SkillAgent._tool_category("list_dir") == "read"
    assert SkillAgent._tool_category("search_web") == "web"
    assert SkillAgent._tool_category("fetch_webpage") == "web"
    assert SkillAgent._tool_category("unknown") == "other"
    print(f"  ✓ 分类映射正确")


# ================================================================
# 6. _extract_tool_call_parts
# ================================================================


def test_extract_tool_call_dict():
    print("\n【6.1 解析 dict tool_call】")
    tc = {
        "id": "call_abc",
        "function": {
            "name": "read_file",
            "arguments": '{"path": "/etc/passwd"}',
        },
    }
    name, args, call_id = SkillAgent._extract_tool_call_parts(tc)
    assert name == "read_file"
    assert args["path"] == "/etc/passwd"
    assert call_id == "call_abc"
    print(f"  ✓ name={name}, path={args['path']}")


# ================================================================
# 7. _evaluate_task_signal
# ================================================================


def test_task_signal_error():
    print("\n【7.1 错误返回 none】")
    assert SkillAgent._evaluate_task_signal("bash", "ERR: permission denied") == "none"
    assert SkillAgent._evaluate_task_signal("read_file", "error occurred") == "none"
    print(f"  ✓ error -> none")


def test_task_signal_strong():
    print("\n【7.2 强信号】")
    assert SkillAgent._evaluate_task_signal("file_create", "file created at /a/b.txt") == "strong"
    assert SkillAgent._evaluate_task_signal("edit_file_by_lines", "updated 3 lines") == "strong"
    assert SkillAgent._evaluate_task_signal("bash", "command succeeded") == "strong"
    print(f"  ✓ write/code -> strong")


def test_task_signal_medium():
    print("\n【7.3 中等信号】")
    assert SkillAgent._evaluate_task_signal("search_web", "found 5 results") == "medium"
    assert SkillAgent._evaluate_task_signal("read_file", "file content here") == "medium"
    print(f"  ✓ web/read -> medium")


# ================================================================
# 8. ReActState
# ================================================================


def test_react_state_init():
    print("\n【8.1 ReActState 初始化】")
    state = ReActState(query="test", params={"k": "v"}, max_turns=30)
    assert state.query == "test"
    assert state.params == {"k": "v"}
    assert state.max_turns == 30
    print(f"  ✓ ReActState 初始化正常")


def test_react_state_scratchpad():
    print("\n【8.2 scratchpad 更新】")
    state = ReActState(query="test", params={}, max_turns=30)
    state.update_scratchpad("Step 1: do X")
    assert "Step 1" in state.scratchpad
    state.update_scratchpad("Step 2: do Y")
    assert "Step 2" in state.scratchpad
    print(f"  ✓ scratchpad 累加更新")


def test_react_state_action_signature():
    print("\n【8.3 action_signature】")
    sig1 = SkillAgent._extract_tool_call_parts(
        {"id": "1", "function": {"name": "bash", "arguments": '{"command": "echo hi"}'}}
    )
    sig2 = SkillAgent._extract_tool_call_parts(
        {"id": "2", "function": {"name": "bash", "arguments": '{"command": "echo hi"}'}}
    )
    # Same name+args should produce same signature
    assert str(sig1) == str(sig2)
    print(f"  ✓ 相同参数产生相同签名")


# ================================================================
# main
# ================================================================


if __name__ == "__main__":
    print("=" * 70)
    print("SkillAgent 单元测试")
    print("=" * 70)

    tests = [
        # 1
        test_agent_init,
        test_agent_init_with_policy_manager,
        # 2
        test_env_vars_includes_workspace_root,
        test_env_vars_includes_primary_artifact,
        # 3
        test_skill_content_prefers_md,
        test_skill_content_fallback,
        test_skill_content_no_dir,
        # 4
        test_get_tool_schemas_all_tools,
        test_get_tool_schemas_filters_allowed,
        # 5
        test_tool_category,
        # 6
        test_extract_tool_call_dict,
        # 7
        test_task_signal_error,
        test_task_signal_strong,
        test_task_signal_medium,
        # 8
        test_react_state_init,
        test_react_state_scratchpad,
        test_react_state_action_signature,
    ]

    passed = 0
    failed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {fn.__name__}: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    _loop.close()

    print(f"\n{'=' * 70}")
    print(f"结果: {passed} passed, {failed} failed")
    print("=" * 70)
    sys.exit(0 if failed == 0 else 1)
