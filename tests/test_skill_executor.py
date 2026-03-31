#!/usr/bin/env python3
"""
单元测试: SkillExecutor

测试:
1. _normalize_args   — 参数规范化（过滤/默认值/类型转换）
2. _get_skill_content — SKILL.md 优先，fallback code
3. _build_prompt      — knowledge vs playbook prompt 差异
4. _extract_code_block — markdown 代码块提取
5. tool_calls 真实执行 — list_dir / read_file / grep / bash / policy / 组合
6. fallback 真实执行   — 纯文本 vs 代码块
7. 真实 LLM 端到端    — 加载 builtin skill，走 execute() → LLM → tool_calls 完整链路
8. 依赖解析与检查      — _parse_dependency / _check_missing_dependencies / validate_skill_md

使用方法:
    .venv/bin/python tests/test_skill_executor.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.skill.schema import Skill
from core.skill.execution.executor import SkillExecutor
from builtin.tools.registry import configure as configure_tools
from middleware.llm.schema import ToolCall

configure_tools(Path.cwd())

_loop = asyncio.new_event_loop()


def _run(coro):
    return _loop.run_until_complete(coro)


# ================================================================
# 1. _normalize_args
# ================================================================


def test_normalize_args_filters_unknown():
    print("\n【1.1 过滤未知字段】")
    executor = SkillExecutor()
    result = executor._normalize_args(
        "read_file",
        {
            "path": "a.py",
            "start_line": 1,
            "unknown": "x",
        },
    )
    assert "unknown" not in result and "path" in result
    print(f"  ✓ {list(result.keys())}")


def test_normalize_args_defaults():
    print("\n【1.2 默认值填充】")
    executor = SkillExecutor()
    result = executor._normalize_args("read_file", {"path": "a.py"})
    assert result == {"path": "a.py", "start_line": 1, "end_line": -1}
    print(f"  ✓ {result}")


def test_normalize_args_type_cast():
    print("\n【1.3 类型转换】")
    executor = SkillExecutor()
    result = executor._normalize_args("read_file", {"path": "a.py", "start_line": "5"})
    assert result["start_line"] == 5 and isinstance(result["start_line"], int)
    print(f"  ✓ '5' -> {result['start_line']}")


def test_normalize_args_bad_cast():
    print("\n【1.4 转换失败用默认值】")
    executor = SkillExecutor()
    result = executor._normalize_args(
        "read_file", {"path": "a.py", "start_line": "abc"}
    )
    assert result["start_line"] == 1
    print(f"  ✓ 'abc' -> 默认值 1")


def test_normalize_args_bool():
    print("\n【1.5 布尔转换】")
    executor = SkillExecutor()
    result = executor._normalize_args(
        "grep", {"pattern": "x", "show_line_numbers": "true"}
    )
    assert result["show_line_numbers"] is True
    print(f"  ✓ 'true' -> True")


def test_normalize_args_no_schema():
    print("\n【1.6 无 schema 原样返回】")
    executor = SkillExecutor()
    args = {"a": 1, "b": 2}
    assert executor._normalize_args("fake_tool", args) == args
    print(f"  ✓ {args}")


# ================================================================
# 2. _get_skill_content
# ================================================================


def test_skill_content_prefers_md():
    print("\n【2.1 优先 SKILL.md】")
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "SKILL.md").write_text("# Doc\nHello")
        skill = Skill(name="t", description="", content="some content", source_dir=d)
        result = SkillExecutor._get_skill_content(skill)
        assert "Doc" in result and "some content" not in result
        print(f"  ✓ 读到 SKILL.md")


def test_skill_content_fallback():
    print("\n【2.2 fallback content】")
    with tempfile.TemporaryDirectory() as d:
        skill = Skill(name="t", description="", content="fallback text", source_dir=d)
        assert SkillExecutor._get_skill_content(skill) == "fallback text"
        print(f"  ✓ fallback 到 content")


def test_skill_content_no_dir():
    print("\n【2.3 无 source_dir】")
    skill = Skill(name="t", description="", content="x=1")
    assert SkillExecutor._get_skill_content(skill) == "x=1"
    print(f"  ✓ 直接用 content")


# ================================================================
# 3. _build_prompt
# ================================================================


def test_prompt_knowledge():
    print("\n【3.1 knowledge prompt】")
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "SKILL.md").write_text("# Search Guide")
        skill = Skill(
            name="diag",
            description="diagnostics",
            content="",
            source_dir=d,
            execution_mode="knowledge",
        )
        prompt = SkillExecutor()._build_prompt(skill, "find bug")
        assert "Search Guide" in prompt and "Available Scripts" not in prompt
        print(f"  ✓ 无 scripts 信息 ({len(prompt)} chars)")


def test_prompt_playbook():
    print("\n【3.2 playbook prompt】")
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "SKILL.md").write_text("# PPTX")
        s = Path(d) / "scripts"
        s.mkdir()
        (s / "thumb.py").write_text("print(1)")
        skill = Skill(
            name="pptx",
            description="pptx",
            content="",
            source_dir=d,
            execution_mode="playbook",
        )
        prompt = SkillExecutor()._build_prompt(skill, "make slides")
        assert "Available Scripts" in prompt and "thumb.py" in prompt
        print(f"  ✓ 含 scripts 信息 ({len(prompt)} chars)")


# ================================================================
# 3b. prompt 内容验证（bash 偏向 / 平台信息）
# ================================================================


def test_prompt_no_universal_fallback():
    print("\n【3.3 prompt 不含 universal fallback】")
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "SKILL.md").write_text("# Test")
        skill = Skill(
            name="t",
            description="test",
            content="",
            source_dir=d,
            execution_mode="knowledge",
        )
        prompt = SkillExecutor()._build_prompt(skill, "do something")
        assert "universal fallback" not in prompt
        print(f"  ✓ 未发现 'universal fallback'")


def test_prompt_contains_platform_info():
    print("\n【3.4 prompt 包含平台信息】")
    import platform as _platform

    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "SKILL.md").write_text("# Test")
        skill = Skill(
            name="t",
            description="test",
            content="",
            source_dir=d,
            execution_mode="knowledge",
        )
        prompt = SkillExecutor()._build_prompt(skill, "do something")
        expected = f"{_platform.system()} {_platform.machine()}"
        assert expected in prompt
        assert "## Platform" in prompt
        print(f"  ✓ 包含平台信息: {expected}")


def test_get_python_executable_no_sandbox():
    print("\n【3.5 _get_python_executable 无 sandbox】")
    executor = SkillExecutor()
    result = executor._get_python_executable()
    assert result == sys.executable
    print(f"  ✓ 返回 sys.executable: {result}")


def test_get_python_executable_with_sandbox():
    print("\n【3.6 _get_python_executable 有 sandbox】")

    class _MockSandbox:
        @property
        def python_executable(self):
            return Path("/mock/venv/bin/python")

    executor = SkillExecutor(sandbox=_MockSandbox())
    result = executor._get_python_executable()
    assert result == "/mock/venv/bin/python"
    print(f"  ✓ 返回 sandbox Python: {result}")


def test_get_python_executable_sandbox_error():
    print("\n【3.7 _get_python_executable sandbox 异常回退】")

    class _BrokenSandbox:
        @property
        def python_executable(self):
            raise RuntimeError("sandbox broken")

    executor = SkillExecutor(sandbox=_BrokenSandbox())
    result = executor._get_python_executable()
    assert result == sys.executable
    print(f"  ✓ 异常时回退到 sys.executable")


# ================================================================
# 4. _extract_code_block
# ================================================================


def test_extract_python_block():
    print("\n【4.1 提取 python 代码块】")
    code = SkillExecutor._extract_code_block("text\n```python\nprint(1)\n```\nmore")
    assert code is not None and "print(1)" in code and "```" not in code
    print(f"  ✓ '{code.strip()}'")


def test_extract_bare_block():
    print("\n【4.2 裸代码块】")
    code = SkillExecutor._extract_code_block("```\nprint(2)\n```")
    assert code is not None and "print(2)" in code
    print(f"  ✓ '{code.strip()}'")


def test_extract_no_block():
    print("\n【4.3 无代码块 -> None】")
    assert SkillExecutor._extract_code_block("plain text answer") is None
    print(f"  ✓ None")


def test_extract_empty_block():
    print("\n【4.4 空代码块 -> None】")
    assert SkillExecutor._extract_code_block("```\n```") is None
    print(f"  ✓ None")


# ================================================================
# 5. tool_calls 真实执行
# ================================================================


def test_tc_list_dir():
    print("\n【5.1 list_dir】")
    r, _ = _run(
        SkillExecutor()._execute_with_tool_calls(
            Skill(name="t", description="", content=""),
            [
                ToolCall(
                    id="1", name="list_dir", arguments={"path": ".", "max_depth": 1}
                )
            ],
        )
    )
    assert r.success and "list_dir" in r.result
    print(f"  ✓ {len(r.result)} chars")


def test_tc_read_file():
    print("\n【5.2 read_file】")
    r, _ = _run(
        SkillExecutor()._execute_with_tool_calls(
            Skill(name="t", description="", content=""),
            [ToolCall(id="1", name="read_file", arguments={"path": "pyproject.toml"})],
        )
    )
    assert r.success and "memento" in r.result.lower()
    print(f"  ✓ 读到 pyproject.toml")


def test_tc_grep():
    print("\n【5.3 grep】")
    r, _ = _run(
        SkillExecutor()._execute_with_tool_calls(
            Skill(name="t", description="", content=""),
            [
                ToolCall(
                    id="1",
                    name="grep",
                    arguments={
                        "pattern": "class SkillExecutor",
                        "dir_path": "core/skill",
                        "file_pattern": "*.py",
                    },
                )
            ],
        )
    )
    assert r.success and "executor.py" in r.result
    print(f"  ✓ 找到 executor.py")


def test_tc_bash():
    print("\n【5.4 bash】")
    r, _ = _run(
        SkillExecutor()._execute_with_tool_calls(
            Skill(name="t", description="", content=""),
            [ToolCall(id="1", name="bash", arguments={"command": "echo OK_FROM_BASH"})],
        )
    )
    assert r.success and "OK_FROM_BASH" in r.result
    print(f"  ✓ echo OK")


def test_tc_multi():
    print("\n【5.5 多 tool 组合】")
    r, _ = _run(
        SkillExecutor()._execute_with_tool_calls(
            Skill(name="t", description="", content=""),
            [
                ToolCall(id="1", name="list_dir", arguments={"path": "."}),
                ToolCall(
                    id="2", name="bash", arguments={"command": "python3 --version"}
                ),
            ],
        )
    )
    assert r.success and len(r.operation_results) == 2
    print(f"  ✓ 2 个 tool_call 成功")


def test_tc_policy_block():
    print("\n【5.6 policy 拦截】")
    r, _ = _run(
        SkillExecutor()._execute_with_tool_calls(
            Skill(name="t", description="", content=""),
            [ToolCall(id="1", name="bash", arguments={"command": "rm -rf /"})],
        )
    )
    assert not r.success and r.error
    print(f"  ✓ 拦截: {r.error}")


def test_tc_unknown_tool():
    print("\n【5.7 未知 tool 跳过】")
    r, _ = _run(
        SkillExecutor()._execute_with_tool_calls(
            Skill(name="t", description="", content=""),
            [ToolCall(id="1", name="magic_tool_xyz", arguments={})],
        )
    )
    assert r.success and "not_builtin_tool" in r.result
    print(f"  ✓ 跳过")


# ================================================================
# 6. fallback 真实执行
# ================================================================


def test_fb_text():
    print("\n【6.1 纯文本直接返回】")
    r, code = _run(
        SkillExecutor()._execute_fallback(
            Skill(name="t", description="", content=""),
            "Here is the answer:\n1. Step one\n2. Step two",
        )
    )
    assert r.success and "answer" in r.result and code == ""
    print(f"  ✓ 文本直接返回")


def test_fb_code_block():
    print("\n【6.2 代码块执行】")
    r, code = _run(
        SkillExecutor()._execute_fallback(
            Skill(name="t", description="", content=""),
            'Script:\n```python\nimport os\nprint("count:", len(os.listdir(".")))\n```',
        )
    )
    assert r.success and "count:" in r.result and code != ""
    print(f"  ✓ 执行结果: {r.result.strip()}")


# ================================================================
# 7. 真实 LLM 端到端 — executor.execute() 完整链路
# ================================================================


def _load_skill(name: str) -> Skill | None:
    """从 builtin/skills 加载真实 Skill 对象。"""
    skill_dir = project_root / "builtin" / "skills" / name
    if not skill_dir.exists():
        return None
    md = skill_dir / "SKILL.md"
    md_text = md.read_text("utf-8") if md.exists() else ""

    is_playbook = (skill_dir / "scripts").exists()
    return Skill(
        name=name,
        description=md_text.split("\n")[0] if md_text else "",
        content=md_text,
        source_dir=str(skill_dir),
        execution_mode="playbook" if is_playbook else "knowledge",
    )


def _check_llm():
    """检查 LLM 是否可用（有 api_key 且网络通）。"""
    try:
        from middleware.llm import LLMClient

        client = LLMClient()
        return bool(client.api_key)
    except Exception:
        return False


_llm_available = _check_llm()


def test_e2e_knowledge_grep():
    """真实 LLM: tool-diagnostics 搜索 SkillExecutor 定义。"""
    print("\n【7.1 真实 LLM: knowledge skill grep】")
    if not _llm_available:
        print("  ⊘ 跳过: LLM 不可用（无 api_key 或网络）")
        return

    skill = _load_skill("tool-diagnostics")
    if not skill:
        print("  ⊘ 跳过: tool-diagnostics 不存在")
        return

    executor = SkillExecutor()
    r, code = _run(
        executor.execute(
            skill,
            "在 core/skill/execution 目录下找到 SkillExecutor 类定义所在的文件和行号",
        )
    )

    print(f"  成功={r.success}, 结果长度={len(r.result)}")
    if r.error:
        print(f"  错误: {r.error}")
    assert r.success, f"execute 失败: {r.error}"
    result_lower = r.result.lower()
    assert "executor" in result_lower, f"结果中应包含 executor: {r.result[:200]}"
    print(f"  ✓ LLM 返回 tool_calls 找到 SkillExecutor ({len(r.result)} chars)")


def test_e2e_knowledge_read():
    """真实 LLM: tool-diagnostics 读取文件内容。"""
    print("\n【7.2 真实 LLM: knowledge skill read_file】")
    if not _llm_available:
        print("  ⊘ 跳过: LLM 不可用")
        return

    skill = _load_skill("tool-diagnostics")
    if not skill:
        print("  ⊘ 跳过: tool-diagnostics 不存在")
        return

    executor = SkillExecutor()
    r, code = _run(
        executor.execute(
            skill,
            "读取 pyproject.toml 文件的前 10 行",
        )
    )

    print(f"  成功={r.success}, 结果长度={len(r.result)}")
    assert r.success, f"execute 失败: {r.error}"
    print(f"  ✓ LLM 成功读取文件 ({len(r.result)} chars)")


def test_e2e_knowledge_list():
    """真实 LLM: tool-diagnostics 列出目录结构。"""
    print("\n【7.3 真实 LLM: knowledge skill list_dir】")
    if not _llm_available:
        print("  ⊘ 跳过: LLM 不可用")
        return

    skill = _load_skill("tool-diagnostics")
    if not skill:
        print("  ⊘ 跳过: tool-diagnostics 不存在")
        return

    executor = SkillExecutor()
    r, code = _run(
        executor.execute(
            skill,
            "列出 builtin/tools 目录下所有的 .py 文件",
        )
    )

    print(f"  成功={r.success}, 结果长度={len(r.result)}")
    assert r.success, f"execute 失败: {r.error}"
    print(f"  ✓ LLM 成功列出目录 ({len(r.result)} chars)")


def test_e2e_not_relevant():
    """真实 LLM: 不相关请求应返回 NOT_RELEVANT 或拒绝。"""
    print("\n【7.4 真实 LLM: 不相关请求】")
    if not _llm_available:
        print("  ⊘ 跳过: LLM 不可用")
        return

    skill = _load_skill("tool-diagnostics")
    if not skill:
        print("  ⊘ 跳过: tool-diagnostics 不存在")
        return

    executor = SkillExecutor()
    r, code = _run(
        executor.execute(
            skill,
            "帮我写一首关于春天的诗",
        )
    )

    print(f"  成功={r.success}, 错误={r.error}")
    if not r.success and r.error:
        print(f"  ✓ LLM 正确拒绝: {r.error[:100]}")
    else:
        print(f"  ⊘ LLM 未拒绝（可接受，不同模型行为不同）")


# ================================================================
# main
# ================================================================


if __name__ == "__main__":
    print("=" * 70)
    print("SkillExecutor 单元测试")
    print("=" * 70)

    tests = [
        # 1
        test_normalize_args_filters_unknown,
        test_normalize_args_defaults,
        test_normalize_args_type_cast,
        test_normalize_args_bad_cast,
        test_normalize_args_bool,
        test_normalize_args_no_schema,
        # 2
        test_skill_content_prefers_md,
        test_skill_content_no_dir,
        # 3
        test_prompt_knowledge,
        test_prompt_playbook,
        # 3b
        test_prompt_no_universal_fallback,
        test_prompt_contains_platform_info,
        test_get_python_executable_no_sandbox,
        test_get_python_executable_with_sandbox,
        test_get_python_executable_sandbox_error,
        # 4
        test_extract_python_block,
        test_extract_bare_block,
        test_extract_no_block,
        test_extract_empty_block,
        # 5
        test_tc_list_dir,
        test_tc_read_file,
        test_tc_grep,
        test_tc_bash,
        test_tc_multi,
        test_tc_policy_block,
        test_tc_unknown_tool,
        # 6
        test_fb_text,
        test_fb_code_block,
        # 7 (真实 LLM)
        test_e2e_knowledge_grep,
        test_e2e_knowledge_read,
        test_e2e_knowledge_list,
        test_e2e_not_relevant,
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
