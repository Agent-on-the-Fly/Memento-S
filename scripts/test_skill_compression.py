# SPDX-License-Identifier: Apache-2.0
"""压缩机制集成测试（使用新架构 compaction.py）。

运行方式:
  python -m pytest scripts/test_skill_compression.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from core.skill.execution.agent import SkillAgent
from core.skill.execution.compaction import (
    TokenBudgetPolicy,
    SummarizerRegistry,
    get_default_registry,
    make_budget_policy,
    BashSummarizer,
    PythonReplSummarizer,
    FileCreateSummarizer,
    ReadFileSummarizer,
)
from core.skill.execution.state import ReActState, ContextCompactor
from shared.schema import SkillConfig
from middleware.llm import LLMClient


# ─── 测试 TokenBudgetPolicy ─────────────────────────────────────────────────

class TestTokenBudgetPolicy:
    def test_warn_threshold_is_80_percent(self):
        policy = TokenBudgetPolicy(budget=100_000)
        assert policy.warn_threshold == 80_000
        assert policy.urgent_threshold == 90_000
        assert policy.compact_threshold == 75_000

    def test_make_budget_policy_from_llm_params(self):
        policy = make_budget_policy(context_window=128000, max_output_tokens=4096)
        assert policy.budget == 123_904  # 128000 - 4096
        assert policy.warn_threshold == int(policy.budget * 0.80)

    def test_make_budget_policy_fallback(self):
        policy = make_budget_policy(context_window=0, max_output_tokens=0)
        assert policy.budget == 8192  # 兜底最小值

    def test_custom_ratios(self):
        policy = TokenBudgetPolicy(budget=100_000, warn_ratio=0.70, urgent_ratio=0.85)
        assert policy.warn_threshold == 70_000
        assert policy.urgent_threshold == 85_000

    def test_microcompact_keep_recent(self):
        policy = TokenBudgetPolicy(budget=100_000, microcompact_keep_recent=5)
        assert policy.microcompact_keep_recent == 5

    def test_truncate_keep_recent(self):
        policy = TokenBudgetPolicy(budget=100_000, truncate_keep_recent=6)
        assert policy.truncate_keep_recent == 6


# ─── 测试 SummarizerRegistry ───────────────────────────────────────────────

class TestSummarizerRegistry:
    def test_default_builtins(self):
        reg = get_default_registry()
        assert reg.summarize("bash", "x" * 500) != "x" * 500
        assert reg.summarize("python_repl", "x" * 500) != "x" * 500
        assert reg.summarize("file_create", "x" * 500) != "x" * 500

    def test_register_new_tool(self):
        class CustomSummarizer:
            def summarize(self, tool_name, content):
                return f"[custom] {len(content)} chars"

        reg = SummarizerRegistry()
        reg.register("custom_tool", CustomSummarizer())
        assert reg.summarize("custom_tool", "hello world") == "[custom] 11 chars"

    def test_alias(self):
        reg = get_default_registry()
        reg.register_alias("uv-pip-install", "bash")
        result = reg.summarize("uv-pip-install", "Installing numpy... exit: 0")
        assert "exit" in result

    def test_unknown_tool_uses_fallback(self):
        reg = get_default_registry()
        result = reg.summarize("unknown_tool", "x" * 500)
        assert len(result) <= 200


# ─── 测试工具专用摘要器 ───────────────────────────────���───────────────────

class TestToolSummarizers:
    def test_bash_extracts_exit_code(self):
        content = (
            "Installing numpy...\n"
            "Collecting numpy\n"
            "Downloading numpy-1.26.0.whl (15MB)\n"
            "Successfully installed numpy-1.26.0\n"
            "exit: 0"
        )
        result = BashSummarizer().summarize("bash", content)
        assert "exit" in result
        assert len(result) < 200

    def test_bash_extracts_file_paths(self):
        content = (
            "Processing /Users/manson/ai/memento/opc_memento_s/data.csv\n"
            "Output saved to /Users/manson/ai/memento/opc_memento_s/result.csv\n"
            "Done."
        )
        result = BashSummarizer().summarize("bash", content)
        assert "result.csv" in result

    def test_python_repl_keeps_last_line(self):
        content = "Step 1: loading data\nStep 2: processing\nResult: 42.0\n"
        result = PythonReplSummarizer().summarize("python_repl", content)
        assert "42.0" in result

    def test_file_create_preserves_path(self):
        content = (
            "File created successfully.\n"
            "Full path: /Users/manson/ai/memento/opc_memento_s/outputs/report.xlsx\n"
            "Size: 2.3 MB\n"
            + "x" * 500
        )
        result = FileCreateSummarizer().summarize("file_create", content)
        assert "report.xlsx" in result
        assert len(result) < 300

    def test_read_file_truncates_long_content(self):
        content = "x" * 500
        result = ReadFileSummarizer().summarize("read_file", content)
        assert len(result) <= 253  # 250 + "..."


# ─── 测试 microcompact（保留在 state.py 中）与新架构的集成 ──────────────────────

class TestMicrocompact:
    def test_microcompact_preserves_recent_3(self):
        compactor = ContextCompactor(threshold=50000, llm=None)

        # 只有 3 条 tool results → KEEP_RECENT=3 → 不压缩
        messages = [
            {"role": "tool", "tool_call_id": "tc_0", "name": "bash", "content": "x" * 200},
            {"role": "tool", "tool_call_id": "tc_1", "name": "bash", "content": "y" * 200},
            {"role": "tool", "tool_call_id": "tc_2", "name": "bash", "content": "z" * 200},
        ]
        compactor._bind_tool_name_map(messages)
        modified = compactor.microcompact(messages)
        assert not modified

    def test_microcompact_compresses_old_inference_logs(self):
        """python_repl（INFERENCE_TOOLS）→ microcompact 压缩为摘要"""
        compactor = ContextCompactor(threshold=50000, llm=None)

        messages = []
        for i in range(6):
            messages.append({"role": "user", "content": "Do it"})
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": f"tc_{i}", "function": {"name": "python_repl"}}],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": f"tc_{i}",
                "name": "python_repl",
                "content": f"[50%] Processing batch {i}/100\n" * 15,
            })

        compactor._bind_tool_name_map(messages)
        modified = compactor.microcompact(messages)
        assert modified

        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        for tm in tool_msgs[:3]:
            assert tm["content"].startswith("[python_repl]")
        for tm in tool_msgs[3:]:
            assert tm["content"].startswith("[50%]")


# ─── 测试 _truncate_old_tool_results 使用 SummarizerRegistry ────────────────

def _make_agent_and_state():
    config = MagicMock(spec=SkillConfig)
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.context_window = 100_000
    mock_llm.max_tokens = 4096
    agent = SkillAgent(config=config, llm=mock_llm)

    state = ReActState(query="test", params=None)
    state._compactor = ContextCompactor(threshold=50000, llm=None)
    state._summarizer_registry = get_default_registry()
    state._budget_policy = TokenBudgetPolicy(budget=100_000, truncate_keep_recent=4)
    return agent, state


class TestTruncateOldToolResults:
    def test_preserves_recent_4(self):
        agent, state = _make_agent_and_state()

        for i in range(8):
            state.context._raw_messages.append({
                "role": "tool",
                "tool_call_id": f"tc_{i}",
                "name": "bash",
                "content": f"output from command {i}: " + "x" * 300,
            })

        state._compactor._bind_tool_name_map(state.context._raw_messages)
        agent._truncate_old_tool_results(state, state._budget_policy)

        for i in range(4):
            assert len(state.context._raw_messages[i]["content"]) <= 200
        for i in range(4, 8):
            assert "x" * 300 in state.context._raw_messages[i]["content"]

    def test_skips_already_compressed(self):
        agent, state = _make_agent_and_state()

        for i in range(4):
            state.context._raw_messages.append({
                "role": "tool",
                "tool_call_id": f"tc_{i}",
                "name": "bash",
                "content": f"[bash] compressed {i}",
            })
        for i in range(4, 8):
            state.context._raw_messages.append({
                "role": "tool",
                "tool_call_id": f"tc_{i}",
                "name": "bash",
                "content": "x" * 300,
            })

        state._compactor._bind_tool_name_map(state.context._raw_messages)
        agent._truncate_old_tool_results(state, state._budget_policy)

        for i in range(4):
            assert state.context._raw_messages[i]["content"] == f"[bash] compressed {i}"
        for i in range(4, 8):
            assert len(state.context._raw_messages[i]["content"]) < 400


# ─── 测试 _budget_aware_compact 使用新架构 ─────────────────────────────────

class TestBudgetAwareCompact:
    def test_no_compact_under_80_percent(self):
        agent, state = _make_agent_and_state()

        state.context._raw_messages.append({
            "role": "tool",
            "tool_call_id": "tc_0",
            "name": "bash",
            "content": "hello",
        })

        original_content = state.context._raw_messages[0]["content"]
        agent._budget_aware_compact(state, system_prompt="x" * 100, budget=100_000)

        assert state.context._raw_messages[0]["content"] == original_content

    def test_triggers_stage2_over_90_percent(self):
        agent, state = _make_agent_and_state()

        # 注入一个 budget=1000 的策略，使 Stage 2 必然触发
        state._budget_policy = TokenBudgetPolicy(
            budget=1000,
            warn_ratio=0.50,
            urgent_ratio=0.80,
            truncate_keep_recent=4,
        )

        for i in range(20):
            state.context._raw_messages.append({
                "role": "tool",
                "tool_call_id": f"tc_{i}",
                "name": "bash",
                "content": f"command output {i}: " + "x" * 500,
            })
        state._compactor._bind_tool_name_map(state.context._raw_messages)

        agent._budget_aware_compact(state, system_prompt="x" * 100, budget=1000)

        old_msg = state.context._raw_messages[0]
        assert len(old_msg["content"]) < 250, (
            f"Stage 2 should truncate, got {len(old_msg['content'])} chars"
        )


# ─── 测试新架构的扩展性 ───────────────────────────────────────────────────

class TestExtensibility:
    def test_custom_policy_override_defaults(self):
        """验证可通过 TokenBudgetPolicy 覆盖所有硬code 阈值"""
        policy = TokenBudgetPolicy(
            budget=100_000,
            warn_ratio=0.60,   # 更早触发
            urgent_ratio=0.80,  # 更早触发
            truncate_keep_recent=6,
            truncate_max_chars=300,
        )
        assert policy.warn_threshold == 60_000
        assert policy.urgent_threshold == 80_000
        assert policy.truncate_keep_recent == 6
        assert policy.truncate_max_chars == 300

    def test_custom_summarizer_for_new_tool(self):
        """验证新增工具只需注册，无需改动业务代码"""
        reg = get_default_registry()

        class MyToolSummarizer:
            def summarize(self, tool_name, content):
                return f"[{tool_name}] processed {len(content)} bytes"

        reg.register("my_new_tool", MyToolSummarizer())
        result = reg.summarize("my_new_tool", "hello world")
        assert result == "[my_new_tool] processed 11 bytes"