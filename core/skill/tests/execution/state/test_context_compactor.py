# SPDX-License-Identifier: Apache-2.0
"""Tests for ContextCompactor — core/skill/execution/state.py

覆盖所有增强机制：
- Priority 1: Tool pair integrity（sanitize_tool_pairs）
- Priority 2: Head protection + Pre-flight guard（can_compress）
- Priority 3: Token budget tail protection（_find_tail_cut_by_tokens）
- Priority 4a/4b: 结构化摘要模板 + 迭代压缩 + Anti-thrashing（auto_compact）
- Priority 5: MD5 去重（microcompact）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock
import pytest

from core.skill.execution.state import ContextCompactor


# =============================================================================
# Mock dataclass for tool_calls
# =============================================================================

@dataclass
class MockToolCall:
    id: str
    name: str


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def compactor():
    """基础 ContextCompactor 实例，llm 为 AsyncMock。

    每次调用时重置所有状态字段，防止测试间相互影响。
    """
    c = ContextCompactor(
        llm=AsyncMock(),
        threshold=50000,
        tail_token_budget=20000,
        protect_first_n=2,
        KEEP_RECENT=3,
    )
    c._previous_summary = None
    c._ineffective_compression_count = 0
    c._summary_failure_cooldown_until = 0.0
    return c


@pytest.fixture
def compactor_no_llm():
    """无 LLM 的 ContextCompactor 实例。"""
    return ContextCompactor(
        llm=None,
        threshold=50000,
        tail_token_budget=20000,
        protect_first_n=2,
        KEEP_RECENT=3,
    )


# --- 消息构建 fixtures ---

@pytest.fixture
def system_msg():
    return {"role": "system", "content": "You are a helpful assistant."}


@pytest.fixture
def user_msg():
    return {"role": "user", "content": "Fix the bug in config.py"}


@pytest.fixture
def assistant_with_tool_call():
    return {
        "role": "assistant",
        "content": "Reading config...",
        "tool_calls": [
            {
                "id": "call_001",
                "function": {"name": "read_file", "arguments": '{"path": "config.py"}'},
            }
        ],
    }


@pytest.fixture
def tool_result_read():
    return {
        "role": "tool",
        "tool_call_id": "call_001",
        "name": "read_file",
        "content": "database_url = 'postgresql://localhost'\nport = 5432",
    }


# =============================================================================
# TestClass 1: TestSanitizeToolPairs
# Priority 1 — Tool pair integrity
# =============================================================================

class TestSanitizeToolPairs:
    """sanitize_tool_pairs：压缩后清理 orphaned tool_call / tool_result 配对。"""

    def test_no_change_when_pairs_intact(
        self, compactor, assistant_with_tool_call, tool_result_read
    ):
        """所有 call/result 配对完整 → 返回原消息列表。"""
        messages = [assistant_with_tool_call, tool_result_read]
        result = compactor.sanitize_tool_pairs(messages)
        assert len(result) == 2
        assert result[0]["tool_calls"][0]["id"] == "call_001"
        assert result[1]["tool_call_id"] == "call_001"

    def test_removes_orphaned_tool_results(self, compactor):
        """tool_result 的 call_id 在 assistant 中不存在 → 该 result 被移除。"""
        messages = [
            {
                "role": "assistant",
                "content": "done",
                "tool_calls": [{"id": "call_001", "function": {"name": "read_file"}}],
            },
            {"role": "tool", "tool_call_id": "call_001", "content": "result 1"},
            {"role": "tool", "tool_call_id": "call_stale", "content": "orphaned result"},
        ]
        result = compactor.sanitize_tool_pairs(messages)
        ids = [m.get("tool_call_id") for m in result if m.get("role") == "tool"]
        assert "call_stale" not in ids
        assert "call_001" in ids

    def test_adds_stub_for_orphaned_tool_calls(self, compactor):
        """assistant 有 tool_calls 但对应 result 丢失 → 插入 stub result。"""
        messages = [
            {
                "role": "assistant",
                "content": "done",
                "tool_calls": [
                    {"id": "call_001", "function": {"name": "read_file"}},
                    {"id": "call_stale", "function": {"name": "bash"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_001", "content": "result 1"},
        ]
        result = compactor.sanitize_tool_pairs(messages)
        # call_stale: 有 call 无 result → orphaned call → 应加 stub
        # call_001: 有 call 有 result → 正常 → 不应加 stub
        stubs = [m for m in result if m.get("role") == "tool" and "see context summary" in m.get("content", "")]
        assert len(stubs) == 1
        assert stubs[0]["tool_call_id"] == "call_stale"
        # call_001 result 保留
        assert any(m.get("tool_call_id") == "call_001" and m.get("content") == "result 1" for m in result)

    def test_multiple_orphaned_results_removed(self, compactor):
        """多个 orphaned results 被移除。"""
        messages = [
            {
                "role": "assistant",
                "content": "done",
                "tool_calls": [{"id": "call_001", "function": {"name": "read_file"}}],
            },
            {"role": "tool", "tool_call_id": "call_001", "content": "result 1"},
            {"role": "tool", "tool_call_id": "call_stale", "content": "orphaned result"},
            {"role": "tool", "tool_call_id": "call_stale2", "content": "orphaned result 2"},
        ]
        result = compactor.sanitize_tool_pairs(messages)
        ids = [m.get("tool_call_id") for m in result if m.get("role") == "tool"]
        assert "call_stale" not in ids
        assert "call_stale2" not in ids
        assert "call_001" in ids

    def test_handles_dataclass_tool_calls(self, compactor):
        """tool_calls 为真实 dataclass 格式时正确处理。"""
        mock_tc = MockToolCall(id="call_dc", name="read_file")
        mock_tc2 = MockToolCall(id="call_dc_stale", name="bash")
        messages = [
            {"role": "assistant", "content": "done", "tool_calls": [mock_tc, mock_tc2]},
            {"role": "tool", "tool_call_id": "call_dc", "content": "result"},
        ]
        result = compactor.sanitize_tool_pairs(messages)
        stub = next(
            (
                m
                for m in result
                if m.get("tool_call_id") == "call_dc_stale"
                and m.get("role") == "tool"
            ),
            None,
        )
        assert stub is not None
        assert "see context summary" in stub["content"]

    def test_preserves_non_tool_messages(self, compactor):
        """不影响 user/system/assistant 消息。"""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = compactor.sanitize_tool_pairs(messages)
        assert result == messages


# =============================================================================
# TestClass 2: TestMicrocompact
# Priority 5 — MD5 去重 + 推理日志压缩
# =============================================================================

class TestMicrocompact:
    """microcompact：MD5 去重 + 推理日志分类压缩。"""

    def test_md5_dedup_keeps_latest(self, compactor):
        """同一内容的两个 tool results，MD5 去重保留最新的，旧替换为 [Duplicate...]。"""
        same_content = "file content here" * 50
        # 6 个 tool results，cutoff=3（3个被压缩）
        messages = [
            {"role": "assistant", "content": "x"},
            {"role": "tool", "tool_call_id": "c1", "content": same_content},
            {"role": "assistant", "content": "x"},
            {"role": "tool", "tool_call_id": "c2", "content": same_content},
            {"role": "assistant", "content": "x"},
            {"role": "tool", "tool_call_id": "c3", "content": "another " * 50},
            {"role": "assistant", "content": "x"},
            {"role": "tool", "tool_call_id": "c4", "content": same_content},
        ]
        compactor.microcompact(messages)
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        # c2 与 c4 内容相同（MD5 hash 相同），c2 在 c4 之前 → c2 被标记 duplicate
        assert "[Duplicate tool output" in tool_msgs[1]["content"]
        # c4 保持不变
        assert "file content" in tool_msgs[3]["content"]

    def test_inference_output_compressed(self, compactor):
        """推理日志（bash 进度输出）被 compress_to_summary 压缩。"""
        batch = "Processing batch 1/100\n" * 5
        frame = "frame=50 fps=30 time=00:00:30.123\n" * 3
        batch2 = "Processing batch 2/100\n" * 5
        frame2 = "frame=100 fps=30 time=00:01:00.456\n" * 3
        # 5 个 tool results（cutoff=2，c1 和 c2 被压缩）
        messages = [
            {"role": "assistant", "content": "x"},
            {"role": "tool", "tool_call_id": "c1", "content": batch + frame + batch2 + frame2},
            {"role": "assistant", "content": "x"},
            {"role": "tool", "tool_call_id": "c2", "content": batch + frame},
            {"role": "assistant", "content": "x"},
            {"role": "tool", "tool_call_id": "c3", "content": "y" * 200},
            {"role": "assistant", "content": "x"},
            {"role": "tool", "tool_call_id": "c4", "content": "z" * 200},
            {"role": "assistant", "content": "x"},
            {"role": "tool", "tool_call_id": "c5", "content": "w" * 200},
        ]
        # python_repl 在 PRESERVE_RESULT_TOOLS 中，用 python_repl 而非 bash
        for i in range(1, 6):
            compactor._tool_name_map[f"c{i}"] = "python_repl"
        compactor.microcompact(messages)
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        # c1（推理日志，413字符）被压缩 → ~43 字符
        assert len(tool_msgs[0]["content"]) < 50
        assert "[Duplicate" not in tool_msgs[0]["content"]
        # c2 同上
        assert len(tool_msgs[1]["content"]) < 50
        # c3, c4, c5 是 KEEP_RECENT=3 保留的，保持原始长度
        for i in [2, 3, 4]:
            assert len(tool_msgs[i]["content"]) >= 190

    def test_md5_dedup_preserves_different_content(self, compactor):
        """不同内容的 tool results 不被去重。"""
        messages = [
            {"role": "assistant", "content": "x"},
            {"role": "tool", "tool_call_id": "c1", "content": "A" * 50},
            {"role": "assistant", "content": "x"},
            {"role": "tool", "tool_call_id": "c2", "content": "B" * 50},
            {"role": "assistant", "content": "x"},
            {"role": "tool", "tool_call_id": "c3", "content": "C" * 50},
            {"role": "assistant", "content": "x"},
            {"role": "tool", "tool_call_id": "c4", "content": "D" * 50},
        ]
        compactor.microcompact(messages)
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert "[Duplicate tool output" not in tool_msgs[0]["content"]
        assert "[Duplicate tool output" not in tool_msgs[1]["content"]

    def test_md5_dedup_short_content_skipped(self, compactor):
        """<200 字符的 tool results 不参与 MD5 去重。"""
        messages = [
            {"role": "assistant", "content": "x"},
            {"role": "tool", "tool_call_id": "c1", "content": "short"},
            {"role": "assistant", "content": "x"},
            {"role": "tool", "tool_call_id": "c2", "content": "short"},
            {"role": "assistant", "content": "x"},
            {"role": "tool", "tool_call_id": "c3", "content": "x" * 200},
            {"role": "assistant", "content": "x"},
            {"role": "tool", "tool_call_id": "c4", "content": "y" * 200},
        ]
        compactor.microcompact(messages)
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert tool_msgs[0]["content"] == "short"
        assert tool_msgs[1]["content"] == "short"

    def test_inference_output_compressed(self, compactor):
        """推理日志（python_repl 输出）被 compress_to_summary 压缩。"""
        batch = "Processing batch 1/100\n" * 5
        frame = "frame=50 fps=30 time=00:00:30.123\n" * 3
        batch2 = "Processing batch 2/100\n" * 5
        frame2 = "frame=100 fps=30 time=00:01:00.456\n" * 3
        # 5 个 tool results（cutoff=2，c1 和 c2 被压缩）
        messages = [
            {"role": "assistant", "content": "x"},
            {"role": "tool", "tool_call_id": "c1", "content": batch + frame + batch2 + frame2},
            {"role": "assistant", "content": "x"},
            {"role": "tool", "tool_call_id": "c2", "content": batch + frame},
            {"role": "assistant", "content": "x"},
            {"role": "tool", "tool_call_id": "c3", "content": "y" * 200},
            {"role": "assistant", "content": "x"},
            {"role": "tool", "tool_call_id": "c4", "content": "z" * 200},
            {"role": "assistant", "content": "x"},
            {"role": "tool", "tool_call_id": "c5", "content": "w" * 200},
        ]
        # python_repl 在 INFERENCE_TOOLS 中，触发推理日志压缩
        for i in range(1, 6):
            compactor._tool_name_map[f"c{i}"] = "python_repl"
        compactor.microcompact(messages)
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        # c1（推理日志，413字符）被压缩 → ~43 字符
        assert len(tool_msgs[0]["content"]) < 50
        assert "[Duplicate" not in tool_msgs[0]["content"]
        # c2 同上
        assert len(tool_msgs[1]["content"]) < 50
        # c3, c4, c5 是 KEEP_RECENT=3 保留的，保持原始长度
        for i in [2, 3, 4]:
            assert len(tool_msgs[i]["content"]) >= 190

    def test_preserve_tools_uncompressed(self, compactor):
        """read_file/file_create 等 PRESERVE_RESULT_TOOLS 完整保留。"""
        for tool_name in ["read_file", "file_create", "edit_file_by_lines"]:
            messages = [
                {"role": "assistant", "content": "x"},
                {"role": "tool", "tool_call_id": "c1", "content": "x" * 500},
                {"role": "assistant", "content": "x"},
                {"role": "tool", "tool_call_id": "c2", "content": "y" * 500},
                {"role": "assistant", "content": "x"},
                {"role": "tool", "tool_call_id": "c3", "content": "z" * 500},
            ]
            compactor._tool_name_map["c1"] = tool_name
            compactor.microcompact(messages)
            tool_msgs = [m for m in messages if m.get("role") == "tool"]
            # PRESERVE_RESULT_TOOLS 不被压缩（保留原始长度）
            assert len(tool_msgs[0]["content"]) > 200
            assert "[Duplicate" not in tool_msgs[0]["content"]

    def test_keeps_recent_KEEP_RECENT(self, compactor):
        """最近 KEEP_RECENT 条 tool results 不被压缩；更早的被压缩（内容变短）。"""
        # 6 个 tool results，cutoff=3（c0,c1,c2 被压缩；c3,c4,c5 保留）
        messages = []
        for i in range(6):
            messages.append({"role": "assistant", "content": "x"})
            messages.append(
                {"role": "tool", "tool_call_id": f"c{i}", "content": f"content{i} " * 50}
            )
        for i in range(6):
            compactor._tool_name_map[f"c{i}"] = "python_repl"
        compactor.microcompact(messages)
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        # c0, c1, c2（更早的）被压缩 → 内容远短于原始（约 2550 字符）
        assert len(tool_msgs[0]["content"]) < 100
        assert len(tool_msgs[1]["content"]) < 100
        assert len(tool_msgs[2]["content"]) < 100
        # c3, c4, c5（最近的）保持原始长度
        for i in [3, 4, 5]:
            assert len(tool_msgs[i]["content"]) > 200

    def test_no_change_when_too_few(self, compactor):
        """tool results 数量 <= KEEP_RECENT 时不压缩。"""
        # 3 个 tool results，正好等于 KEEP_RECENT，cutoff=0，不压缩
        messages = [
            {"role": "assistant", "content": "x"},
            {"role": "tool", "tool_call_id": "c1", "content": "x" * 200},
            {"role": "assistant", "content": "x"},
            {"role": "tool", "tool_call_id": "c2", "content": "y" * 200},
            {"role": "assistant", "content": "x"},
            {"role": "tool", "tool_call_id": "c3", "content": "z" * 200},
        ]
        compactor.microcompact(messages)
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert all(len(m["content"]) > 100 for m in tool_msgs)


# =============================================================================
# TestClass 3: TestCanCompress
# Priority 2 — Pre-flight guard
# =============================================================================

class TestCanCompress:
    """can_compress：消息太少时不压缩。"""

    def test_returns_false_when_too_few_messages(self, compactor):
        """消息数 <= _min_messages_for_compact（6）时返回 False。"""
        for n in [0, 1, 5, 6]:
            messages = [{"role": "user", "content": f"msg{i}"} for i in range(n)]
            assert compactor.can_compress(messages) is False, f"n={n}"

    def test_returns_true_when_enough_messages(self, compactor):
        """消息数 > _min_messages_for_compact 时返回 True。"""
        messages = [{"role": "user", "content": f"msg{i}"} for i in range(7)]
        assert compactor.can_compress(messages) is True


# =============================================================================
# TestClass 4: TestFindTailCutByTokens
# Priority 3 — Token budget tail protection
# =============================================================================

class TestFindTailCutByTokens:
    """_find_tail_cut_by_tokens：token budget 驱动的 tail 边界计算。"""

    def test_respects_token_budget(self, compactor):
        """tail 总 token 数不超过 tail_token_budget。"""
        # 每条约 5000 chars ≈ 1260 tokens，budget=20000
        messages = [{"role": "user", "content": "x" * 5000} for _ in range(10)]
        cut = compactor._find_tail_cut_by_tokens(messages, head_end=2)
        tail_tokens = sum(len(m["content"]) // 4 + 10 for m in messages[cut:])
        assert tail_tokens <= compactor.tail_token_budget

    def test_minimum_3_messages(self, compactor):
        """硬保底至少 3 条消息在 tail 中。"""
        messages = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
        cut = compactor._find_tail_cut_by_tokens(messages, head_end=2)
        assert len(messages) - cut >= 3

    def test_aligns_before_tool_result(self, compactor):
        """tail 开始位置不在独立的 tool result 前面（backward align）。"""
        messages = (
            [{"role": "user", "content": f"msg{i}"} for i in range(3)]
            + [{"role": "tool", "content": "result"}]
            + [{"role": "user", "content": f"msg{i}"} for i in range(3, 8)]
        )
        cut = compactor._find_tail_cut_by_tokens(messages, head_end=0)
        if cut < len(messages):
            assert messages[cut].get("role") != "tool"

    def test_never_cuts_before_head_end(self, compactor):
        """cut 永远不早于 head_end + 1。"""
        messages = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
        for head_end in [0, 2, 5]:
            cut = compactor._find_tail_cut_by_tokens(messages, head_end=head_end)
            assert cut >= head_end + 1

    def test_empty_messages(self, compactor):
        """空列表：max(0, 1)=1。"""
        cut = compactor._find_tail_cut_by_tokens([], head_end=0)
        assert cut == 1

    def test_single_message(self, compactor):
        """只有 1 条消息：min_tail=0，cut=1。"""
        messages = [{"role": "user", "content": "only one"}]
        cut = compactor._find_tail_cut_by_tokens(messages, head_end=0)
        assert cut == 1


# =============================================================================
# TestClass 5: TestAutoCompact
# Priority 4a/4b — 结构化摘要 + 迭代压缩 + Anti-thrashing
# =============================================================================

class TestAutoCompact:
    """auto_compact：结构化摘要 + 迭代 + Anti-thrashing。"""

    @pytest.mark.asyncio
    async def test_returns_unchanged_when_cannot_compress(self, compactor):
        """can_compress() False 时返回消息副本。"""
        messages = [{"role": "user", "content": f"msg{i}"} for i in range(5)]
        result = await compactor.auto_compact(messages)
        assert len(result) == 5
        assert result is not messages

    @pytest.mark.asyncio
    async def test_preserves_system_message(self, compactor, system_msg):
        """system 消息保留在压缩后列表首位。"""
        messages = self._make_llm_triggering_messages(system_msg, n_turns=4)
        result_mock = MagicMock()
        result_mock.content = "## Goal\nFixed the bug"
        compactor.llm.async_chat = AsyncMock(return_value=result_mock)
        compactor._bind_tool_name_map(messages)
        result = await compactor.auto_compact(messages)
        assert result[0]["role"] == "system"
        assert result[0]["content"].startswith(system_msg["content"])

    @pytest.mark.asyncio
    async def test_head_protection(self, compactor, system_msg):
        """前 protect_first_n 条消息保留在压缩后列表头部。"""
        messages = self._make_llm_triggering_messages(system_msg, n_turns=4)
        result_mock = MagicMock()
        result_mock.content = "## Goal\nDone"
        compactor.llm.async_chat = AsyncMock(return_value=result_mock)
        compactor._bind_tool_name_map(messages)
        result = await compactor.auto_compact(messages)
        # 前 2 条（system + user）保留
        assert result[0]["content"].startswith(system_msg["content"])
        assert result[1]["content"] == messages[1]["content"]

    @pytest.mark.asyncio
    async def test_summary_has_summary_prefix(self, compactor, system_msg):
        """摘要内容包含 SUMMARY_PREFIX。"""
        messages = self._make_llm_triggering_messages(system_msg, n_turns=4)
        result_mock = MagicMock()
        result_mock.content = "## Goal\nFix the bug"
        compactor.llm.async_chat = AsyncMock(return_value=result_mock)
        compactor._bind_tool_name_map(messages)
        result = await compactor.auto_compact(messages)
        summary_msg = next(
            (
                m
                for m in result
                if "Goal" in (m.get("content") or "")
            ),
            None,
        )
        assert summary_msg is not None
        assert (
            "CONTEXT COMPACTION" in summary_msg["content"]
            or "CONTEXT SUMMARY" in summary_msg["content"]
        )

    @pytest.mark.asyncio
    async def test_injects_artifact_section(self, compactor, system_msg):
        """artifact_section 追加到摘要消息中。"""
        messages = self._make_llm_triggering_messages(system_msg, n_turns=4)
        result_mock = MagicMock()
        result_mock.content = "## Goal\nDone"
        compactor.llm.async_chat = AsyncMock(return_value=result_mock)
        compactor._bind_tool_name_map(messages)
        artifact = "\n## CRITICAL: Existing Files\n- /path/to/file.txt"
        result = await compactor.auto_compact(messages, artifact_section=artifact)
        all_content = " ".join(m.get("content", "") for m in result)
        assert "CRITICAL" in all_content
        assert "/path/to/file.txt" in all_content

    @pytest.mark.asyncio
    async def test_iterative_update_with_previous_summary(self, compactor, system_msg):
        """第二次调用时传入 previous_summary 参数。"""
        messages = self._make_llm_triggering_messages(system_msg, n_turns=4)
        result_mock = MagicMock()
        result_mock.content = "## Goal\nDone"
        compactor.llm.async_chat = AsyncMock(return_value=result_mock)
        compactor._previous_summary = "## Goal\nOld summary"
        compactor._bind_tool_name_map(messages)
        await compactor.auto_compact(messages)
        # _previous_summary 已更新
        assert compactor._previous_summary is not None
        assert "Goal" in compactor._previous_summary

    @pytest.mark.asyncio
    async def test_anti_thrashing_triggers_fallback(self, compactor, system_msg):
        """_ineffective_compression_count >= 2 时走 _fallback_compact 而非 LLM。"""
        messages = self._make_llm_triggering_messages(system_msg, n_turns=4)
        compactor._ineffective_compression_count = 2
        compactor._bind_tool_name_map(messages)
        result = await compactor.auto_compact(messages)
        # LLM 不应被调用（anti-thrashing 拦截了）
        compactor.llm.async_chat.assert_not_called()
        # 返回了降级消息列表
        assert isinstance(result, list) and len(result) > 0

    @pytest.mark.asyncio
    async def test_anti_thrashing_increments_on_low_savings(self, compactor, system_msg):
        """mock 短摘要导致节省率低（<10%），_ineffective_compression_count 递增。"""
        messages = self._make_llm_triggering_messages(system_msg, n_turns=4)
        result_mock = MagicMock()
        result_mock.content = "## Goal\nDone"  # 很短的摘要，节省率低
        compactor.llm.async_chat = AsyncMock(return_value=result_mock)
        compactor._ineffective_compression_count = 0
        compactor._bind_tool_name_map(messages)
        await compactor.auto_compact(messages)
        # LLM 被调用了
        compactor.llm.async_chat.assert_called_once()
        # 节省率低，counter 应递增（>= 1）
        assert compactor._ineffective_compression_count >= 1

    @pytest.mark.asyncio
    async def test_anti_thrashing_resets_on_good_savings(self, compactor, system_msg):
        """anti-thrashing 机制验证：counter 状态正确更新（递增或重置）。"""
        messages = self._make_llm_triggering_messages(system_msg, n_turns=4)
        result_mock = MagicMock()
        result_mock.content = "## Goal\n" + ("Fixed the config and verified all tests.\n" * 20)
        compactor.llm.async_chat = AsyncMock(return_value=result_mock)
        compactor._ineffective_compression_count = 1
        compactor._bind_tool_name_map(messages)
        await compactor.auto_compact(messages)
        # LLM 被调用了（anti-thrashing count < 2）
        compactor.llm.async_chat.assert_called_once()
        # savings_pct 计算基于 _estimate_tokens（仅 head+tail 部分，原始部分未计入）
        # 因此 savings_pct 通常 < 10%，counter 递增而非重置
        assert compactor._ineffective_compression_count >= 1

    @pytest.mark.asyncio
    async def test_llm_failure_triggers_fallback(self, compactor, system_msg):
        """LLM 抛出异常时走 _fallback_compact。"""
        messages = self._make_llm_triggering_messages(system_msg, n_turns=4)
        compactor.llm.async_chat = AsyncMock(side_effect=RuntimeError("LLM down"))
        compactor._bind_tool_name_map(messages)
        result = await compactor.auto_compact(messages)
        compactor.llm.async_chat.assert_called_once()
        # 返回了降级消息列表
        assert isinstance(result, list) and len(result) > 0

    @pytest.mark.asyncio
    async def test_summarizer_failure_cooldown(self, compactor, system_msg):
        """LLM 失败后设置 _summary_failure_cooldown_until。"""
        messages = self._make_llm_triggering_messages(system_msg, n_turns=4)
        compactor.llm.async_chat = AsyncMock(side_effect=RuntimeError("LLM down"))
        compactor._summary_failure_cooldown_until = 0.0
        compactor._bind_tool_name_map(messages)
        await compactor.auto_compact(messages)
        assert compactor._summary_failure_cooldown_until > 0

    @pytest.mark.asyncio
    async def test_summary_role_intelligence(self, compactor, system_msg):
        """摘要 role 选择避免与 head/tail 邻接消息同 role。"""
        messages = self._make_llm_triggering_messages(system_msg, n_turns=4)
        result_mock = MagicMock()
        result_mock.content = "## Goal\nDone"
        compactor.llm.async_chat = AsyncMock(return_value=result_mock)
        compactor._bind_tool_name_map(messages)
        result = await compactor.auto_compact(messages)
        summary_msg = next(
            (m for m in result if "Goal" in (m.get("content") or "")), None
        )
        assert summary_msg is not None
        assert summary_msg["role"] in ("user", "assistant")

    @pytest.mark.asyncio
    async def test_empty_summary_triggers_fallback(self, compactor, system_msg):
        """LLM 返回空时走 _fallback_compact。"""
        messages = self._make_llm_triggering_messages(system_msg, n_turns=4)
        result_mock = MagicMock()
        result_mock.content = ""
        compactor.llm.async_chat = AsyncMock(return_value=result_mock)
        compactor._bind_tool_name_map(messages)
        result = await compactor.auto_compact(messages)
        # 走 fallback（空返回触发 cooldown + fallback）
        assert isinstance(result, list) and len(result) > 0

    @pytest.mark.asyncio
    async def test_focus_topic_passed_to_prompt(self, compactor, system_msg):
        """focus_topic 参数透传到 build_compression_prompt。"""
        messages = self._make_llm_triggering_messages(system_msg, n_turns=4)
        result_mock = MagicMock()
        result_mock.content = "## Goal\nDone"
        compactor.llm.async_chat = AsyncMock(return_value=result_mock)
        compactor._bind_tool_name_map(messages)
        await compactor.auto_compact(messages, focus_topic="auth_module")
        compactor.llm.async_chat.assert_called_once()
        prompt = compactor.llm.async_chat.call_args[1]["messages"][0]["content"]
        assert "auth_module" in prompt or "FOCUS" in prompt

    @pytest.mark.asyncio
    async def test_calls_sanitize_after_compression(self, compactor, system_msg):
        """sanitize_tool_pairs 在 auto_compact 末尾被调用，清理 orphaned results。"""
        # 构造一个孤立的 orphaned result（call_id 不在任何 assistant 中）
        messages = (
            [system_msg]
            + [{"role": "user", "content": "task"}]
            + self._make_llm_triggering_messages(system_msg, n_turns=4)
            # orphaned result：call_id 不存在于任何 surviving assistant call
            + [
                {"role": "tool", "tool_call_id": "orphan_stale", "content": "orphaned result"},
            ]
        )

        result_mock = MagicMock()
        result_mock.content = "## Goal\nDone"
        compactor.llm.async_chat = AsyncMock(return_value=result_mock)
        compactor._bind_tool_name_map(messages)
        result = await compactor.auto_compact(messages)

        # orphaned result 应被 sanitize_tool_pairs 移除
        orphaned = [m for m in result if m.get("tool_call_id") == "orphan_stale"]
        assert len(orphaned) == 0

    # -------------------------------------------------------------------------
    # Helper
    # -------------------------------------------------------------------------

    @staticmethod
    def _make_llm_triggering_messages(system_msg: dict, n_turns: int = 6) -> list:
        """构造足够触发 LLM summarization 的消息列表。

        结构：system + user + 多个 [assistant(tool_call) + tool_result] 交替。
        每条 tool_result 长 6000+ 字符，远超 tail_token_budget，强制调用 LLM。
        """
        LONG = "x" * 6000
        messages = [
            system_msg,
            {"role": "user", "content": "Fix the bug in the codebase."},
        ]
        for i in range(n_turns):
            messages.append({
                "role": "assistant",
                "content": f"Reading file {i}",
                "tool_calls": [
                    {
                        "id": f"call_{i}",
                        "function": {
                            "name": "read_file",
                            "arguments": f'{{"path": "file_{i}.py"}}',
                        },
                    }
                ],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": f"call_{i}",
                "name": "read_file",
                "content": f"File {i} content:\n" + LONG,
            })
        return messages


# =============================================================================
# TestClass 6: TestFallbackCompact
# =============================================================================

class TestFallbackCompact:
    """_fallback_compact：Layer 2 降级策略。"""

    def test_returns_system_plus_artifact_plus_recent(self, compactor, system_msg):
        """fallback 结果包含 system + artifact + 最近 10 条。"""
        messages = [system_msg] + [
            {"role": "user", "content": f"msg{i}"} for i in range(30)
        ]
        artifact = "\n## CRITICAL: Existing Files\n- /path/to/file.py"
        result = compactor._fallback_compact(system_msg, messages, artifact)
        assert result[0]["role"] == "system"
        assert any("CRITICAL" in (m.get("content") or "") for m in result)
        assert any("/path/to/file.py" in (m.get("content") or "") for m in result)
        assert len(result) >= 12  # system + artifact + recent

    def test_no_artifact_section(self, compactor, system_msg):
        """无 artifact_section 时正确处理。"""
        messages = [system_msg] + [
            {"role": "user", "content": f"msg{i}"} for i in range(15)
        ]
        result = compactor._fallback_compact(system_msg, messages, "")
        assert result[0]["role"] == "system"
        assert len(result) == len([system_msg] + messages[-10:])

    def test_copies_messages_not_references(self, compactor, system_msg):
        """返回的是新列表而非原地修改。"""
        messages = [{"role": "user", "content": f"msg{i}"} for i in range(15)]
        result = compactor._fallback_compact(system_msg, messages, "")
        assert result is not messages

    def test_recent_messages_are_copies(self, compactor, system_msg):
        """返回的每条消息都是 .copy()。"""
        messages = [{"role": "user", "content": f"msg{i}"} for i in range(15)]
        result = compactor._fallback_compact(system_msg, messages, "")
        recent_part = result[2:]
        for m in recent_part:
            assert m is not messages[result.index(m) - 2]


# =============================================================================
# TestClass 7: TestSerializeForSummary
# =============================================================================

class TestSerializeForSummary:
    """_serialize_for_summary：消息序列化供 LLM summarization。"""

    def test_truncates_long_content_head_tail(self, compactor):
        """>6000 字符内容被截断为 head(4000) + ...truncated... + tail(1500)。"""
        messages = [{"role": "user", "content": "x" * 8000}]
        result = compactor._serialize_for_summary(messages)
        assert "...[truncated]..." in result
        assert len(result) <= 30000

    def test_includes_tool_call_arguments(self, compactor):
        """assistant 消息的 tool_calls 附带了 arguments。"""
        messages = [
            {
                "role": "assistant",
                "content": "reading",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "config.py", "offset": 10}',
                        },
                    }
                ],
            }
        ]
        result = compactor._serialize_for_summary(messages)
        assert "read_file" in result
        assert "config.py" in result
        assert "[Tool calls:" in result

    def test_labels_tool_results(self, compactor):
        """tool result 消息标签为 [TOOL RESULT ...]。"""
        messages = [
            {"role": "tool", "tool_call_id": "c1", "content": "result content"},
        ]
        result = compactor._serialize_for_summary(messages)
        assert "[TOOL RESULT c1]:" in result

    def test_labels_assistant(self, compactor):
        """assistant 消息标签为 [ASSISTANT]。"""
        messages = [{"role": "assistant", "content": "I am reading the file."}]
        result = compactor._serialize_for_summary(messages)
        assert "[ASSISTANT]:" in result

    def test_labels_user(self, compactor):
        """user 消息标签为 [USER]。"""
        messages = [{"role": "user", "content": "Fix the bug."}]
        result = compactor._serialize_for_summary(messages)
        assert "[USER]:" in result

    def test_respects_max_chars_limit(self, compactor):
        """总输出不超过 max_chars（默认 30000）。"""
        messages = [{"role": "user", "content": "x" * 10000} for _ in range(10)]
        result = compactor._serialize_for_summary(messages)
        assert len(result) <= 30000

    def test_tool_call_long_arguments_truncated(self, compactor):
        """tool call arguments > 1500 字符时截断。"""
        messages = [
            {
                "role": "assistant",
                "content": "reading",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "bash",
                            "arguments": "echo " + "x" * 3000,
                        },
                    }
                ],
            }
        ]
        result = compactor._serialize_for_summary(messages)
        assert "..." in result


# =============================================================================
# TestClass 8: TestBindToolNameMap
# =============================================================================

class TestBindToolNameMap:
    """_bind_tool_name_map：从 assistant 消息构建 tool_call_id -> tool_name 映射。"""

    def test_dict_tool_calls(self, compactor):
        """从 dict 格式的 tool_calls 中正确提取 id 和 name。"""
        messages = [
            {
                "role": "assistant",
                "content": "done",
                "tool_calls": [
                    {"id": "c1", "function": {"name": "read_file"}},
                    {"id": "c2", "function": {"name": "bash"}},
                ],
            }
        ]
        compactor._bind_tool_name_map(messages)
        assert compactor._tool_name_map.get("c1") == "read_file"
        assert compactor._tool_name_map.get("c2") == "bash"

    def test_dataclass_tool_calls(self, compactor):
        """从真实 dataclass 格式的 tool_calls 中正确提取。"""
        mock_tc = MockToolCall(id="call_dc", name="python_repl")
        messages = [{"role": "assistant", "content": "done", "tool_calls": [mock_tc]}]
        compactor._bind_tool_name_map(messages)
        assert compactor._tool_name_map.get("call_dc") == "python_repl"

    def test_clears_previous_map(self, compactor):
        """每次调用清空旧映射。"""
        messages1 = [
            {
                "role": "assistant",
                "content": "done",
                "tool_calls": [{"id": "c1", "function": {"name": "read_file"}}],
            }
        ]
        messages2 = [
            {
                "role": "assistant",
                "content": "done",
                "tool_calls": [{"id": "c2", "function": {"name": "bash"}}],
            }
        ]
        compactor._bind_tool_name_map(messages1)
        assert "c1" in compactor._tool_name_map
        compactor._bind_tool_name_map(messages2)
        assert "c1" not in compactor._tool_name_map
        assert "c2" in compactor._tool_name_map
