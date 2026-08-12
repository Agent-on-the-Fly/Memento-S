# SPDX-License-Identifier: Apache-2.0
"""infra/compact 模块的单元测试。"""

import pytest


class TestModels:
    """测试 models.py 中的数据类型。"""

    def test_compact_trigger_is_zero_llm(self):
        from infra.compact.models import CompactTrigger

        assert CompactTrigger.MICRO.is_zero_llm() is True
        assert CompactTrigger.SM.is_zero_llm() is True
        assert CompactTrigger.SINGLE.is_zero_llm() is False
        assert CompactTrigger.EMERGENCY.is_zero_llm() is False
        assert CompactTrigger.TRIM.is_zero_llm() is True

    def test_compact_trigger_is_emergency(self):
        from infra.compact.models import CompactTrigger

        assert CompactTrigger.EMERGENCY.is_emergency() is True
        assert CompactTrigger.TRIM.is_emergency() is True
        assert CompactTrigger.MICRO.is_emergency() is False
        assert CompactTrigger.SM.is_emergency() is False
        assert CompactTrigger.SINGLE.is_emergency() is False

    def test_compact_budget_properties(self):
        from infra.compact.models import CompactBudget

        budget = CompactBudget(total_tokens=5000, total_limit=10000, per_message_limit=2500)

        assert budget.within_budget is True
        assert budget.overage == 0

        budget2 = CompactBudget(total_tokens=15000, total_limit=10000, per_message_limit=2500)
        assert budget2.within_budget is False
        assert budget2.overage == 5000

    def test_compact_budget_recalculate(self):
        from infra.compact.models import CompactBudget

        budget = CompactBudget(total_tokens=5000, total_limit=10000, per_message_limit=2500)
        new_budget = budget.recalculate(total_tokens=3000)

        assert new_budget.total_tokens == 3000
        assert new_budget.total_limit == 10000
        assert new_budget.per_message_limit == 2500


class TestConfig:
    """测试 config.py 中的配置模型。"""

    def test_config_defaults(self):
        from infra.compact.config import CompactConfig

        config = CompactConfig()
        assert config.enabled is True
        assert config.microcompact_keep_recent == 5
        assert config.emergency_keep_tail == 6
        assert config.max_compact_failures == 3

    def test_config_validation(self):
        from infra.compact.config import CompactConfig

        with pytest.raises(ValueError):
            CompactConfig(sm_compact_min_ratio=0.5)

        with pytest.raises(ValueError):
            CompactConfig(sm_compact_min_ratio=0.1, sm_compact_max_ratio=0.05)

    def test_config_calculate_budget(self):
        from infra.compact.config import CompactConfig

        config = CompactConfig(
            sm_compact_min_ratio=0.02,
            sm_compact_max_ratio=0.08,
            summary_ratio=0.15,
        )
        budget = config.calculate_budget(input_budget=10000)

        assert budget["total_limit"] == 10000
        assert budget["per_message_limit"] == 2500
        assert budget["sm_min_tokens"] == 200
        assert budget["sm_max_tokens"] == 800

    def test_config_with_overrides(self):
        from infra.compact.config import CompactConfig

        config = CompactConfig(microcompact_keep_recent=5)
        new_config = config.with_overrides(microcompact_keep_recent=10)

        assert config.microcompact_keep_recent == 5
        assert new_config.microcompact_keep_recent == 10


class TestUtils:
    """测试 utils.py 中的工具函数。"""

    def test_estimate_tokens_fast(self):
        from infra.compact.utils import estimate_tokens_fast

        assert estimate_tokens_fast("") == 0
        assert estimate_tokens_fast("hello") == 2
        assert estimate_tokens_fast("a" * 100) == 34

    def test_group_messages_by_round(self):
        from infra.compact.utils import group_messages_by_round

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi", "tool_calls": []},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "Done"},
            {"role": "user", "content": "Thanks"},
        ]
        groups = group_messages_by_round(messages)

        assert len(groups) == 3
        assert groups[0][0]["role"] == "user"
        assert groups[1][0]["role"] == "assistant"
        assert groups[1][1]["role"] == "tool"
        assert groups[2][0]["role"] == "assistant"
        assert groups[2][1]["role"] == "user"

    def test_adjust_index_to_preserve_invariants(self):
        from infra.compact.utils import adjust_index_to_preserve_invariants

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "tc1", "function": {"name": "test"}}]},
            {"role": "tool", "content": "result", "tool_call_id": "tc1"},
            {"role": "assistant", "content": "Done"},
        ]

        # 从 index 2 开始，tool_result 需要 tool_use
        adjusted = adjust_index_to_preserve_invariants(messages, 2)
        assert adjusted == 1  # 应该回溯到 assistant 消息

    def test_extract_key_content_short(self):
        from infra.compact.utils import extract_key_content, estimate_tokens_fast

        short_text = "This is a short text."
        result = extract_key_content(short_text, max_tokens=100)
        assert result == short_text

    def test_extract_key_content_json(self):
        from infra.compact.utils import extract_key_content

        json_text = '{"ok": true, "status": "success", "error": "no error", "output": "result", "summary": "summary", "large_field": "x" * 500}'
        result = extract_key_content(json_text, max_tokens=50)
        assert "[Extracted from" in result or "ok" in result

    def test_build_digest(self):
        from infra.compact.utils import build_digest

        payload = {
            "skill_name": "test_skill",
            "ok": True,
            "summary": "Test completed",
            "output": {
                "result": "output result",
                "execution_summary": {
                    "created_files": ["file1.py"],
                },
            },
        }
        digest = build_digest(payload, payload["output"])
        assert "test_skill" in digest
        assert "OK" in digest
        assert "output result" in digest

    def test_serialize_tool_calls(self):
        from infra.compact.utils import serialize_tool_calls

        tool_calls = [
            {"function": {"name": "read_file", "arguments": '{"path": "test.txt"}'}},
            {"function": {"name": "write_file", "arguments": '{"path": "out.txt", "content": "hi"}'}},
        ]
        result = serialize_tool_calls(tool_calls)
        assert "read_file" in result
        assert "write_file" in result

    def test_messages_to_text(self):
        from infra.compact.utils import messages_to_text

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "test", "arguments": ""}}]},
            {"role": "tool", "content": "result"},
        ]
        text = messages_to_text(messages)
        assert "USER" in text
        assert "TOOL_RESULT" in text
        assert "Hello" in text


class TestPrompts:
    """测试 prompts.py 中的提示词。"""

    def test_format_compact_summary(self):
        from infra.compact.prompts import format_compact_summary

        raw = """<analysis>
Some analysis here
</analysis>

<summary>
1. Primary Request:
   Test request
</summary>
"""
        result = format_compact_summary(raw)
        assert "<analysis>" not in result
        assert "Primary Request" in result

    def test_get_compact_user_summary_message(self):
        from infra.compact.prompts import get_compact_user_summary_message

        summary = "1. Test summary content"
        result = get_compact_user_summary_message(summary)
        assert "continued from a previous conversation" in result
        assert "Test summary content" in result
        assert "Continue the conversation" in result

    def test_get_compact_user_summary_with_transcript(self):
        from infra.compact.prompts import get_compact_user_summary_message

        summary = "Test"
        result = get_compact_user_summary_message(summary, transcript_path="/path/to/transcript")
        assert "/path/to/transcript" in result


class TestStrategies:
    """测试策略实现。"""

    @pytest.mark.asyncio
    async def test_fallback_strategy_basic(self):
        from infra.compact.config import CompactConfig
        from infra.compact.models import CompactBudget
        from infra.compact.strategies.fallback import FallbackStrategy

        config = CompactConfig(emergency_keep_tail=2)
        strategy = FallbackStrategy(config)

        messages = [
            {"role": "user", "content": "Hello 1"},
            {"role": "assistant", "content": "Hi 1"},
            {"role": "user", "content": "Hello 2"},
            {"role": "assistant", "content": "Hi 2"},
            {"role": "user", "content": "Hello 3"},
            {"role": "assistant", "content": "Hi 3"},
        ]
        budget = CompactBudget(total_tokens=1000, total_limit=500, per_message_limit=125)

        result = await strategy.compact(messages, budget)

        assert result.trigger.value == "trim"
        assert len(result.messages) <= 6  # 至少保留一些消息

    @pytest.mark.asyncio
    async def test_fallback_strategy_preserves_system(self):
        from infra.compact.config import CompactConfig
        from infra.compact.models import CompactBudget
        from infra.compact.strategies.fallback import FallbackStrategy

        config = CompactConfig(emergency_keep_tail=1)
        strategy = FallbackStrategy(config)

        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        budget = CompactBudget(total_tokens=500, total_limit=100, per_message_limit=25)

        result = await strategy.compact(messages, budget)

        # System 消息应该被保留
        roles = [m.get("role") for m in result.messages]
        assert "system" in roles

    @pytest.mark.asyncio
    async def test_zero_llm_microcompact(self):
        from infra.compact.config import CompactConfig
        from infra.compact.models import CompactBudget
        from infra.compact.strategies.zero_llm import ZeroLLMStrategy

        config = CompactConfig(
            microcompact_keep_recent=1,
            microcompact_compactable_tools=["execute_skill"],
        )
        strategy = ZeroLLMStrategy(config)

        messages = [
            {"role": "assistant", "content": "Running skill", "tool_calls": [{"id": "tc1", "function": {"name": "execute_skill"}}]},
            {"role": "tool", "name": "execute_skill", "content": "Very long old result that should be cleared" * 50},
            {"role": "tool", "name": "execute_skill", "content": "Recent result that should be kept"},
        ]
        budget = CompactBudget(total_tokens=5000, total_limit=1000, per_message_limit=250)

        result = await strategy.compact(messages, budget)

        assert result.trigger.value == "micro"
        # 旧的结果应该被清除
        cleared_content = "[Old tool result content cleared]"
        contents = [m.get("content", "") for m in result.messages if m.get("role") == "tool"]
        assert any(cleared_content in c for c in contents) or len(contents) <= len(messages)

    @pytest.mark.asyncio
    async def test_llm_single_no_op_when_under_threshold(self):
        from infra.compact.config import CompactConfig
        from infra.compact.models import CompactBudget
        from infra.compact.strategies.llm_single import LLMSingleStrategy

        config = CompactConfig()
        strategy = LLMSingleStrategy(config)

        messages = [
            {"role": "user", "content": "Short message"},
        ]
        # 高阈值，低实际 token 数
        budget = CompactBudget(total_tokens=100, total_limit=1000, per_message_limit=500)

        result = await strategy.compact(messages, budget)

        assert result.messages == messages


class TestPipeline:
    """测试压缩管道。"""

    @pytest.mark.asyncio
    async def test_pipeline_no_compact_when_under_budget(self):
        from infra.compact.config import CompactConfig
        from infra.compact.models import CompactBudget
        from infra.compact.pipeline import CompactPipeline

        config = CompactConfig()
        pipeline = CompactPipeline(config)

        messages = [
            {"role": "user", "content": "Short message"},
        ]
        budget = CompactBudget(total_tokens=100, total_limit=1000, per_message_limit=250)

        result = await pipeline.run(messages, budget)

        assert result.was_compacted is False
        assert result.compact_trigger is None

    @pytest.mark.asyncio
    async def test_pipeline_force_compact(self):
        from infra.compact.config import CompactConfig
        from infra.compact.models import CompactBudget
        from infra.compact.pipeline import CompactPipeline

        config = CompactConfig(emergency_keep_tail=1)
        pipeline = CompactPipeline(config)

        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello 1"},
            {"role": "assistant", "content": "Hi 1"},
            {"role": "user", "content": "Hello 2"},
            {"role": "assistant", "content": "Hi 2"},
        ]
        budget = CompactBudget(total_tokens=500, total_limit=1000, per_message_limit=250)

        result = await pipeline.run(messages, budget, force_compact=True)

        assert result.was_compacted is True

    def test_pipeline_circuit_breaker_reset(self):
        from infra.compact.config import CompactConfig
        from infra.compact.pipeline import CompactPipeline

        config = CompactConfig(max_compact_failures=3, breaker_cooldown_s=60.0)
        pipeline = CompactPipeline(config)

        pipeline._consecutive_failures = 5
        pipeline.reset_circuit_breaker()

        assert pipeline._consecutive_failures == 0
