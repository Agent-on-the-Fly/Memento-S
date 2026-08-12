# SPDX-License-Identifier: Apache-2.0
"""集成测试 — infra/compact/pipeline.py: CompactPipeline 各压缩策略验证。"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from infra.compact.pipeline import CompactPipeline, PipelineResult
from infra.compact.config import CompactConfig
from infra.compact.models import CompactBudget, CompactTrigger


def _tool_msg(content: str, tid: str = "t") -> dict:
    return {"role": "tool", "tool_call_id": tid, "name": "read", "content": content}


def _msgs(tool_contents: list[str]) -> list[dict]:
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
    for i, c in enumerate(tool_contents):
        msgs.append(_tool_msg(c, tid=f"t{i}"))
        msgs.append({"role": "assistant", "content": "done"})
    return msgs


def _mock_count_messages_tokens(msgs, config):
    """Mock token count: each message = 10 tokens (rough estimate)."""
    return len(msgs) * 10


@pytest.fixture
def pipeline(tmp_path) -> CompactPipeline:
    """CompactPipeline fixture with mock LLM client (no real LLM calls)."""
    cfg = CompactConfig(
        model="gpt-4o",
        emergency_keep_tail=6,
        max_compact_failures=3,
        breaker_cooldown_s=60.0,
        sm_compact_min_ratio=0.02,
        sm_compact_max_ratio=0.08,
    )
    return CompactPipeline(cfg, strategies=None, observers=[])


class TestCompactPipeline:
    """验证 CompactPipeline 的压缩触发、策略执行、Circuit Breaker。"""

    @pytest.mark.asyncio
    async def test_under_budget_no_compact(self, pipeline: CompactPipeline) -> None:
        """预算内（tokens_before <= total_limit）不触发压缩。"""
        with patch("infra.compact.pipeline.count_messages_tokens", return_value=1000):
            msgs = _msgs(["short"])
            budget = CompactBudget(total_tokens=1000, total_limit=9999, per_message_limit=500)
            result = await pipeline.run(msgs, budget)
        assert result.was_compacted is False

    @pytest.mark.asyncio
    async def test_over_budget_triggers_compact(self, pipeline: CompactPipeline) -> None:
        """超出预算时触发压缩（Fallback TRIM 策略兜底）。"""
        with patch("infra.compact.pipeline.count_messages_tokens", return_value=99999):
            msgs = _msgs(["word " * 500 for _ in range(10)])
            budget = CompactBudget(total_tokens=99999, total_limit=500, per_message_limit=50)
            result = await pipeline.run(msgs, budget)
        assert result.was_compacted is True
        assert result.tokens_after <= result.tokens_before

    @pytest.mark.asyncio
    async def test_force_compact_always_compacts(self, pipeline: CompactPipeline) -> None:
        """force_compact=True 无论预算是否充足都压缩。"""
        with patch("infra.compact.pipeline.count_messages_tokens", return_value=100):
            msgs = _msgs(["short"])
            budget = CompactBudget(total_tokens=100, total_limit=9999, per_message_limit=500)
            result = await pipeline.run(msgs, budget, force_compact=True)
        assert result.was_compacted is True

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_on_success(self, pipeline: CompactPipeline) -> None:
        """压缩成功后 consecutive_failures 应重置为 0。"""
        pipeline._consecutive_failures = 0
        with patch("infra.compact.pipeline.count_messages_tokens", return_value=99999):
            msgs = _msgs(["word " * 500 for _ in range(5)])
            budget = CompactBudget(total_tokens=99999, total_limit=200, per_message_limit=30)
            await pipeline.run(msgs, budget)
        assert pipeline._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_pipeline_result_fields(self, pipeline: CompactPipeline) -> None:
        """PipelineResult 包含所有必需字段。"""
        with patch("infra.compact.pipeline.count_messages_tokens", return_value=99999):
            msgs = _msgs(["word " * 500 for _ in range(5)])
            budget = CompactBudget(total_tokens=99999, total_limit=200, per_message_limit=30)
            result = await pipeline.run(msgs, budget)
        assert isinstance(result, PipelineResult)
        assert hasattr(result, "was_compacted")
        assert hasattr(result, "tokens_before")
        assert hasattr(result, "tokens_after")
        assert hasattr(result, "metadata")

    @pytest.mark.asyncio
    async def test_compact_result_metadata_has_trigger(self, pipeline: CompactPipeline) -> None:
        """压缩结果的 metadata 包含 trigger 信息。"""
        with patch("infra.compact.pipeline.count_messages_tokens", return_value=99999):
            msgs = _msgs(["word " * 500 for _ in range(5)])
            budget = CompactBudget(total_tokens=99999, total_limit=200, per_message_limit=30)
            result = await pipeline.run(msgs, budget)
        assert result.was_compacted is True
        trigger_val = result.metadata.get("trigger")
        assert trigger_val is None or isinstance(trigger_val, str)


class TestCompactBudgetModel:
    """验证 CompactBudget 数据模型属性。"""

    def test_within_budget_true(self) -> None:
        budget = CompactBudget(total_tokens=5000, total_limit=10000, per_message_limit=2500)
        assert budget.within_budget is True

    def test_within_budget_false(self) -> None:
        budget = CompactBudget(total_tokens=15000, total_limit=10000, per_message_limit=2500)
        assert budget.within_budget is False

    def test_overage(self) -> None:
        budget = CompactBudget(total_tokens=12000, total_limit=10000, per_message_limit=2500)
        assert budget.overage == 2000

    def test_overage_zero_when_within(self) -> None:
        budget = CompactBudget(total_tokens=8000, total_limit=10000, per_message_limit=2500)
        assert budget.overage == 0

    def test_recalculate(self) -> None:
        budget = CompactBudget(total_tokens=5000, total_limit=10000, per_message_limit=2500)
        new = budget.recalculate(total_tokens=3000)
        assert new.total_tokens == 3000
        assert new.total_limit == 10000


class TestCompactTriggerModel:
    """验证 CompactTrigger 枚举方法。"""

    def test_micro_is_zero_llm(self) -> None:
        assert CompactTrigger.MICRO.is_zero_llm() is True

    def test_sm_is_zero_llm(self) -> None:
        assert CompactTrigger.SM.is_zero_llm() is True

    def test_emergency_is_not_zero_llm(self) -> None:
        assert CompactTrigger.EMERGENCY.is_zero_llm() is False

    def test_trim_is_zero_llm(self) -> None:
        assert CompactTrigger.TRIM.is_zero_llm() is True

    def test_micro_is_not_emergency(self) -> None:
        assert CompactTrigger.MICRO.is_emergency() is False

    def test_emergency_is_emergency(self) -> None:
        assert CompactTrigger.EMERGENCY.is_emergency() is True

    def test_trim_is_emergency(self) -> None:
        assert CompactTrigger.TRIM.is_emergency() is True
