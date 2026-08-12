# SPDX-License-Identifier: Apache-2.0
"""压缩管道 — 渐进式压缩编排引擎。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from infra.compact.config import CompactConfig
from infra.compact.models import CompactBudget, CompactContext, CompactResult, CompactTrigger
from infra.compact.strategies.base import BaseStrategy
from infra.compact.strategies.fallback import FallbackStrategy
from infra.compact.strategies.llm_emergency import LLMEmergencyStrategy
from infra.compact.strategies.llm_single import LLMSingleStrategy
from infra.compact.strategies.zero_llm import ZeroLLMStrategy
from infra.compact.utils import count_messages_tokens

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """Pipeline.run() 的返回结果 — 与原有 PreApiPipeline.PipelineResult 兼容。"""

    messages_for_api: list[dict[str, Any]]
    was_compacted: bool = False
    compact_trigger: CompactTrigger | None = None
    tokens_before: int = 0
    tokens_after: int = 0
    metadata: dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class CompactPipeline:
    """压缩管道 — 渐进式执行多种压缩策略。

    策略执行顺序:
    1. ZeroLLMStrategy — Microcompact (零 LLM)
    2. LLMSingleStrategy — 单消息压缩 (LLM)
    3. LLMEmergencyStrategy — 13 字段结构化摘要 (LLM，支持迭代压缩)
    4. FallbackStrategy — token-budget-aware group-aware-trim (最后防线)

    Circuit Breaker: 连续失败 N 次后，跳过 LLM 策略直接执行 Fallback。
    Anti-Thrashing: 无效压缩 (<10% 节省) 累积退避。
    """

    def __init__(
        self,
        config: CompactConfig,
        *,
        strategies: dict[CompactTrigger, BaseStrategy] | None = None,
        observers: list[Any] | None = None,
    ) -> None:
        """初始化压缩管道。

        Args:
            config: 压缩配置
            strategies: 可选的策略字典 (覆盖默认策略)
            observers: 压缩回调列表
        """
        self._config = config
        self._observers = observers or []

        # 默认策略组合
        if strategies is None:
            strategies = self._build_default_strategies()
        self._strategies = strategies

        # Circuit Breaker 状态
        self._consecutive_failures: int = 0
        self._last_failure_time: float = 0.0

        # 迭代压缩状态 (共享给 LLMEmergencyStrategy)
        self._previous_summary: str | None = None

    def _build_default_strategies(self) -> dict[CompactTrigger, BaseStrategy]:
        """构建默认策略组合。"""
        emergency = LLMEmergencyStrategy(self._config)
        return {
            CompactTrigger.MICRO: ZeroLLMStrategy(self._config),
            CompactTrigger.SM: ZeroLLMStrategy(self._config),
            CompactTrigger.SINGLE: LLMSingleStrategy(self._config),
            CompactTrigger.EMERGENCY: emergency,
            CompactTrigger.TRIM: FallbackStrategy(self._config),
        }

    async def run(
        self,
        messages: list[dict[str, Any]],
        budget: CompactBudget,
        *,
        force_compact: bool = False,
    ) -> PipelineResult:
        """运行压缩管道。

        Args:
            messages: 原始消息列表
            budget: 压缩预算
            force_compact: 强制压缩 (跳过预算检查)

        Returns:
            PipelineResult 包含压缩后的消息和元数据
        """
        tokens_before = count_messages_tokens(messages, self._config)
        working = list(messages)

        # Step 1: 检查是否需要压缩
        if not force_compact and tokens_before <= budget.total_limit:
            return PipelineResult(
                messages_for_api=working,
                was_compacted=False,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
            )

        # Step 2: 同步迭代摘要状态到 LLMEmergencyStrategy
        emergency = self._strategies.get(CompactTrigger.EMERGENCY)
        if emergency is not None and hasattr(emergency, "_previous_summary"):
            emergency._previous_summary = self._previous_summary

        # Step 3: 执行渐进式压缩
        result = await self._run_progressive_compact(working, budget)

        # Step 4: 更新 Circuit Breaker + Anti-Thrashing 状态
        if result.was_compacted:
            self._consecutive_failures = 0
            await self._notify_observers_before(result, tokens_before)

            # 提取并保存迭代摘要供下次使用
            if emergency is not None and hasattr(emergency, "_previous_summary"):
                self._previous_summary = emergency._previous_summary

            # Anti-thrashing: 记录 savings
            savings = tokens_before - result.tokens_after
            savings_pct = (savings / tokens_before * 100) if tokens_before > 0 else 0
            if savings_pct < 10:
                self._consecutive_failures += 1
                self._last_failure_time = time.monotonic()
        else:
            self._consecutive_failures += 1
            self._last_failure_time = time.monotonic()

        tokens_after = count_messages_tokens(result.messages_for_api, self._config)

        return PipelineResult(
            messages_for_api=result.messages_for_api,
            was_compacted=result.was_compacted,
            compact_trigger=result.compact_trigger,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            metadata={
                "trigger": result.compact_trigger.value if result.compact_trigger else None,
                "saved_tokens": tokens_before - tokens_after,
            },
        )

    async def _run_progressive_compact(
        self,
        messages: list[dict[str, Any]],
        budget: CompactBudget,
    ) -> PipelineResult:
        """渐进式执行压缩策略。"""
        # Circuit Breaker 检查
        if self._is_circuit_broken():
            logger.warning(
                "Compact circuit breaker active ({} failures, {:.0f}s cooldown remaining)",
                self._consecutive_failures,
                self._breaker_remaining(),
            )
            strategy = self._strategies[CompactTrigger.TRIM]
            result = await strategy.compact(messages, budget)
            return PipelineResult(
                messages_for_api=result.messages,
                was_compacted=True,
                compact_trigger=result.trigger,
            )

        # 构建压缩上下文
        context = CompactContext(
            messages=list(messages),
            budget=budget,
            original_count=len(messages),
        )

        # 策略执行顺序: ZeroLLM -> Single -> Emergency -> Fallback
        strategy_order: list[CompactTrigger] = [
            CompactTrigger.MICRO,
            CompactTrigger.SINGLE,
            CompactTrigger.EMERGENCY,
            CompactTrigger.TRIM,
        ]

        for trigger in strategy_order:
            strategy = self._strategies.get(trigger)
            if strategy is None:
                continue

            try:
                result = await strategy.compact(context.messages, context.budget, context)

                # 检查是否在预算内
                result_tokens = count_messages_tokens(result.messages, self._config)
                if result_tokens <= budget.total_limit:
                    logger.info(
                        "Compact succeeded: trigger={}, {} msgs -> {} msgs, tokens {} -> {}",
                        result.trigger.value,
                        len(messages),
                        len(result.messages),
                        count_messages_tokens(messages, self._config),
                        result_tokens,
                    )
                    return PipelineResult(
                        messages_for_api=result.messages,
                        was_compacted=True,
                        compact_trigger=result.trigger,
                    )

                # 更新上下文用于下次策略
                context.messages = result.messages
                context.update_budget(result_tokens)

            except Exception as exc:
                logger.opt(exception=True).warning(
                    "Strategy {} failed: {}", trigger.value, exc
                )
                self._consecutive_failures += 1
                self._last_failure_time = time.monotonic()
                continue

        # 所有策略都失败或未能满足预算，执行 Fallback
        logger.warning(
            "All compact strategies failed, executing fallback ({} failures)",
            self._consecutive_failures,
        )
        strategy = self._strategies[CompactTrigger.TRIM]
        result = await strategy.compact(messages, budget)
        return PipelineResult(
            messages_for_api=result.messages,
            was_compacted=True,
            compact_trigger=CompactTrigger.TRIM,
        )

    def _is_circuit_broken(self) -> bool:
        """检查 Circuit Breaker 是否激活。"""
        if self._consecutive_failures < self._config.max_compact_failures:
            return False

        elapsed = time.monotonic() - self._last_failure_time
        return elapsed < self._config.breaker_cooldown_s

    def _breaker_remaining(self) -> float:
        """获取 Circuit Breaker 剩余时间。"""
        if self._consecutive_failures < self._config.max_compact_failures:
            return 0.0
        elapsed = time.monotonic() - self._last_failure_time
        return max(0.0, self._config.breaker_cooldown_s - elapsed)

    async def _notify_observers_before(
        self,
        result: PipelineResult,
        tokens_before: int,
    ) -> None:
        """通知所有观察者 (压缩前)。"""
        for observer in self._observers:
            try:
                await observer.on_before_compact(
                    result.messages_for_api,
                    result.compact_trigger,
                )
            except Exception as exc:
                logger.warning("Observer {} failed: {}", observer, exc)

    async def _notify_observers_after(
        self,
        result: PipelineResult,
        tokens_before: int,
        tokens_after: int,
    ) -> None:
        """通知所有观察者 (压缩后)。"""
        for observer in self._observers:
            try:
                await observer.on_after_compact(
                    result,
                    tokens_before,
                    tokens_after,
                )
            except Exception as exc:
                logger.warning("Observer {} failed: {}", observer, exc)

    def reset_circuit_breaker(self) -> None:
        """重置 Circuit Breaker 和迭代摘要状态 (调用在 /new 或 /reset)。"""
        self._consecutive_failures = 0
        self._last_failure_time = 0.0
        self._previous_summary = None
        # 重置 LLMEmergencyStrategy 的内部状态
        emergency = self._strategies.get(CompactTrigger.EMERGENCY)
        if emergency is not None and hasattr(emergency, "reset_state"):
            emergency.reset_state()

    def get_status(self) -> dict[str, Any]:
        """返回管道状态快照 (用于可观测性)。"""
        emergency = self._strategies.get(CompactTrigger.EMERGENCY)
        comp_count = 0
        if emergency is not None and hasattr(emergency, "compression_count"):
            comp_count = getattr(emergency, "compression_count", 0)
        return {
            "consecutive_failures": self._consecutive_failures,
            "breaker_remaining_s": round(self._breaker_remaining(), 1),
            "has_iterative_summary": self._previous_summary is not None,
        }
