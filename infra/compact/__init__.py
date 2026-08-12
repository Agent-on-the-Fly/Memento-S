# SPDX-License-Identifier: Apache-2.0
"""Compact Module — 独立的上下文压缩引擎。

支持多层级渐进式压缩:
- Zero-LLM: Microcompact, SM Compact
- LLM: Single Message, Emergency (9段摘要)
- Fallback: Group-Aware Trim

Usage:
    from infra.compact import CompactPipeline, CompactConfig, CompactBudget

    config = CompactConfig(...)
    pipeline = CompactPipeline(config, llm_client=..., token_counter=...)
    result = await pipeline.run(messages, budget)
"""

from infra.compact.models import (
    CompactBudget,
    CompactResult,
    CompactTrigger,
    MessageGroup,
    ToolResultReplacementState,
)
from infra.compact.config import CompactConfig
from infra.compact.constants import (
    DEFAULT_EMERGENCY_KEEP_TAIL,
    DEFAULT_FLOOR_TOKENS,
    DEFAULT_LLM_SINGLE_SUMMARY_TOKENS,
    DEFAULT_MICROCOMPACT_KEEP_RECENT,
    DEFAULT_PTL_TRUNCATE_RATIO,
    DEFAULT_SM_COMPACT_MAX_RATIO,
    DEFAULT_SM_COMPACT_MIN_RATIO,
    DEFAULT_SUMMARY_RATIO,
    DEFAULT_SUMMARY_TOKENS,
)
from infra.compact.pipeline import CompactPipeline
from infra.compact.abc import (
    CompactEngine,
    CompactObserver,
    LLMClient,
    TokenCounter,
    StorageBackend,
)
from infra.compact.observer import CompactObserverImpl
from infra.compact.strategies.base import BaseStrategy
from infra.compact.strategies.zero_llm import ZeroLLMStrategy
from infra.compact.strategies.llm_single import LLMSingleStrategy
from infra.compact.strategies.llm_emergency import LLMEmergencyStrategy
from infra.compact.strategies.fallback import FallbackStrategy

# CompactTrigger 映射: infra/compact -> core/context
# 解除 core 层对 infra 内部枚举的直接引用
_TRIGGER_MAP = {
    CompactTrigger.MICRO: "auto",
    CompactTrigger.SM: "auto",
    CompactTrigger.SINGLE: "auto",
    CompactTrigger.EMERGENCY: "emergency",
    CompactTrigger.TRIM: "emergency",
}


def compact_trigger_label(trigger: CompactTrigger | None) -> str | None:
    """将 CompactTrigger 映射为 core 层可用的字符串标签。

    隔离 infra/compact 内部枚举与 core 层，避免 core 层直接引用
    CompactTrigger 枚举值（如 CompactTrigger.MICRO）。
    """
    if trigger is None:
        return None
    return _TRIGGER_MAP.get(trigger)


__all__ = [
    # Core
    "CompactPipeline",
    "CompactConfig",
    "CompactBudget",
    "CompactResult",
    "CompactTrigger",
    "CompactEngine",
    "CompactObserver",
    "CompactObserverImpl",
    # Protocols
    "LLMClient",
    "TokenCounter",
    "StorageBackend",
    # Data classes
    "ToolResultReplacementState",
    # Strategies
    "BaseStrategy",
    "ZeroLLMStrategy",
    "LLMSingleStrategy",
    "LLMEmergencyStrategy",
    "FallbackStrategy",
    # Utils
    "MessageGroup",
    # Constants
    "DEFAULT_MICROCOMPACT_KEEP_RECENT",
    "DEFAULT_EMERGENCY_KEEP_TAIL",
    "DEFAULT_MAX_PTL_RETRIES",
    "DEFAULT_PTL_TRUNCATE_RATIO",
    "DEFAULT_SM_COMPACT_MIN_RATIO",
    "DEFAULT_SM_COMPACT_MAX_RATIO",
    "DEFAULT_SUMMARY_RATIO",
    "DEFAULT_SUMMARY_TOKENS",
    "DEFAULT_LLM_SINGLE_SUMMARY_TOKENS",
    "DEFAULT_FLOOR_TOKENS",
    # Utils
    "compact_trigger_label",
]
