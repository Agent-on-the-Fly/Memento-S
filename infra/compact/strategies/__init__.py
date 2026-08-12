# SPDX-License-Identifier: Apache-2.0
"""策略实现包。"""

from infra.compact.strategies.base import BaseStrategy
from infra.compact.strategies.zero_llm import ZeroLLMStrategy
from infra.compact.strategies.llm_single import LLMSingleStrategy
from infra.compact.strategies.llm_emergency import LLMEmergencyStrategy
from infra.compact.strategies.fallback import FallbackStrategy

__all__ = [
    "BaseStrategy",
    "ZeroLLMStrategy",
    "LLMSingleStrategy",
    "LLMEmergencyStrategy",
    "FallbackStrategy",
]
