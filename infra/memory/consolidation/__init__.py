# SPDX-License-Identifier: Apache-2.0
"""infra/memory/consolidation — 唯一的整合逻辑入口。

由 Dream、Agent 或独立后台循环触发，engine 负责"怎么整合"。
"""

from __future__ import annotations

from .engine import ConsolidationContext, MemoryConsolidationEngine
from .loop import AutoConsolidationLoop
from .result_applier import ResultApplier

__all__ = [
    "ConsolidationContext",
    "MemoryConsolidationEngine",
    "AutoConsolidationLoop",
    "ResultApplier",
]
