# SPDX-License-Identifier: Apache-2.0
"""infra/ — 独立的基础设施层，为 Agent 提供可插拔的 Context 和 Memory 实现。

目录结构:
    infra/
        config.py        — 配置模型
        service.py       — InfraService 单一入口类
        context/          — Context 抽象层
        memory/           — Memory 实现层
"""

from infra.context import ContextProvider
from infra.memory import SessionMemory, LongTermMemory, build_memory_context_block
from infra.service import InfraService, InfraContextConfig

__all__ = [
    "InfraService",
    "InfraContextConfig",
    "ContextProvider",
    "SessionMemory",
    "LongTermMemory",
    "build_memory_context_block",
]
