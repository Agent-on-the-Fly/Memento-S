# SPDX-License-Identifier: Apache-2.0
"""infra/context/providers — Context 内部子模块。

Compactor: Pre-API 压缩流水线 (ABC + 实现)
Artifact: Tool Result 持久化 (ABC + 实现)
Microcompact: 零 LLM 内容清除 (实现)
Shared: prompts + utils (纯函数，无 core/ 依赖)
"""

from infra.compact.models import CompactTrigger
from infra.context.providers.compactor import (
    CompactorProvider,
    CompactorStats,
)
from infra.context.providers.compactor_impl import TokenBudgetCompactor
from infra.context.providers.artifact import ArtifactProvider
from infra.context.providers.artifact_impl import ArtifactProviderImpl

__all__ = [
    "CompactorProvider",
    "CompactorStats",
    "CompactTrigger",
    "TokenBudgetCompactor",
    "ArtifactProvider",
    "ArtifactProviderImpl",
]
