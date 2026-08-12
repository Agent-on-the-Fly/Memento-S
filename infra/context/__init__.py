# SPDX-License-Identifier: Apache-2.0
"""infra/context — 独立的 Context 抽象层。

提供 ContextProvider 抽象基类和具体实现，使 Context 管理与 Agent 解耦。
"""

from infra.context.base import ContextProvider
from infra.context.impl.file_context import FileContextProvider
from infra.context.factory import create_context, ContextFactoryConfig
from infra.context.providers.compactor import CompactorProvider
from infra.context.providers.artifact import ArtifactProvider

__all__ = [
    "ContextProvider",
    "FileContextProvider",
    "create_context",
    "ContextFactoryConfig",
    "CompactorProvider",
    "ArtifactProvider",
]
