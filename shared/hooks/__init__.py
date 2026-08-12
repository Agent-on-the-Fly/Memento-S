# SPDX-License-Identifier: Apache-2.0
"""shared/hooks — 跨模块共享的 Hook 基础设施"""

from shared.hooks.types import HookEvent, HookPayload, HookResult
from shared.hooks.executor import HookDefinition, CommandHook, HookExecutor

__all__ = [
    "HookEvent",
    "HookPayload",
    "HookResult",
    "HookDefinition",
    "CommandHook",
    "HookExecutor",
]
