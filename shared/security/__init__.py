# SPDX-License-Identifier: Apache-2.0
"""Shared security module — policy engine and security checks.

职责定位：
- 提供跨模块统一的"是否允许执行"判定（allow/deny）
- 聚合通用安全策略（危险命令、文件操作边界）
- 提供 per-action 速率限制

边界约束：
- 本模块只做"策略判定"，不做路径归一化/参数清洗。
- 路径解析、输入校验等工具函数放在 shared/tools 模块。
"""

from shared.security.policy import (
    PolicyFunc,
    PolicyResult,
    RateLimit,
    PolicyManager,
    block_dangerous_bash,
)

__all__ = [
    "PolicyFunc",
    "PolicyResult",
    "RateLimit",
    "PolicyManager",
    "block_dangerous_bash",
]
