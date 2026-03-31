"""Shared policy engine (decision layer).

职责定位：
- 提供跨模块统一的“是否允许执行”判定（allow/deny）
- 聚合通用安全策略（危险命令、文件操作边界）
- 提供 per-action 速率限制

边界约束：
- 本模块只做“策略判定”，不做路径归一化/参数清洗。
- 路径解析、输入校验等工具函数放在 `core.shared.tool_security`。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from middleware.config import g_config


class PolicyFunc(Protocol):
    """Policy function protocol."""

    def __call__(self, action_name: str, args: dict[str, Any]) -> bool: ...


@dataclass(slots=True)
class PolicyResult:
    """Policy check result."""

    allowed: bool
    reason: str = ""


@dataclass
class RateLimit:
    """Per-tool rate limit configuration."""

    max_calls: int
    window_secs: float
    _timestamps: list[float] = field(default_factory=list, repr=False)

    def check(self) -> bool:
        """Return True if call is within rate limit."""
        now = time.monotonic()
        cutoff = now - self.window_secs
        self._timestamps = [t for t in self._timestamps if t > cutoff]
        if len(self._timestamps) >= self.max_calls:
            return False
        self._timestamps.append(now)
        return True


_DEFAULT_RATE_LIMITS: dict[str, tuple[int, float]] = {
    "bash": (200, 60.0),
    "file_create": (200, 6.0),
    "edit_file_by_lines": (200, 6.0),
    "fetch_webpage": (200, 60.0),
}


_BASH_EXACT_PATTERNS = [
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd if=",
    ":(){ :|:& };:",
    "format c:",
    "format d:",
    "rd /s /q c:\\",
    "rd /s /q \\",
]

_BASH_SUBSTRING_PATTERNS = [
    "sudo rm -rf",
    "chmod 777",
    "chmod -r 777",
    "> /dev/sda",
    "mv /* ",
    "rm -rf ~",
    "rm -rf $home",
    "shutdown",
    "reboot",
    "init 0",
    "init 6",
    "del /f /s /q",
    "rd /s /q",
    "reg delete",
    "bcdedit",
    "git push --force",
    "git push -f",
    "drop table",
    "drop database",
    "truncate table",
]

_BASH_INJECTION_PATTERNS = [
    re.compile(r"\$\(curl\s"),
    re.compile(r"`curl\s"),
    re.compile(r"base64\s.*\|\s*sh"),
    re.compile(r"base64\s.*\|\s*bash"),
    re.compile(r"eval\s+[\"']"),
    re.compile(r"\|\s*sh\b"),
    re.compile(r"\|\s*bash\b"),
    re.compile(r"curl.*\|\s*(sh|bash)"),
    re.compile(r"wget.*\|\s*(sh|bash)"),
    re.compile(r"\\x[0-9a-f]{2}", re.IGNORECASE),
]


def block_dangerous_bash(action_name: str, args: dict[str, Any]) -> bool:
    """Block destructive shell patterns with multi-layer detection."""
    if action_name != "bash":
        return True

    command = str(args.get("command", ""))
    lowered = command.lower()

    if any(p in lowered for p in _BASH_EXACT_PATTERNS):
        return False
    if any(p in lowered for p in _BASH_SUBSTRING_PATTERNS):
        return False
    if any(pat.search(command) for pat in _BASH_INJECTION_PATTERNS):
        return False

    return True


class PolicyManager:
    """Unified policy manager with built-in policy registration and rate limits."""

    def __init__(
        self,
        rate_limit_overrides: dict[str, tuple[int, float]] | None = None,
    ) -> None:
        self._policies: list[tuple[str, PolicyFunc]] = []
        self._rate_limits: dict[str, RateLimit] = {}
        self._register_builtin_policies()
        self._init_rate_limits(overrides=rate_limit_overrides)

    def _register_builtin_policies(self) -> None:
        self._register(block_dangerous_bash, name="block_dangerous_bash")
        # Note: restrict_file_ops removed - path validation now handled by PathValidator
        # in ToolBridge.execute() using dynamic allow_roots from ToolContext

    def _init_rate_limits(
        self,
        overrides: dict[str, tuple[int, float]] | None = None,
    ) -> None:
        config = dict(_DEFAULT_RATE_LIMITS)
        if overrides:
            config.update(overrides)
        for tool_name, (max_calls, window) in config.items():
            self._rate_limits[tool_name] = RateLimit(
                max_calls=max_calls,
                window_secs=window,
            )

    def _register(self, policy: PolicyFunc, *, name: str | None = None) -> None:
        policy_name = name or getattr(policy, "__name__", "anonymous_policy")
        self._policies.append((policy_name, policy))

    def check(self, action_name: str, args: dict[str, Any]) -> PolicyResult:
        """Check if action is allowed by rate limits and policies."""
        rate_limit = self._rate_limits.get(action_name)
        if rate_limit and not rate_limit.check():
            return PolicyResult(
                allowed=False,
                reason=(
                    f"Rate limit exceeded for '{action_name}' "
                    f"({rate_limit.max_calls} calls per {rate_limit.window_secs}s)"
                ),
            )

        for policy_name, policy in self._policies:
            try:
                ok = bool(policy(action_name, args))
            except Exception as e:
                return PolicyResult(
                    allowed=False,
                    reason=f"Policy '{policy_name}' raised exception: {e}",
                )
            if not ok:
                return PolicyResult(
                    allowed=False,
                    reason=f"Policy '{policy_name}' denied action '{action_name}'",
                )

        return PolicyResult(allowed=True)
