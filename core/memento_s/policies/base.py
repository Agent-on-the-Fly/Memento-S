"""Policy system types and base classes."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .builtin_policies import block_dangerous_bash, restrict_file_ops


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


# Default rate limits per tool
_DEFAULT_RATE_LIMITS: dict[str, tuple[int, float]] = {
    "bash": (100, 60.0),
    "file_create": (200, 200.0),
    "edit_file_by_lines": (200, 200.0),
    "fetch_webpage": (60, 60.0),
}


class PolicyManager:
    """Policy manager with auto-registration of built-in policies.

    All policies are registered during initialization.
    Includes rate limiting per tool.
    """

    def __init__(
        self,
        rate_limit_overrides: dict[str, tuple[int, float]] | None = None,
    ) -> None:
        self._policies: list[tuple[str, PolicyFunc]] = []
        self._rate_limits: dict[str, RateLimit] = {}
        self._register_builtin_policies()
        self._init_rate_limits(overrides=rate_limit_overrides)

    def _register_builtin_policies(self) -> None:
        """Register all built-in policies."""
        self._register(block_dangerous_bash, name="block_dangerous_bash")
        self._register(restrict_file_ops, name="restrict_file_ops")

    def _init_rate_limits(
        self,
        overrides: dict[str, tuple[int, float]] | None = None,
    ) -> None:
        """Initialize rate limits with optional overrides."""
        config = dict(_DEFAULT_RATE_LIMITS)
        if overrides:
            config.update(overrides)
        for tool_name, (max_calls, window) in config.items():
            self._rate_limits[tool_name] = RateLimit(
                max_calls=max_calls, window_secs=window
            )

    def _register(self, policy: PolicyFunc, *, name: str | None = None) -> None:
        """Internal method to register a policy."""
        policy_name = name or getattr(policy, "__name__", "anonymous_policy")
        self._policies.append((policy_name, policy))

    def check(self, action_name: str, args: dict[str, Any]) -> PolicyResult:
        """Check if action is allowed by all policies and rate limits."""
        rate_limit = self._rate_limits.get(action_name)
        if rate_limit and not rate_limit.check():
            return PolicyResult(
                allowed=False,
                reason=f"Rate limit exceeded for '{action_name}' "
                       f"({rate_limit.max_calls} calls per {rate_limit.window_secs}s)",
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
