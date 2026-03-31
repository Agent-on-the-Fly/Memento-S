"""Tool gate checks during execution."""

from __future__ import annotations

from typing import Any

from core.skill.execution.policy.types import PolicyDecision, PolicyStage


class ToolGate:
    """Runtime tool-call gate backed by shared PolicyManager."""

    def __init__(self, *, policy_manager):
        self._policy_manager = policy_manager

    def check(self, tool_name: str, args: dict[str, Any]) -> PolicyDecision:
        policy = self._policy_manager.check(tool_name, args)
        if not policy.allowed:
            return PolicyDecision(
                allowed=False,
                stage=PolicyStage.TOOL_GATE,
                reason=policy.reason or f"tool '{tool_name}' denied by policy",
                detail={"tool": tool_name},
            )
        return PolicyDecision(
            allowed=True,
            stage=PolicyStage.TOOL_GATE,
            reason="",
            detail={"tool": tool_name},
        )
