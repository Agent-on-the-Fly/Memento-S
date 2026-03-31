"""Policy decision types for skill execution lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class PolicyStage(str, Enum):
    PRE_EXECUTE = "pre_execute"
    TOOL_GATE = "tool_gate"
    POST_EXECUTE = "post_execute"


class RecoveryAction(str, Enum):
    ABORT = "abort"
    RETRY = "retry"
    AUTO_FIX = "auto_fix"
    PROMPT_USER = "prompt_user"


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    stage: PolicyStage
    reason: str = ""
    detail: dict[str, Any] | None = None


@dataclass(frozen=True)
class RecoveryDecision:
    action: RecoveryAction
    reason: str
    detail: dict[str, Any] | None = None
