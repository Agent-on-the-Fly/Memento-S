"""Agent configuration — the single AgentConfig dataclass.

Phase-specific types (IntentMode, IntentResult, TaskPlan, etc.) live in their
respective phase modules to avoid circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.context.schemas import ContextConfig


@dataclass
class AgentConfig:
    """Agent runtime parameters — threaded through to all phase functions."""

    # Execution control (per plan step)
    max_react_per_step: int = 5
    max_replans: int = 2
    max_consecutive_exec_failures: int = 5
    max_duplicate_tool_calls: int = 2

    # Reflection limits
    reflection_input_chars: int = 15000
    reflection_max_tokens: int = 30000

    # History summary (used by intent phase)
    history_summary_max_rounds: int = 3
    history_summary_max_tokens: int = 800

    # Session cache
    max_session_contexts: int = 100

    # Context module config
    context: ContextConfig = field(default_factory=ContextConfig)

    def __post_init__(self) -> None:
        if self.max_react_per_step < 1:
            raise ValueError(f"max_react_per_step must be >= 1, got {self.max_react_per_step}")
        if self.max_consecutive_exec_failures < 1:
            raise ValueError(f"max_consecutive_exec_failures must be >= 1, got {self.max_consecutive_exec_failures}")
        if self.max_replans < 0:
            raise ValueError(f"max_replans must be >= 0, got {self.max_replans}")
        if self.max_duplicate_tool_calls < 1:
            raise ValueError(f"max_duplicate_tool_calls must be >= 1, got {self.max_duplicate_tool_calls}")
        if self.max_session_contexts < 1:
            raise ValueError(f"max_session_contexts must be >= 1, got {self.max_session_contexts}")
