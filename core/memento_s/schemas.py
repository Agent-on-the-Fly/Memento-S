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
    max_react_per_step: int = 3
    max_replans: int = 2
    max_consecutive_exec_failures: int = 10

    # Reflection limits
    reflection_input_chars: int = 15000
    reflection_max_tokens: int = 30000

    # History summary (used by intent phase)
    history_summary_max_rounds: int = 3
    history_summary_max_tokens: int = 800

    # Context module config
    context: ContextConfig = field(default_factory=ContextConfig)
