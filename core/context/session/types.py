# SPDX-License-Identifier: Apache-2.0
"""core.context.session.types — pure data schemas, zero behavior.

This module contains dataclasses that represent pure state without methods.
All behavior (updates, rendering) lives in higher-level modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


RECENT_ACTIONS_DISPLAY = 5
"""How many recent ActionRecord entries to show in prompt sections."""

RECENT_ACTIONS_INTENT = 3
"""How many recent actions to include in intent recognition context."""

GOAL_MAX_LENGTH = 200
"""Maximum character length for the session goal text."""


@dataclass
class ActionRecord:
    """Record of a single tool/skill execution — pure data container.

    Factory: use ``ActionRecord.from_tool_call()`` for structured creation.
    """

    tool_name: str
    skill_name: str = ""
    args_summary: str = ""
    result_summary: str = ""
    success: bool = True
    timestamp: float = 0.0

    @classmethod
    def from_tool_call(
        cls,
        tool_name: str,
        args: dict[str, Any],
        result: str,
        success: bool = True,
    ) -> ActionRecord:
        """Construct from raw tool call data."""
        skill_name = ""
        if tool_name == "execute_skill":
            skill_name = args.get("skill_name", "")

        args_str = str(args)
        if len(args_str) > 100:
            args_str = args_str[:97] + "..."

        result_str = str(result)
        if len(result_str) > 200:
            result_str = result_str[:197] + "..."

        return cls(
            tool_name=tool_name,
            skill_name=skill_name,
            args_summary=args_str,
            result_summary=result_str,
            success=success,
            timestamp=datetime.now().timestamp(),
        )


@dataclass
class SessionGoal:
    """Pure goal state — extracted from per-session user input.

    Lifecycle:
      - Created empty
      - ``refine(user_msg)`` called on first turn to capture goal
      - ``text`` remains read-only afterwards
    """

    text: str = ""
    turn_count: int = 0
    created_at: str = ""

    def refine(self, user_msg: str) -> None:
        """Progressive goal refinement — called on first turn only.

        Subsequent calls increment ``turn_count`` but leave ``text`` unchanged.
        """
        self.turn_count += 1
        if self.turn_count == 1:
            goal = user_msg.strip()
            if len(goal) > GOAL_MAX_LENGTH:
                goal = goal[: GOAL_MAX_LENGTH - 3] + "..."
            self.text = goal
            self.created_at = datetime.now().isoformat(timespec="seconds")
