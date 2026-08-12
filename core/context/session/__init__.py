# SPDX-License-Identifier: Apache-2.0
"""core.context.session — session-level runtime state (pure data schemas).

Migrated from core/session/ — contains only pure data schemas and stateless helpers.

Exports:
  - ActionRecord: tool/skill execution record (schema only)
  - SessionGoal: per-session goal state (schema only)
  - SessionStatus: lifecycle status enum (StrEnum)
  - RECENT_ACTIONS_DISPLAY / RECENT_ACTIONS_INTENT: constants
  - build_session_context_block: prompt-building helper
  - ActionHistory: encapsulated action history (zero global state pollution)
"""

from .types import (
    ActionRecord,
    RECENT_ACTIONS_DISPLAY,
    RECENT_ACTIONS_INTENT,
    SessionGoal,
)
from .enums import SessionStatus
from .builders import build_session_context_block

__all__ = [
    "ActionRecord",
    "SessionGoal",
    "SessionStatus",
    "RECENT_ACTIONS_DISPLAY",
    "RECENT_ACTIONS_INTENT",
    "build_session_context_block",
]
