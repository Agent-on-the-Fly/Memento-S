# SPDX-License-Identifier: Apache-2.0
"""core.context — unified context management API.

Public entry points:
  - ContextManager: the single Agent-facing context interface
  - RuntimeState / RuntimeStateStore: per-session persistence
  - sync_from_agent_run: AgentRunState → RuntimeState synchroniser
  - SessionStatus / SessionGoal / ActionRecord: session data schemas
  - PreApiPipeline: pre-API context processing pipeline

Submodules:
  - history: HistoryManager (load & slim history)
  - session: SessionGoal, SessionStatus, ActionRecord (migrated from core/session)
    (imported directly from submodules to avoid circular deps)
"""

from .context_manager import ContextManager
from .runtime import RuntimeState, RuntimeStateStore, sync_from_agent_run
from .session import SessionStatus, SessionGoal, ActionRecord, build_session_context_block
from .session_context import SessionContext
from .pre_api_pipeline import PreApiPipeline
from .scratchpad import Scratchpad

__all__ = [
    "ContextManager",
    "RuntimeState",
    "RuntimeStateStore",
    "sync_from_agent_run",
    "SessionStatus",
    "SessionGoal",
    "ActionRecord",
    "build_session_context_block",
    "SessionContext",
    "PreApiPipeline",
    "Scratchpad",
]
