"""AG-UI wire-format event types and builders.

Depends only on .types (leaf enums).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4


class AGUIEventType(StrEnum):
    """AG-UI protocol wire event types."""

    RUN_STARTED = "RUN_STARTED"
    STEP_STARTED = "STEP_STARTED"
    STEP_FINISHED = "STEP_FINISHED"
    TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
    TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
    TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
    TOOL_CALL_START = "TOOL_CALL_START"
    TOOL_CALL_RESULT = "TOOL_CALL_RESULT"
    INTENT_RECOGNIZED = "INTENT_RECOGNIZED"
    PLAN_GENERATED = "PLAN_GENERATED"
    REFLECTION_RESULT = "REFLECTION_RESULT"
    USER_INPUT_REQUESTED = "USER_INPUT_REQUESTED"
    AWAITING_USER_INPUT = "AWAITING_USER_INPUT"
    RUN_FINISHED = "RUN_FINISHED"
    RUN_ERROR = "RUN_ERROR"


def utc_now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    """Generate a unique run / message identifier."""
    return str(uuid4())


def build_event(
    event_type: str,
    run_id: str,
    thread_id: str,
    **payload: Any,
) -> dict[str, Any]:
    """Construct a single AG-UI event dict."""
    event: dict[str, Any] = {
        "type": event_type,
        "runId": run_id,
        "threadId": thread_id,
        "timestamp": utc_now_iso(),
    }
    event.update(payload)
    return event
