"""Protocol adapter — translates semantic PhaseSignalType to wire-format events.

Depends on .types and .events.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .events import AGUIEventType, build_event
from .types import PhaseSignalType


class ProtocolAdapter(ABC):
    """Abstract protocol adapter.

    Subclasses map ``PhaseSignalType`` semantic signals to a concrete
    wire-format (e.g. AG-UI JSON events).
    """

    @abstractmethod
    def translate(
        self,
        signal_type: PhaseSignalType,
        run_id: str,
        thread_id: str,
        **payload: Any,
    ) -> dict[str, Any]: ...


class AGUIProtocolAdapter(ProtocolAdapter):
    """AG-UI protocol implementation."""

    _SIGNAL_MAP: dict[PhaseSignalType, AGUIEventType] = {
        PhaseSignalType.RUN_STARTED: AGUIEventType.RUN_STARTED,
        PhaseSignalType.RUN_FINISHED: AGUIEventType.RUN_FINISHED,
        PhaseSignalType.RUN_ERROR: AGUIEventType.RUN_ERROR,
        PhaseSignalType.INTENT_RECOGNIZED: AGUIEventType.INTENT_RECOGNIZED,
        PhaseSignalType.PLAN_GENERATED: AGUIEventType.PLAN_GENERATED,
        PhaseSignalType.STEP_STARTED: AGUIEventType.STEP_STARTED,
        PhaseSignalType.STEP_FINISHED: AGUIEventType.STEP_FINISHED,
        PhaseSignalType.TEXT_MESSAGE_START: AGUIEventType.TEXT_MESSAGE_START,
        PhaseSignalType.TEXT_MESSAGE_CONTENT: AGUIEventType.TEXT_MESSAGE_CONTENT,
        PhaseSignalType.TEXT_MESSAGE_END: AGUIEventType.TEXT_MESSAGE_END,
        PhaseSignalType.TOOL_CALL_START: AGUIEventType.TOOL_CALL_START,
        PhaseSignalType.TOOL_CALL_RESULT: AGUIEventType.TOOL_CALL_RESULT,
        PhaseSignalType.REFLECTION_RESULT: AGUIEventType.REFLECTION_RESULT,
        PhaseSignalType.USER_INPUT_REQUESTED: AGUIEventType.USER_INPUT_REQUESTED,
        PhaseSignalType.AWAITING_USER_INPUT: AGUIEventType.AWAITING_USER_INPUT,
    }

    def translate(
        self,
        signal_type: PhaseSignalType,
        run_id: str,
        thread_id: str,
        **payload: Any,
    ) -> dict[str, Any]:
        event_type = self._SIGNAL_MAP[signal_type]
        return build_event(event_type, run_id, thread_id, **payload)
