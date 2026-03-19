"""AG-UI event types, builders, accumulators, and sinks."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable
from uuid import uuid4


# ═══════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════


class AgentFinishReason(str, Enum):
    """Reason for a run finishing."""

    FINAL_ANSWER = "final_answer_generated"
    MAX_ITERATIONS = "max_iterations_reached"
    EXEC_FAILURES_EXCEEDED = "execute_skill_failed_too_many"
    ERROR_POLICY_ABORT = "execute_skill_abort"
    ERROR_POLICY_PROMPT = "execute_skill_prompt_user"


class AGUIEventType:
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
    RUN_FINISHED = "RUN_FINISHED"
    RUN_ERROR = "RUN_ERROR"


# ═══════════════════════════════════════════════════════════════════
# Event helpers
# ═══════════════════════════════════════════════════════════════════


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_event(
    event_type: str, run_id: str, thread_id: str, **payload: Any,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": event_type,
        "runId": run_id,
        "threadId": thread_id,
        "timestamp": utc_now_iso(),
    }
    event.update(payload)
    return event


def new_run_id() -> str:
    return str(uuid4())


# ═══════════════════════════════════════════════════════════════════
# Accumulator & Sinks
# ═══════════════════════════════════════════════════════════════════


@dataclass
class RunAccumulator:
    """Aggregate AG-UI stream into persistable run result."""

    run_id: str
    thread_id: str
    status: str = "running"
    final_text: str = ""
    usage: dict[str, Any] | None = None
    current_message_id: str | None = None
    _buffer: list[str] = field(default_factory=list)

    def consume(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")

        if event_type == AGUIEventType.TEXT_MESSAGE_START:
            self.current_message_id = event.get("messageId")
            self._buffer = []
        elif event_type == AGUIEventType.TEXT_MESSAGE_CONTENT:
            delta = event.get("delta", "")
            if delta:
                self._buffer.append(str(delta))
        elif event_type == AGUIEventType.TEXT_MESSAGE_END:
            self.final_text = "".join(self._buffer)
            self.current_message_id = None
        elif event_type == AGUIEventType.RUN_FINISHED:
            self.status = "finished"
            output_text = event.get("outputText")
            if isinstance(output_text, str) and output_text:
                self.final_text = output_text
            if event.get("usage"):
                self.usage = event["usage"]
        elif event_type == AGUIEventType.RUN_ERROR:
            self.status = "error"


class AGUIEventSink:
    """Base sink for AG-UI event fan-out."""

    async def handle(self, event: dict[str, Any]) -> None:
        return


class AGUIEventPipeline:
    """Dispatch AG-UI events to multiple sinks."""

    def __init__(self) -> None:
        self._sinks: list[AGUIEventSink] = []

    def add_sink(self, sink: AGUIEventSink) -> None:
        self._sinks.append(sink)

    async def emit(self, event: dict[str, Any]) -> None:
        for sink in self._sinks:
            await sink.handle(event)


class PersistenceSink(AGUIEventSink):
    """Persist assistant output at RUN_FINISHED boundary."""

    def __init__(
        self,
        callback: Callable[..., Any | Awaitable[Any]] | None = None,
    ) -> None:
        self._callback = callback
        self._accumulators: dict[str, RunAccumulator] = {}

    async def handle(self, event: dict[str, Any]) -> None:
        run_id = event.get("runId")
        if not run_id:
            return

        acc = self._accumulators.get(run_id)
        if acc is None:
            acc = RunAccumulator(
                run_id=run_id,
                thread_id=event.get("threadId", ""),
            )
            self._accumulators[run_id] = acc

        acc.consume(event)

        event_type = event.get("type")
        if event_type == AGUIEventType.RUN_FINISHED:
            await self._invoke_callback(acc)
            self._accumulators.pop(run_id, None)
        elif event_type == AGUIEventType.RUN_ERROR:
            self._accumulators.pop(run_id, None)

    async def _invoke_callback(self, acc: RunAccumulator) -> None:
        if not self._callback or not acc.final_text:
            return
        try:
            sig = inspect.signature(self._callback)
            param_count = len(sig.parameters)
        except (TypeError, ValueError):
            param_count = 2

        if param_count <= 1:
            result = self._callback(acc.final_text)
        else:
            result = self._callback(acc.final_text, acc.usage)

        if inspect.isawaitable(result):
            await result
