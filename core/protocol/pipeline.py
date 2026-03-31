"""AG-UI event pipeline — accumulators, sinks, and fan-out.

Depends on .events and .types.
"""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from .events import AGUIEventType
from .types import RunStatus

logger = logging.getLogger(__name__)


@runtime_checkable
class ToolTranscriptPersister(Protocol):
    """Callback protocol for persisting tool call transcripts."""

    def __call__(
        self,
        role: str,
        title: str,
        content: str,
        tool_call_id: str | None,
        tool_calls: list[dict] | None,
    ) -> Any | Awaitable[Any]: ...


@dataclass
class RunAccumulator:
    """Aggregate AG-UI stream into persistable run result."""

    run_id: str
    thread_id: str
    status: RunStatus = RunStatus.RUNNING
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
            chunk = "".join(self._buffer)
            if chunk:
                self.final_text = (
                    f"{self.final_text}\n{chunk}".strip() if self.final_text else chunk
                )
            self.current_message_id = None
        elif event_type == AGUIEventType.RUN_FINISHED:
            self.status = RunStatus.FINISHED
            output_text = event.get("outputText")
            if isinstance(output_text, str) and output_text:
                self.final_text = output_text
            if event.get("usage"):
                self.usage = event["usage"]
        elif event_type == AGUIEventType.RUN_ERROR:
            self.status = RunStatus.ERROR


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


class ToolTranscriptSink(AGUIEventSink):
    """Persist tool call transcripts (assistant tool_calls + tool results) to DB."""

    def __init__(self, persister: ToolTranscriptPersister) -> None:
        self._persister = persister
        self._pending_calls: dict[str, dict[str, Any]] = {}

    async def handle(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")

        if event_type in (AGUIEventType.RUN_FINISHED, AGUIEventType.RUN_ERROR):
            self._pending_calls.clear()
            return

        if event_type == AGUIEventType.TOOL_CALL_START:
            call_id = event.get("toolCallId", "")
            tool_name = event.get("toolName", "")
            arguments = event.get("arguments", {})
            self._pending_calls[call_id] = {
                "name": tool_name,
                "arguments": arguments,
            }

        elif event_type == AGUIEventType.TOOL_CALL_RESULT:
            call_id = event.get("toolCallId", "")
            tool_name = event.get("toolName", "")
            result_content = event.get("result", "")

            pending = self._pending_calls.pop(call_id, None)
            tool_calls_payload = None
            if pending:
                raw_args = pending["arguments"]
                args_str = (
                    json.dumps(raw_args, ensure_ascii=False)
                    if isinstance(raw_args, dict)
                    else str(raw_args)
                )
                tool_calls_payload = [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": pending["name"],
                            "arguments": args_str,
                        },
                    }
                ]

            await self._persist(
                role="assistant",
                title=f"tool_call: {tool_name}",
                content="",
                tool_call_id=None,
                tool_calls=tool_calls_payload,
            )
            await self._persist(
                role="tool",
                title=f"result: {tool_name}",
                content=result_content or "",
                tool_call_id=call_id,
                tool_calls=None,
            )

    async def _persist(
        self,
        role: str,
        title: str,
        content: str,
        tool_call_id: str | None,
        tool_calls: list[dict] | None,
    ) -> None:
        try:
            result = self._persister(role, title, content, tool_call_id, tool_calls)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.warning(
                "ToolTranscriptSink._persist failed (role=%s, title=%s)",
                role,
                title,
                exc_info=True,
            )
