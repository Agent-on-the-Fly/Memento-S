"""RunEmitter — protocol-agnostic event emitter for a single agent run.

Agent and phase code calls typed methods here; the adapter translates
each call into the concrete wire-format event dict.

Depends on .types and .adapter.
"""

from __future__ import annotations

from typing import Any

from .adapter import ProtocolAdapter
from .events import new_run_id
from .types import AgentFinishReason, PhaseSignalType, StepStatus


class RunEmitter:
    """Protocol-agnostic event emitter bound to a single run.

    Every public method returns a ``dict[str, Any]`` event that can be
    yielded directly from an async generator.
    """

    def __init__(
        self,
        run_id: str,
        thread_id: str,
        adapter: ProtocolAdapter,
    ) -> None:
        self._run_id = run_id
        self._thread_id = thread_id
        self._adapter = adapter

    def _emit(self, signal: PhaseSignalType, **payload: Any) -> dict[str, Any]:
        return self._adapter.translate(
            signal, self._run_id, self._thread_id, **payload
        )

    # ── Run lifecycle ───────────────────────────────────────────────

    def run_started(self, *, input_text: str) -> dict[str, Any]:
        return self._emit(PhaseSignalType.RUN_STARTED, inputText=input_text)

    def run_finished(
        self,
        *,
        output_text: str,
        reason: AgentFinishReason,
        usage: dict[str, Any] | None = None,
        context_tokens: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "outputText": output_text,
            "reason": reason.value,
            "usage": usage,
        }
        if context_tokens is not None:
            payload["contextTokens"] = context_tokens
        return self._emit(PhaseSignalType.RUN_FINISHED, **payload)

    def run_error(self, *, message: str) -> dict[str, Any]:
        return self._emit(PhaseSignalType.RUN_ERROR, message=message)

    # ── Intent ──────────────────────────────────────────────────────

    def intent_recognized(self, *, mode: str, task: str) -> dict[str, Any]:
        return self._emit(
            PhaseSignalType.INTENT_RECOGNIZED, mode=mode, task=task
        )

    # ── Plan ────────────────────────────────────────────────────────

    def plan_generated(
        self,
        *,
        goal: str,
        steps: list[dict[str, Any]],
        replan: bool = False,
        **extra: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"goal": goal, "steps": steps}
        if replan:
            payload["replan"] = True
        payload.update(extra)
        return self._emit(PhaseSignalType.PLAN_GENERATED, **payload)

    # ── Step lifecycle ──────────────────────────────────────────────

    def step_started(self, *, step: int, name: str) -> dict[str, Any]:
        return self._emit(
            PhaseSignalType.STEP_STARTED, step=step, name=name
        )

    def step_finished(
        self, *, step: int, status: StepStatus
    ) -> dict[str, Any]:
        return self._emit(
            PhaseSignalType.STEP_FINISHED, step=step, status=status.value
        )

    # ── Text message trio ───────────────────────────────────────────

    def text_message_start(
        self, *, message_id: str, role: str = "assistant"
    ) -> dict[str, Any]:
        return self._emit(
            PhaseSignalType.TEXT_MESSAGE_START,
            messageId=message_id,
            role=role,
        )

    def text_delta(
        self, *, message_id: str, delta: str
    ) -> dict[str, Any]:
        return self._emit(
            PhaseSignalType.TEXT_MESSAGE_CONTENT,
            messageId=message_id,
            delta=delta,
        )

    def text_message_end(self, *, message_id: str) -> dict[str, Any]:
        return self._emit(
            PhaseSignalType.TEXT_MESSAGE_END, messageId=message_id
        )

    # ── Tool calls ──────────────────────────────────────────────────

    def tool_call_start(
        self,
        *,
        step: int,
        call_id: str,
        name: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        return self._emit(
            PhaseSignalType.TOOL_CALL_START,
            step=step,
            toolCallId=call_id,
            toolName=name,
            arguments=args,
        )

    def tool_call_result(
        self,
        *,
        step: int,
        call_id: str,
        name: str,
        result: str,
    ) -> dict[str, Any]:
        return self._emit(
            PhaseSignalType.TOOL_CALL_RESULT,
            step=step,
            toolCallId=call_id,
            toolName=name,
            result=result,
        )

    # ── Reflection ──────────────────────────────────────────────────

    def reflection_result(
        self,
        *,
        decision: str,
        reason: str,
        completed_step_id: int | None = None,
        next_step_hint: str | None = None,
    ) -> dict[str, Any]:
        return self._emit(
            PhaseSignalType.REFLECTION_RESULT,
            decision=decision,
            reason=reason,
            completedStepId=completed_step_id,
            nextStepHint=next_step_hint,
        )

    # ── HITL ───────────────────────────────────────────────────────

    def user_input_requested(self, *, question: str) -> dict[str, Any]:
        return self._emit(
            PhaseSignalType.USER_INPUT_REQUESTED,
            question=question,
        )

    # ── Helpers ─────────────────────────────────────────────────────

    def new_message_id(self) -> str:
        return new_run_id()
