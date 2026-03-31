"""Protocol-layer enumerations — zero dependencies, leaf of the import graph."""

from __future__ import annotations

from enum import StrEnum


class PhaseSignalType(StrEnum):
    """Semantic signal types that agent phases can emit."""

    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    RUN_ERROR = "run_error"
    INTENT_RECOGNIZED = "intent_recognized"
    PLAN_GENERATED = "plan_generated"
    STEP_STARTED = "step_started"
    STEP_FINISHED = "step_finished"
    TEXT_MESSAGE_START = "text_message_start"
    TEXT_MESSAGE_CONTENT = "text_message_content"
    TEXT_MESSAGE_END = "text_message_end"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_RESULT = "tool_call_result"
    REFLECTION_RESULT = "reflection_result"
    USER_INPUT_REQUESTED = "user_input_requested"
    AWAITING_USER_INPUT = "awaiting_user_input"


class StepStatus(StrEnum):
    """Status reported when a step finishes."""

    DONE = "done"
    CONTINUE = "continue"
    FINALIZE = "finalize"


class PlanStepStatus(StrEnum):
    """Lifecycle status of a single plan step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"


class RunStatus(StrEnum):
    """Lifecycle status of a run."""

    RUNNING = "running"
    FINISHED = "finished"
    ERROR = "error"


class AgentFinishReason(StrEnum):
    """Reason for a run finishing."""

    FINAL_ANSWER = "final_answer_generated"
    MAX_ITERATIONS = "max_iterations_reached"
    EXEC_FAILURES_EXCEEDED = "execute_skill_failed_too_many"
    ERROR_POLICY_ABORT = "execute_skill_abort"
    ERROR_POLICY_PROMPT = "execute_skill_prompt_user"
    ERROR = "error"
