"""Phase: Intent recognition — classify user intent as the *Comprehender* role.

Responsibilities:
  - Understand what the user is saying
  - Classify request type (DIRECT / AGENTIC / CONFIRM / INTERRUPT)
  - Detect context shifts
  - Surface ambiguity

Does NOT: match skills, extract parameters, decide implementation details.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from core.manager.session_context import RECENT_ACTIONS_INTENT
from core.prompts.templates import INTENT_PROMPT
from middleware.llm import LLMClient
from utils.debug_logger import log_agent_phase
from utils.logger import get_logger

from ..schemas import AgentConfig
from ..utils import extract_json

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════


class IntentMode(str, Enum):
    """Four-way intent classification."""

    DIRECT = "direct"
    AGENTIC = "agentic"
    CONFIRM = "confirm"
    INTERRUPT = "interrupt"


class IntentResult(BaseModel):
    """Output of the intent phase."""

    mode: IntentMode = Field(description="direct / agentic / confirm / interrupt")
    task: str = Field(description="User's task in their original language")
    task_summary: str = Field(default="", description="Short English summary for internal logging")
    intent_shifted: bool = Field(default=False)
    ambiguity: str = Field(default="", description="Ambiguity description when mode=confirm")
    clarification_question: str = Field(default="", description="Question to ask user when mode=confirm")


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _build_session_context_block(session_context: Any, user_content: str) -> str:
    """Build a concise session-context block for the intent prompt."""
    if session_context is None:
        return "- No active session context"

    lines: list[str] = []

    goal = getattr(session_context, "session_goal", "")
    if goal and goal.strip() != user_content.strip():
        lines.append(f"- Current session goal: {goal[:150]}")

    action_history = getattr(session_context, "action_history", [])
    if action_history:
        recent = action_history[-RECENT_ACTIONS_INTENT:]
        summaries: list[str] = []
        for a in recent:
            name = getattr(a, "skill_name", "") or getattr(a, "tool_name", "unknown")
            ok = "OK" if getattr(a, "success", False) else "FAIL"
            res = getattr(a, "result_summary", "")[:60]
            summaries.append(f"{name}({ok}): {res}")
        lines.append(
            f"- Actions so far: {len(action_history)} total. "
            f"Recent: {'; '.join(summaries)}"
        )

    has_plan = getattr(session_context, "has_active_plan", False)
    plan_count = getattr(session_context, "plan_step_count", 0)
    if plan_count:
        # Try to get done count from statuses
        statuses = getattr(session_context, "_plan_statuses", [])
        done = sum(1 for s in statuses if str(s) == "done")
        lines.append(f"- Active task plan: {done}/{plan_count} steps completed")
    lines.append(f"- Multi-step task running: {'YES' if has_plan else 'no'}")

    return "\n".join(lines) if lines else "- No active session context"


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════


async def recognize_intent(
    user_content: str,
    history: list[dict[str, Any]] | None,
    llm: LLMClient,
    context_manager: Any,
    session_context: Any = None,
    config: AgentConfig | None = None,
) -> IntentResult:
    """Recognise user intent. Preserves the user's original language in ``task``.

    Returns an ``IntentResult`` with ``mode``, ``task``, ``intent_shifted``,
    and optionally ``ambiguity`` / ``clarification_question`` for CONFIRM mode.
    """
    cfg = config or AgentConfig()
    history_summary = context_manager.build_history_summary(
        history,
        max_rounds=cfg.history_summary_max_rounds,
        max_tokens=cfg.history_summary_max_tokens,
    )
    session_ctx_block = _build_session_context_block(session_context, user_content)

    prompt = INTENT_PROMPT.format(
        user_message=user_content,
        history_summary=history_summary,
        session_context=session_ctx_block,
    )

    session_id = getattr(session_context, "session_id", "unknown")

    try:
        log_agent_phase("INTENT_LLM_CALL", session_id, f"prompt_len={len(prompt)}")
        resp = await llm.async_chat(messages=[{"role": "user", "content": prompt}])
        raw = (resp.content or "").strip()
        data = extract_json(raw)

        mode_str = data.get("mode", "agentic")
        try:
            data["mode"] = IntentMode(mode_str)
        except ValueError:
            data["mode"] = IntentMode.AGENTIC

        result = IntentResult(**data)
        log_agent_phase(
            "INTENT_RESULT", session_id,
            f"mode={result.mode.value}, task={result.task[:60]}",
        )
        return result

    except Exception as e:
        logger.warning("Intent recognition failed, defaulting to agentic: {}", e)
        return IntentResult(
            mode=IntentMode.AGENTIC,
            task=user_content,
            intent_shifted=False,
        )
