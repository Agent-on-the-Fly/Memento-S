"""Phase: Intent recognition — classify user intent and normalize to English task.

Owns IntentMode and IntentResult (co-located to avoid circular imports).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

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
    """Three-way intent classification."""

    DIRECT = "direct"
    AGENTIC = "agentic"
    INTERRUPT = "interrupt"


class IntentResult(BaseModel):
    """Output of the intent phase."""

    mode: IntentMode = Field(description="direct / agentic / interrupt")
    task: str = Field(description="Normalized complete English task description")
    intent_shifted: bool = Field(default=False)


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
        recent = action_history[-3:]
        success_count = sum(1 for a in recent if getattr(a, "success", False))
        lines.append(
            f"- Actions so far: {len(action_history)} total, "
            f"last {len(recent)} steps had {success_count} successes"
        )

    has_plan = getattr(session_context, "has_active_plan", False)
    plan_count = getattr(session_context, "plan_step_count", 0)
    if plan_count:
        lines.append(f"- Active task plan: {plan_count} steps defined")
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
    """Recognise user intent and normalise to an English task description.

    Returns an ``IntentResult`` with ``mode``, ``task`` and ``intent_shifted``.
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
