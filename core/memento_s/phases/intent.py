# SPDX-License-Identifier: Apache-2.0
"""Phase: Intent recognition — classify user intent as the *Comprehender* role.

Responsibilities:
  - Understand what the user is saying
  - Classify request type (DIRECT / AGENTIC / CONFIRM / INTERRUPT)
  - Detect context shifts
  - Surface ambiguity

Does NOT: match skills, extract parameters, decide implementation details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.context import ContextManager
    from core.context.session.types import SessionGoal

from pydantic import BaseModel, Field

from core.protocol.types import IntentMode
from core.context.session import build_session_context_block
from core.prompts.templates import INTENT_PROMPT
from middleware.llm import LLMClient
from utils.debug_logger import log_agent_phase
from utils.logger import get_logger

from ..schemas import AgentRuntimeConfig
from ..utils import extract_json

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════


class IntentResult(BaseModel):
    """Output of the intent phase."""

    mode: IntentMode = Field(description="direct / agentic / confirm / interrupt")
    task: str = Field(description="User's task in their original language")
    task_summary: str = Field(default="", description="Short English summary for internal logging")
    intent_shifted: bool = Field(default=False)
    ambiguity: str | None = Field(default=None, description="Ambiguity description when mode=confirm")
    clarification_question: str | None = Field(default=None, description="Question to ask user when mode=confirm")


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════


async def recognize_intent(
    user_content: str,
    history: list[dict[str, Any]] | None,
    llm: LLMClient,
    context_manager: ContextManager,
    session_context: SessionGoal | None = None,
    config: AgentRuntimeConfig | None = None,
) -> IntentResult:
    """Recognise user intent. Preserves the user's original language in ``task``.

    Returns an ``IntentResult`` with ``mode``, ``task``, ``intent_shifted``,
    and optionally ``ambiguity`` / ``clarification_question`` for CONFIRM mode.
    """
    cfg = config or AgentRuntimeConfig()
    history_summary = context_manager.build_history_summary(
        history,
        max_rounds=cfg.history_summary_max_rounds,
        max_tokens=cfg.history_summary_max_tokens,
    )
    session_ctx_block = build_session_context_block(session_context, user_content)

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
