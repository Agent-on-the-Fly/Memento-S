"""MementoSAgent adapter — wraps the ReAct agent for the evolve loop."""

from __future__ import annotations

import uuid
from typing import Any

from .text_utils import dedupe_keep_order
from .tracker import TaskRunResult


class _StreamState:
    """Mutable accumulator for agent stream parsing."""

    __slots__ = ("used_skills", "trace", "final_text", "step_num", "status")

    def __init__(self, *, step_num: int = 0, status: str = "done") -> None:
        self.used_skills: list[str] = []
        self.trace: list[dict[str, Any]] = []
        self.final_text: str = ""
        self.step_num: int = step_num
        self.status: str = status


async def _consume_agent_stream(
    agent: Any,
    session_id: str,
    user_content: str,
    state: _StreamState,
) -> None:
    """Parse MementoSAgent.reply_stream() events into *state*.

    Step budget is controlled by AGENT_MAX_ITERATIONS inside MementoSAgent.

    Event types yielded by reply_stream():
      - {"type": "text_delta", "content": "..."}
      - {"type": "skill_call_start", "skill_name": "...", "call_id": "...", "arguments": {...}}
      - {"type": "skill_call_result", "skill_name": "...", "call_id": "...", "result": "..."}
      - {"type": "status", "message": "..."}
      - {"type": "final", "content": "..."}
      - {"type": "error", "message": "..."}
    """
    text_parts: list[str] = []

    async for event in agent.reply_stream(session_id, user_content):
        if not isinstance(event, dict):
            continue

        event_type = event.get("type", "")

        if event_type == "skill_call_start":
            state.step_num += 1
            skill_name = str(event.get("skill_name") or "").strip()
            if skill_name and skill_name not in state.used_skills:
                state.used_skills.append(skill_name)
            # When agent calls read_skill, the actual domain skill is in arguments.skill_name
            arguments = event.get("arguments")
            if skill_name == "read_skill" and isinstance(arguments, dict):
                domain_skill = str(arguments.get("skill_name") or "").strip()
                if domain_skill and domain_skill not in state.used_skills:
                    state.used_skills.append(domain_skill)
            state.trace.append({
                "step_num": state.step_num,
                "status": "running",
                "skill_name": skill_name,
                "tool": skill_name,
                "result_preview": str(arguments or "")[:300],
            })

        elif event_type == "skill_call_result":
            result_text = str(event.get("result") or "")
            skill_name = str(event.get("skill_name") or "").strip()
            # Update the last trace entry for this skill with result preview
            for entry in reversed(state.trace):
                if entry.get("skill_name") == skill_name and entry.get("status") == "running":
                    entry["status"] = "done"
                    entry["result_preview"] = result_text[:300]
                    break

        elif event_type == "text_delta":
            content = event.get("content") or ""
            if content:
                text_parts.append(content)

        elif event_type == "final":
            final = str(event.get("content") or "").strip()
            if final:
                state.final_text = final

        elif event_type == "error":
            state.status = "error"
            err_msg = str(event.get("message") or "unknown error")
            state.trace.append({
                "step_num": state.step_num + 1,
                "status": "error",
                "skill_name": "",
                "result_preview": err_msg[:300],
            })

        elif event_type == "status":
            pass  # informational

    # If no final event was received, assemble from text deltas
    if not state.final_text and text_parts:
        state.final_text = "".join(text_parts).strip()


async def execute_task_once(
    agent: Any,  # MementoSAgent
    *,
    user_text: str,
    create_on_miss_enabled: bool = False,
    skill_manager: Any = None,  # SkillManager (optional, for skill context injection)
) -> TaskRunResult:
    """Run one task through MementoSAgent and extract structured results.

    Step budget is controlled by ``AGENT_MAX_ITERATIONS`` in ``.env``.

    1. Optionally injects ``[Matched Skills]`` context into the user message (via *skill_manager*).
    2. Calls ``agent.reply_stream(session_id, user_content)`` and parses the event stream.
    3. Extracts which skills were called, tool trace, and the final answer.
    4. If no skills matched and *create_on_miss_enabled*, instructs agent to use ``skill-creator``.

    Returns a :class:`TaskRunResult` with the answer, used skills, status, and trace.
    """
    # Generate a unique session_id for this task attempt
    session_id = f"evolve_{uuid.uuid4().hex[:12]}"

    # --- Stream the agent ---
    # Skill routing is now handled by the agent via the `route_skill` tool
    # at any step, rather than one-shot prompt injection here.
    state = _StreamState()

    try:
        await _consume_agent_stream(agent, session_id, user_text, state)
    except Exception as exc:
        state.status = "error"
        state.trace.append({
            "step_num": state.step_num + 1,
            "status": "error",
            "skill_name": "",
            "result_preview": f"{type(exc).__name__}: {exc}",
        })
        if not state.final_text:
            state.final_text = f"ERROR: {type(exc).__name__}: {exc}"

    # --- 3. Handle no-match + create_on_miss ---
    if not state.used_skills and create_on_miss_enabled and state.status == "done":
        create_prompt = (
            f"{user_text}\n\n"
            "No existing skill matched this task. Please:\n"
            "1. Use `read_skill` with `skill-creator` to learn how to create a new skill.\n"
            "2. Create a suitable reusable skill for this type of task.\n"
            "3. Then use the newly created skill to solve the task.\n\n"
            "Return the final answer in the required format."
        )
        create_session_id = f"evolve_create_{uuid.uuid4().hex[:12]}"
        state.trace.append({
            "step_num": state.step_num + 1,
            "status": "create_on_miss_attempt",
            "skill_name": "skill-creator",
            "result_preview": "Instructing agent to create a new skill",
        })
        try:
            await _consume_agent_stream(agent, create_session_id, create_prompt, state)
        except Exception as exc:
            state.trace.append({
                "step_num": state.step_num + 1,
                "status": "create_on_miss_error",
                "skill_name": "",
                "result_preview": f"{type(exc).__name__}: {exc}",
            })

    # --- 4. Format final answer ---
    if not state.final_text:
        state.final_text = "(no response)"
        state.status = "error" if state.status == "done" else state.status

    answer = state.final_text

    return TaskRunResult(
        answer=answer,
        used_skills=dedupe_keep_order(state.used_skills),
        status=state.status,
        trace=state.trace,
    )
