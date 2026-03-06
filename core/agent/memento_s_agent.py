
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, AsyncGenerator, Final

from core.config import g_settings
from core.config.logging import get_logger
from core.llm import LLM
from core.llm.schema import LLMResponse
from core.skills import SkillManager
from core.skills.provider import DeltaSkillsProvider
from core.tools import (
    BUILTIN_TOOL_SCHEMAS,
    configure_builtin_tools,
    is_builtin_tool,
    execute_builtin_tool,
)
from core.skills.schema import SkillCall

from .session_manager import SessionManager
from .stateful_context_manager import StatefulContextManager
from .utils import skill_call_to_openai_payload
from ..skills.provider.delta_skills.bootstrap import create_app_context

logger = get_logger(__name__)

MEMENTO_S_FINAL_TAG: Final[str] = "memento_s_final"

_END_MARKER_PATTERNS: Final[list[re.Pattern]] = [
    re.compile(
        rf"```[\w-]*\s*<\s*{MEMENTO_S_FINAL_TAG}\s*>(.*?)<\s*/\s*{MEMENTO_S_FINAL_TAG}\s*>\s*```",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        rf"<\s*{MEMENTO_S_FINAL_TAG}\s*>(.*?)<\s*/\s*{MEMENTO_S_FINAL_TAG}\s*>",
        re.IGNORECASE | re.DOTALL,
    ),
]



_PATH_RE: re.Pattern = re.compile(
    r"(?:^|[\s\"'`=({\[])(/[^\s\"'`<>|;\\\n]{3,})",
    re.MULTILINE,
)

_LAYER1_SELF_CHECK: str = (
    "[System] SELF-CHECK: You sent a final answer without calling any tools.\n\n"
    "• If this task required a real-world action (creating/writing files, running commands, "
    "downloading, querying APIs, etc.) → you have NOT completed it. "
    "Call the appropriate skill NOW.\n"
    "• If this is genuinely a conversational reply (greeting, factual Q&A, explanation) "
    "that truly needs no tool → resend your answer inside <memento_s_final>.\n\n"
    "Do NOT output bare text. Either call a tool or re-wrap your reply in <memento_s_final>."
)


def _extract_paths(text: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for m in _PATH_RE.finditer(text):
        p = m.group(1).rstrip(".,;:)'\"")
        if p and p not in seen and len(p) > 3:
            seen.add(p)
            result.append(p)
    return result


def _verify_paths_note(args_text: str, result_text: str) -> str:
    candidates = _extract_paths(args_text + "\n" + result_text)
    if not candidates:
        return ""

    notes: list[str] = []
    for path_str in candidates:
        try:
            p = Path(path_str)
            if p.exists():
                kind = "dir" if p.is_dir() else "file"
                notes.append(f"  ✓ EXISTS ({kind}): {path_str}")
            elif p.parent.exists():
                notes.append(f"  ✗ NOT FOUND (parent dir exists): {path_str}")
        except Exception:
            pass

    if not notes:
        return ""
    return "\n\n[System Path Verification]\n" + "\n".join(notes)


def _parse_final_marker(content: str | None) -> str | None:
    if not content:
        return None

    if MEMENTO_S_FINAL_TAG not in content.lower():
        return None

    for pat in _END_MARKER_PATTERNS:
        match = pat.search(content)
        if match:
            inner_text = match.group(1).strip()
            if inner_text:
                return inner_text

    return None


def _response_to_skill_calls(response: LLMResponse) -> list[SkillCall]:
    return [
        SkillCall(id=tc.id, name=tc.name, arguments=tc.arguments)
        for tc in response.tool_calls
    ]


class MementoSAgent:

    def __init__(
        self,
        workspace: Path,
        *,
        llm: LLM | None = None,
        context_manager: StatefulContextManager | None = None,
        skill_manager: SkillManager | None = None,
        session_manager: SessionManager | None = None,
        model: str | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.llm = llm if llm is not None else LLM()
        _app_context = create_app_context(llm=self.llm)
        self.skill_manager = skill_manager if skill_manager is not None else SkillManager(provider=DeltaSkillsProvider(app_context=_app_context))
        self.session_manager = session_manager if session_manager is not None else SessionManager(self.workspace)
        self.context_manager = context_manager if context_manager is not None else StatefulContextManager(
            workspace=self.workspace,
            skill_manager=self.skill_manager,
            session_manager=self.session_manager,
        )
        self.model = model
        configure_builtin_tools(self.workspace, skill_library=_app_context.library)

    async def _llm_step(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]] | None,
    ) -> LLMResponse:
        tools_count = len(tool_schemas or [])
        model_name = self.model or self.llm.default_model
        logger.info(
            "LLM step: model=%s, messages=%d, tools=%d",
            model_name,
            len(messages),
            tools_count,
        )
        response = await self.llm.chat(
            messages=messages,
            tools=tool_schemas,
            model=self.model,
        )
        if response.tool_calls:
            names = {tc.name for tc in response.tool_calls}
            logger.info(
                "LLM response: %d tool_calls -> %s",
                len(response.tool_calls),
                ", ".join(sorted(names)),
            )
        else:
            logger.info(
                "LLM response: no tool_calls, content_len=%d",
                len(response.content or ""),
            )
        return response

    async def _append_skill_turn(
        self,
        messages: list[dict[str, Any]],
        session_list: list[dict[str, Any]],
        response: LLMResponse,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        skill_calls = _response_to_skill_calls(response)
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": response.content or "",
            "tool_calls": [skill_call_to_openai_payload(sc) for sc in skill_calls],
        }
        tool_msgs: list[dict[str, Any]] = []
        for sc in skill_calls:
            args_str = json.dumps(sc.arguments, ensure_ascii=False)
            logger.info("Skill call: %s(%s)", sc.name, args_str[:200])
            try:
                if is_builtin_tool(sc.name):
                    result = await execute_builtin_tool(sc.name, sc.arguments)
                else:
                    result = await self.skill_manager.call(sc.name, sc.arguments)
                result_preview = (result[:200] + "…") if len(result) > 200 else result
                logger.info(
                    "Skill call result: %s -> len=%d, preview=%s",
                    sc.name, len(result), result_preview,
                )
            except Exception as e:
                logger.exception("Skill %s failed: %s", sc.name, e)
                result = f"Error: {e}"
                logger.info("Skill call result: %s -> Error: %s", sc.name, e)

            verification = _verify_paths_note(args_str, result)
            if verification:
                result = result + verification
                logger.info("Layer2 path verification appended for %s", sc.name)

            self._any_tool_called = True
            tool_msgs.append({"role": "tool", "tool_call_id": sc.id, "content": result})

        new_messages = list(messages) + [assistant_msg] + tool_msgs
        new_session_list = list(session_list) + [assistant_msg] + tool_msgs
        return new_messages, new_session_list


    async def _preflight_discover(
        self,
        user_content: str,
        messages: list[dict[str, Any]],
    ) -> None:
        return

    async def _auto_create_skill_on_no_tool_calls(
        self,
        user_content: str,
        messages: list[dict[str, Any]],
    ) -> bool:
        return False

        if created_any:
            new_system = self.context_manager.assemble_system_prompt()
            if messages and messages[0].get("role") == "system":
                messages[0] = {"role": "system", "content": new_system}
            logger.info("Auto-create: refreshed system prompt after creating new skill(s)")

        return created_any

    async def reply(
        self,
        session_id: str,
        user_content: str,
        *,
        media: list[str] | list[Path] | None = None,
        skill_names: list[str] | None = None,
    ) -> str:
        logger.info(
            "Agent.reply: session=%s, user_len=%d, media=%d, skill_filter=%s",
            session_id,
            len(user_content or ""),
            0 if media is None else len(media),
            ",".join(skill_names) if skill_names else "*",
        )
        self._any_tool_called = False

        history = await self.session_manager.load_messages(session_id)
        messages = await self.context_manager.assemble_messages(
            history=history,
            current_message=user_content,
            skill_names=skill_names,
            media=media,
        )
        user_msg = messages[-1]
        session_list: list[dict[str, Any]] = list(history) + [user_msg]

        tool_schemas = list(BUILTIN_TOOL_SCHEMAS)
        logger.info(
            "Tools: %d total (%s)",
            len(tool_schemas),
            ", ".join(sorted(s.get("function", s).get("name", "") for s in tool_schemas)),
        )

        await self._preflight_discover(user_content, messages)

        matched_skills_context = await asyncio.to_thread(
            self.skill_manager.get_matched_skills_context, user_content, 5,
        )
        if matched_skills_context:
            user_msg = messages[-1]
            if isinstance(user_msg.get("content"), str):
                user_msg["content"] = user_msg["content"] + "\n\n" + matched_skills_context
            logger.info("Injected matched skills context into user message")

        final_content: str | None = None
        max_iter = g_settings.agent_max_iterations
        reminder_count = 0
        max_reminders = 2
        no_tool_self_checked = False  # Layer 1: fired at most once per turn

        for step in range(1, max_iter + 1):
            logger.info("ReAct step %d/%d", step, max_iter)
            response = await self._llm_step(messages, tool_schemas)

            if response.tool_calls:
                messages, session_list = await self._append_skill_turn(messages, session_list, response)
                logger.info(
                    "ReAct: stacked assistant + %d tool result(s), messages len=%d",
                    len(response.tool_calls),
                    len(messages),
                )
                if any(tc.name in ("skill_creator", "bash_tool") and "skill-creator" in json.dumps(tc.arguments) for tc in response.tool_calls):
                    library = getattr(self.skill_manager._provider, "_library", None)
                    if library:
                        await asyncio.to_thread(library.refresh_from_disk)
                    logger.info("Refreshed skill library after skill_creator")
                reminder_count = 0
                continue

            content = response.content or ""
            parsed = _parse_final_marker(content)
            if parsed is not None:
                if not self._any_tool_called and not no_tool_self_checked:
                    no_tool_self_checked = True
                    assistant_msg = {"role": "assistant", "content": content}
                    check_msg = {"role": "user", "content": _LAYER1_SELF_CHECK}
                    messages = list(messages) + [assistant_msg, check_msg]
                    logger.info(
                        "Layer1: no skills called yet but model sent <memento_s_final>; "
                        "injecting self-check (step %d).",
                        step,
                    )
                    continue

                final_content = parsed
                assistant_msg = {"role": "assistant", "content": content}
                session_list = list(session_list) + [assistant_msg]
                logger.info("ReAct: received end marker <memento_s_final>, ending turn.")
                break

            if reminder_count == 0:
                created = await self._auto_create_skill_on_no_tool_calls(
                    user_content, messages,
                )
                if created:
                    assistant_msg = {"role": "assistant", "content": content}
                    hint_msg = {
                        "role": "user",
                        "content": (
                            "[System] New skill(s) have been auto-created. "
                            "Use `read_skill` to learn about them, then proceed."
                        ),
                    }
                    messages = list(messages) + [assistant_msg, hint_msg]
                    logger.info("ReAct: auto-created skill(s), giving LLM another chance.")
                    continue

            if reminder_count >= max_reminders:
                final_content = content
                assistant_msg = {"role": "assistant", "content": content}
                session_list = list(session_list) + [assistant_msg]
                logger.warning(
                    "ReAct: reached max reminders (%d), treating text as final reply.",
                    max_reminders,
                )
                break

            reminder = (
                "[System] VIOLATION: Your reply was NOT wrapped in <memento_s_final>. "
                "The system CANNOT deliver unwrapped text to the user.\n\n"
                "Step 1: Evaluate — is the task fully complete?\n"
                "Step 2: If YES → resend your reply wrapped in <memento_s_final>...</memento_s_final>\n"
                "Step 3: If NO → call the appropriate tool to continue.\n\n"
                "You MUST do one of the above. Do NOT output plain text again."
            )
            assistant_msg = {"role": "assistant", "content": content}
            reminder_msg = {"role": "user", "content": reminder}
            messages = list(messages) + [assistant_msg, reminder_msg]
            reminder_count += 1
            logger.info(
                "ReAct: no tool_calls and no end marker, sent reminder (%d/%d) for <memento_s_final>, continuing.",
                reminder_count, max_reminders,
            )

        if final_content is None:
            final_content = ""
            session_list = list(session_list) + [{"role": "assistant", "content": final_content}]

        await self.session_manager.save_messages(session_id, session_list)
        return final_content

    async def reply_stream(
        self,
        session_id: str,
        user_content: str,
        *,
        media: list[str] | list[Path] | None = None,
        skill_names: list[str] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        logger.info(
            "Agent.reply_stream: session=%s, user_len=%d, media=%d, skill_filter=%s",
            session_id,
            len(user_content or ""),
            0 if media is None else len(media),
            ",".join(skill_names) if skill_names else "*",
        )
        self._any_tool_called = False

        try:
            history = await self.session_manager.load_messages(session_id)
            messages = await self.context_manager.assemble_messages(
                history=history,
                current_message=user_content,
                skill_names=skill_names,
                media=media,
            )
            user_msg = messages[-1]
            session_list: list[dict[str, Any]] = list(history) + [user_msg]

            tool_schemas = list(BUILTIN_TOOL_SCHEMAS)
            logger.info(
                "Tools (stream): %d total (%s)",
                len(tool_schemas),
                ", ".join(sorted(s.get("function", s).get("name", "") for s in tool_schemas)),
            )

            await self._preflight_discover(user_content, messages)

            matched_skills_context = await asyncio.to_thread(
                self.skill_manager.get_matched_skills_context, user_content, 5,
            )
            if matched_skills_context:
                user_msg = messages[-1]
                if isinstance(user_msg.get("content"), str):
                    user_msg["content"] = user_msg["content"] + "\n\n" + matched_skills_context
                logger.info("Injected matched skills context into user message (stream)")

            final_content: str | None = None
            max_iter = g_settings.agent_max_iterations
            reminder_count = 0
            max_reminders = 2
            no_tool_self_checked = False  # Layer 1: fired at most once per turn

            for step in range(1, max_iter + 1):
                logger.info("ReAct stream step %d/%d", step, max_iter)
                yield {"type": "status", "message": f"Thinking (step {step})..."}

                content_parts: list[str] = []
                tool_calls_raw: dict[int, dict[str, Any]] = {}
                model_name = self.model or self.llm.default_model
                logger.info(
                    "LLM stream step: model=%s, messages=%d, tools=%d",
                    model_name, len(messages), len(tool_schemas),
                )

                async for chunk in self.llm.chat_stream(
                    messages=messages,
                    tools=tool_schemas,
                    model=self.model,
                ):
                    if chunk.delta_content:
                        content_parts.append(chunk.delta_content)
                        yield {"type": "text_delta", "content": chunk.delta_content}
                    if chunk.tool_calls_delta:
                        for tc_delta in chunk.tool_calls_delta:
                            idx = getattr(tc_delta, "index", 0) if hasattr(tc_delta, "index") else 0
                            if idx not in tool_calls_raw:
                                tool_calls_raw[idx] = {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            entry = tool_calls_raw[idx]
                            tc_id = getattr(tc_delta, "id", None)
                            if tc_id:
                                entry["id"] = tc_id
                            func = getattr(tc_delta, "function", None)
                            if func:
                                func_name = getattr(func, "name", None)
                                if func_name:
                                    entry["function"]["name"] += str(func_name)
                                func_args = getattr(func, "arguments", None)
                                if func_args:
                                    entry["function"]["arguments"] += str(func_args)

                from core.llm.client import _parse_tool_calls
                assembled_content = "".join(content_parts) if content_parts else None
                assembled_tool_calls = (
                    _parse_tool_calls([tool_calls_raw[k] for k in sorted(tool_calls_raw)])
                    if tool_calls_raw else []
                )
                response = LLMResponse(content=assembled_content, tool_calls=assembled_tool_calls)

                if response.tool_calls:
                    logger.info(
                        "LLM stream response: %d tool_calls -> %s",
                        len(response.tool_calls),
                        ", ".join(sorted(tc.name for tc in response.tool_calls)),
                    )
                else:
                    logger.info(
                        "LLM stream response: no tool_calls, content_len=%d",
                        len(response.content or ""),
                    )

                if response.tool_calls:
                    skill_calls = _response_to_skill_calls(response)
                    assistant_msg: dict[str, Any] = {
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": [skill_call_to_openai_payload(sc) for sc in skill_calls],
                    }
                    tool_msgs: list[dict[str, Any]] = []
                    for sc in skill_calls:
                        yield {
                            "type": "skill_call_start",
                            "skill_name": sc.name,
                            "call_id": sc.id,
                            "arguments": sc.arguments,
                        }
                        args_str = json.dumps(sc.arguments, ensure_ascii=False)
                        logger.info("Skill call (stream): %s(%s)", sc.name, args_str[:200])
                        try:
                            if is_builtin_tool(sc.name):
                                result = await execute_builtin_tool(sc.name, sc.arguments)
                            else:
                                result = await self.skill_manager.call(sc.name, sc.arguments)
                            result_preview = (result[:200] + "…") if len(result) > 200 else result
                            logger.info(
                                "Skill result (stream): %s -> len=%d, preview=%s",
                                sc.name, len(result), result_preview,
                            )
                        except Exception as e:
                            logger.exception("Skill %s failed (stream): %s", sc.name, e)
                            result = f"Error: {e}"

                        verification = _verify_paths_note(args_str, result)
                        if verification:
                            result = result + verification
                            logger.info("Layer2 path verification appended for %s (stream)", sc.name)

                        self._any_tool_called = True
                        yield {
                            "type": "skill_call_result",
                            "skill_name": sc.name,
                            "call_id": sc.id,
                            "result": result,
                        }
                        tool_msgs.append({"role": "tool", "tool_call_id": sc.id, "content": result})

                    messages = list(messages) + [assistant_msg] + tool_msgs
                    session_list = list(session_list) + [assistant_msg] + tool_msgs
                    reminder_count = 0
                    logger.info(
                        "ReAct stream: stacked assistant + %d tool result(s), messages len=%d",
                        len(response.tool_calls), len(messages),
                    )
                    if any(tc.name in ("skill_creator", "bash_tool") and "skill-creator" in json.dumps(tc.arguments) for tc in response.tool_calls):
                        library = getattr(self.skill_manager._provider, "_library", None)
                        if library:
                            await asyncio.to_thread(library.refresh_from_disk)
                        logger.info("Refreshed skill library after skill_creator")
                    continue

                content = response.content or ""
                parsed = _parse_final_marker(content)
                if parsed is not None:
                    if not self._any_tool_called and not no_tool_self_checked:
                        no_tool_self_checked = True
                        assistant_msg = {"role": "assistant", "content": content}
                        check_msg = {"role": "user", "content": _LAYER1_SELF_CHECK}
                        messages = list(messages) + [assistant_msg, check_msg]
                        logger.info(
                            "Layer1 (stream): no skills called yet but model sent "
                            "<memento_s_final>; injecting self-check (step %d).",
                            step,
                        )
                        continue

                    final_content = parsed
                    assistant_msg = {"role": "assistant", "content": content}
                    session_list = list(session_list) + [assistant_msg]
                    logger.info("ReAct stream: received end marker <memento_s_final>, ending turn.")
                    break

                if reminder_count >= max_reminders:
                    final_content = content
                    assistant_msg = {"role": "assistant", "content": content}
                    session_list = list(session_list) + [assistant_msg]
                    logger.warning(
                        "ReAct stream: reached max reminders (%d), treating text as final reply.",
                        max_reminders,
                    )
                    break

                reminder = (
                    "[System] VIOLATION: Your reply was NOT wrapped in <memento_s_final>. "
                    "The system CANNOT deliver unwrapped text to the user.\n\n"
                    "Step 1: Evaluate — is the task fully complete?\n"
                    "Step 2: If YES → resend your reply wrapped in <memento_s_final>...</memento_s_final>\n"
                    "Step 3: If NO → call the appropriate tool to continue.\n\n"
                    "You MUST do one of the above. Do NOT output plain text again."
                )
                assistant_msg = {"role": "assistant", "content": content}
                reminder_msg = {"role": "user", "content": reminder}
                messages = list(messages) + [assistant_msg, reminder_msg]
                reminder_count += 1
                logger.info(
                    "ReAct stream: no end marker, sent reminder (%d/%d), continuing.",
                    reminder_count, max_reminders,
                )

            if final_content is None:
                final_content = ""
                session_list = list(session_list) + [{"role": "assistant", "content": final_content}]

            await self.session_manager.save_messages(session_id, session_list)
            yield {"type": "final", "content": final_content}

        except Exception as exc:
            logger.exception("Agent.reply_stream error: %s", exc)
            yield {"type": "error", "message": str(exc)}
