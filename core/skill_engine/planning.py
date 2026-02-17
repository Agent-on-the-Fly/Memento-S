"""Plan generation and validation helpers for skill execution."""

from __future__ import annotations

import json
import re
from typing import Any

from core.config import DEBUG
from core.llm import openrouter_messages
from core.utils.json_utils import parse_json_output
from core.utils.logging_utils import log_event
from core.skill_engine.skill_executor import normalize_plan_shape

def ask_for_plan(
    user_text: str,
    skill_md: str,
    skill_name: str,
    messages: list[dict] | None = None,
) -> dict:
    """
    Produce a single skill plan (JSON).

    If *messages* is provided it is used as the full message history (already
    containing the loaded SKILL.md and user prompt + any tool outputs).
    """
    if messages is None:
        messages = [
            {"role": "user", "content": f"# Loaded SKILL.md\n\n{skill_md}"},
            {"role": "user", "content": user_text},
        ]
    log_event(
        "ask_for_plan_input",
        skill_name=skill_name,
        user_text=user_text,
        messages=messages,
    )

    system_prompt = (
        "Follow the loaded SKILL.md exactly and return JSON only (no markdown). "
        'If no external actions are needed, return {"final":"..."}. '
        'If actions are needed, return OpenAI-style {"tool_calls":[...]} (type=function, function.name, '
        "function.arguments JSON string) using bridge-friendly tool names only: "
        "call_skill, run_command/shell, filesystem tools (read_file/write_file/edit_file/replace_text/append_file/"
        "list_directory/directory_tree/create_directory/mkdir/move_file/copy_file/delete_file/file_info/"
        "search_files/file_exists), web tools (web_search/search/google_search/fetch/fetch_url/fetch_markdown), "
        "and uv tools (check/install/list). "
        "For call_skill include 'skill' and 'plan' (or 'tool_calls'). "
        "For bundled skill resources, commands may use relative paths like scripts/... or references/...; "
        "the runtime resolves them against the active skill directory. "
        "For intermediate/generated files without an explicit user path, prefer placing them under workspace/."
    )

    output = openrouter_messages(system_prompt, messages)
    log_event("ask_for_plan_raw_output", skill_name=skill_name, output=output)
    if DEBUG:
        print(f"Assistant(raw)>\n{output}\n")
    plan = normalize_plan_shape(parse_json_output(output))
    log_event("ask_for_plan_parsed", skill_name=skill_name, plan=plan)

    if not plan and output.strip():
        log_event("ask_for_plan_fallback_final", skill_name=skill_name, final=output.strip())
        return {"final": output.strip()}

    # --- skill-creator specific retry ---
    if (
        skill_name == "skill-creator"
        and not plan.get("action")
        and not plan.get("files")
        and not plan.get("SKILL.md")
        and not plan.get("tool_calls")
    ):
        retry_schema = (
            "Return ONLY a single JSON object with keys: "
            "action, skills_dir, skill_name, tool_calls, notes. "
            "No other keys. No prose.\n\n"
            "Schema:\n"
            "{\n"
            '  "action": "create" | "update",\n'
            '  "skills_dir": "skills",\n'
            '  "skill_name": "skill-name",\n'
            '  "tool_calls": [\n'
            '    { "id":"call_1","type":"function","function":{"name":"mkdir","arguments":"{\\"path\\":\\"references\\"}"} },\n'
            '    { "id":"call_2","type":"function","function":{"name":"write_file","arguments":"{\\"path\\":\\"SKILL.md\\",\\"content\\":\\"...\\",\\"overwrite\\":true}"} },\n'
            '    { "id":"call_3","type":"function","function":{"name":"write_file","arguments":"{\\"path\\":\\"scripts/run.py\\",\\"content\\":\\"...\\",\\"overwrite\\":true}"} }\n'
            "  ],\n"
            '  "notes": "short human-readable summary"\n'
            "}\n"
        )
        retry_output = openrouter_messages(
            "Return ONLY valid JSON. No extra keys. No prose.",
            messages + [{"role": "user", "content": retry_schema}],
        )
        log_event("ask_for_plan_retry_raw_output", skill_name=skill_name, output=retry_output, retry_stage="schema")
        if DEBUG:
            print(f"Assistant(raw retry)>\n{retry_output}\n")
        plan = normalize_plan_shape(parse_json_output(retry_output))
        log_event("ask_for_plan_retry_parsed", skill_name=skill_name, plan=plan, retry_stage="schema")

    # --- strict schema retry ---
    if not validate_plan_for_skill(plan, skill_name):
        strict = build_strict_schema_prompt(skill_name)
        retry_output = openrouter_messages(
            "Return ONLY valid JSON. No extra keys. No prose.",
            messages + [{"role": "user", "content": strict}],
        )
        log_event("ask_for_plan_retry_raw_output", skill_name=skill_name, output=retry_output, retry_stage="strict")
        if DEBUG:
            print(f"Assistant(raw retry2)>\n{retry_output}\n")
        plan = normalize_plan_shape(parse_json_output(retry_output))
        log_event("ask_for_plan_retry_parsed", skill_name=skill_name, plan=plan, retry_stage="strict")
        if not plan and retry_output.strip():
            log_event("ask_for_plan_fallback_final", skill_name=skill_name, final=retry_output.strip())
            return {"final": retry_output.strip()}
        if not validate_plan_for_skill(plan, skill_name):
            keys = ", ".join(sorted(plan.keys())) if isinstance(plan, dict) else str(type(plan))
            rejected = {
                "_handled": True,
                "result": (
                    "Invalid skill plan (refused). Expected JSON with non-empty "
                    "`tool_calls` (legacy `ops` also accepted), or a final answer like {\"final\":\"...\"} "
                    "(and not `type=code`). "
                    f"Got keys: {keys}"
                ),
            }
            log_event("ask_for_plan_rejected", skill_name=skill_name, plan=plan, rejection=rejected)
            return rejected

    normalized_plan = normalize_skill_creator_plan(plan)
    log_event("ask_for_plan_output", skill_name=skill_name, plan=normalized_plan)
    return normalized_plan


# ---------------------------------------------------------------------------
# Plan validation helpers
# ---------------------------------------------------------------------------

def validate_plan_for_skill(plan: dict, skill_name: str) -> bool:
    """Validate that a plan has the correct format (tool_calls array or final answer)."""
    if not isinstance(plan, dict) or not plan:
        return False

    # Reject code-type responses
    if isinstance(plan.get("type"), str) and plan.get("type", "").strip().lower() == "code":
        return False

    # Allow a final answer as escape hatch
    final = plan.get("final")
    if not isinstance(final, str) or not final.strip():
        final = plan.get("result")
    if isinstance(final, str) and final.strip():
        return True

    # All skills should use tool_calls (ops is accepted for back-compat).
    tool_calls = plan.get("tool_calls")
    has_tool_calls = isinstance(tool_calls, list) and len(tool_calls) > 0
    ops = plan.get("ops")
    has_ops = isinstance(ops, list) and len(ops) > 0

    if skill_name == "skill-creator":
        return bool(plan.get("action") and plan.get("skill_name") and (has_tool_calls or has_ops))

    return has_tool_calls or has_ops


def build_strict_schema_prompt(skill_name: str) -> str:
    if skill_name == "skill-creator":
        return (
            "Return ONLY a single JSON object with keys: action, skills_dir, skill_name, tool_calls, notes.\n"
            "No other keys."
        )
    if skill_name == "filesystem":
        return (
            "Return ONLY a single JSON object. No markdown. No prose.\n"
            "CRITICAL: Use OpenAI-style `tool_calls` format. Never use mcp_call or mcp_tool wrappers.\n\n"
            "CORRECT format:\n"
            '{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"directory_tree","arguments":"{\\"path\\":\\"/path\\",\\"depth\\":3}"}}]}\n'
            '{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\\"path\\":\\"/path/to/file\\"}"}}]}\n'
            '{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"edit_file","arguments":"{\\"path\\":\\"/path\\",\\"old_text\\":\\"...\\",\\"new_text\\":\\"...\\"}"}}]}\n\n'
            "WRONG (do not use):\n"
            '{"tool_calls":[{"type":"mcp_call", ...}]}  <- WRONG!\n'
            '{"tool_calls":[{"type":"mcp_tool", ...}]}  <- WRONG!\n'
        )
    if skill_name == "terminal":
        return (
            "Return ONLY a single JSON object. No markdown. No prose.\n"
            "IMPORTANT: Use OpenAI-style `tool_calls` format as defined in SKILL.md.\n"
            "Example:\n"
            '{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"run_command","arguments":"{\\"command\\":\\"git status\\",\\"working_dir\\":\\"/path\\"}"}}]}\n'
            "Do NOT return wrapper formats like mcp_call/mcp_tool.\n"
        )
    return (
        "Return ONLY a single JSON object. No markdown. No prose.\n"
        "Rules:\n"
        "- Do NOT include keys: thoughts, type, code, mcp.\n"
        "- Must include:\n"
        '  - "tool_calls": [...] (non-empty array)\n'
        '  - OR a final answer: {"final":"..."}\n'
        "- Each tool call should be OpenAI-style with type=function + function.name + function.arguments.\n"
        "- For bundled skill resources, prefer relative paths like scripts/... and references/...\n"
        "- Follow the loaded SKILL.md schema exactly.\n"
    )


def _tool_call(name: str, arguments: dict[str, Any], *, call_id: str) -> dict[str, Any]:
    """Build one OpenAI-style function tool call."""
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def normalize_skill_creator_plan(plan: dict) -> dict:
    """Back-compat: convert \"files\" payloads into skill-creator tool_calls when action is missing."""
    if plan.get("action"):
        return plan

    if isinstance(plan.get("SKILL.md"), str):
        skill_md = plan.get("SKILL.md", "")
        skill_name = "new-skill"
        m = re.search(r"^name:\s*(.+)$", skill_md, re.MULTILINE)
        if m:
            skill_name = m.group(1).strip().strip('"').strip("'")
        script_stub = (
            "def main():\n"
            "    raise SystemExit('Not implemented: fill in script logic')\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
        return {
            "action": "create",
            "skills_dir": "skills",
            "skill_name": skill_name,
            "tool_calls": [
                _tool_call(
                    "write_file",
                    {
                        "path": "SKILL.md",
                        "content": skill_md,
                        "overwrite": True,
                    },
                    call_id="call_1",
                ),
                _tool_call(
                    "write_file",
                    {
                        "path": "scripts/run.py",
                        "content": script_stub,
                        "overwrite": True,
                    },
                    call_id="call_2",
                ),
            ],
            "notes": "normalized from SKILL.md payload",
        }

    files = plan.get("files")
    if not isinstance(files, list) or not files:
        return plan

    first_path = files[0].get("path", "")
    skill_name = None
    if isinstance(first_path, str) and "/" in first_path:
        skill_name = first_path.split("/", 1)[0].strip() or None

    write_calls: list[dict[str, Any]] = []
    write_paths: list[str] = []
    for f in files:
        path = f.get("path")
        content = f.get("content", "")
        if not path:
            continue
        rel = path
        if skill_name and path.startswith(f"{skill_name}/"):
            rel = path[len(skill_name) + 1:]
        write_paths.append(rel)
        write_calls.append(
            _tool_call(
                "write_file",
                {
                    "path": rel,
                    "content": content,
                    "overwrite": True,
                },
                call_id=f"call_{len(write_calls) + 1}",
            )
        )
    # Enforce at least one script for operational skills.
    if not any(path.startswith("scripts/") for path in write_paths):
        write_calls.append(
            _tool_call(
                "write_file",
                {
                    "path": "scripts/run.py",
                    "content": (
                        "def main():\n"
                        "    raise SystemExit('Not implemented: fill in script logic')\n"
                        "\n"
                        "if __name__ == '__main__':\n"
                        "    main()\n"
                    ),
                    "overwrite": True,
                },
                call_id=f"call_{len(write_calls) + 1}",
            )
        )

    return {
        "action": "create",
        "skills_dir": "skills",
        "skill_name": skill_name or "new-skill",
        "tool_calls": write_calls,
        "notes": "normalized from files payload",
    }
