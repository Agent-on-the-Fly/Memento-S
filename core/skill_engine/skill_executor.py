"""Skill execution functions extracted from agent.py.

This module contains all plan normalization, skill context helpers,
and executor functions for filesystem, terminal, web, uv-pip, and
skill-creator operations, plus the bridge dispatcher.
"""

import json
from pathlib import Path
from typing import Any, Callable

from core.config import (
    BUILTIN_BRIDGE_SKILLS,
)
from core.utils.logging_utils import log_event
from core.utils.path_utils import (
    _parse_json_object,
    _safe_subpath,
)
from . import executor_utils as _executor_utils
from .executor.executor_fs import (
    execute_filesystem_op as _delegate_execute_filesystem_op,
    execute_filesystem_ops as _delegate_execute_filesystem_ops,
    filesystem_tree as _delegate_filesystem_tree,
)
from .executor.executor_terminal import (
    convert_pip_to_uv as _delegate_convert_pip_to_uv,
    execute_terminal_ops as _delegate_execute_terminal_ops,
    execute_uv_pip_ops as _delegate_execute_uv_pip_ops,
    run_uv_pip as _delegate_run_uv_pip,
)
from .executor.executor_web import (
    execute_web_ops as _delegate_execute_web_ops,
    fetch_async as _delegate_fetch_async,
    web_fetch as _delegate_web_fetch,
    web_google_search as _delegate_web_google_search,
)
from .bridge.dispatcher import (
    coerce_call_stack as _bridge_coerce_call_stack,
    dispatch_bridge_op as _bridge_dispatch_bridge_op,
)
from .bridge.registry import build_tool_registry
from .error_model import build_execution_result_payload, format_error, infer_ok_from_output
from core.skill_engine.skill_resolver import _resolve_skill_dir


def _canonicalize_op_type(raw_type: Any) -> str:
    return _executor_utils.canonicalize_op_type(raw_type)


def _parse_bool(value: Any, default: bool = False) -> bool:
    return _executor_utils.parse_bool(value, default)


def _parse_int(
    value: Any,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    return _executor_utils.parse_int(
        value,
        default,
        minimum=minimum,
        maximum=maximum,
    )


def _coerce_existing_dir(raw_dir: Any) -> Path | None:
    """Resolve raw directory path and return it only when it exists."""
    if not isinstance(raw_dir, str) or not raw_dir.strip():
        return None
    path = Path(raw_dir.strip())
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    if path.exists() and path.is_dir():
        return path
    return None


def _read_skill_context(raw_ctx: Any) -> tuple[str | None, Path | None]:
    """Parse raw _skill_context payload into (name, dir)."""
    if not isinstance(raw_ctx, dict):
        return None, None
    raw_name = raw_ctx.get("name")
    skill_name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else None
    return skill_name, _coerce_existing_dir(raw_ctx.get("dir"))




# ===================================================================
# 1. Plan normalization
# ===================================================================

def _normalize_op_dict(op: Any) -> dict[str, Any] | None:
    """Normalize one operation dict, handling legacy wrapper formats."""
    if not isinstance(op, dict):
        return None

    out = dict(op)
    op_type = out.get("type") or out.get("op") or out.get("action")
    if isinstance(op_type, str) and op_type.strip():
        out["type"] = op_type.strip()

    wrapper_type = (out.get("type") or "").strip().lower()
    if wrapper_type in {"mcp_tool", "mcp_call", "mcp"}:
        actual_tool = out.get("tool") or out.get("name")
        args = _parse_json_object(out.get("args") or out.get("arguments") or out.get("parameters"))
        merged: dict[str, Any] = {}
        merged.update(args)
        merged.update({k: v for k, v in out.items() if k not in {"args", "arguments", "parameters"}})
        if isinstance(actual_tool, str) and actual_tool.strip():
            merged["type"] = actual_tool.strip()
        out = merged

    if isinstance(out.get("arguments"), str):
        parsed_args = _parse_json_object(out.get("arguments"))
        if parsed_args:
            merged = dict(parsed_args)
            merged.update({k: v for k, v in out.items() if k != "arguments"})
            out = merged

    if "type" not in out and isinstance(op_type, str) and op_type.strip():
        out["type"] = op_type.strip()

    return out


def _tool_call_to_op(call: Any) -> dict[str, Any] | None:
    """Convert a tool_calls-style entry to a normalized op dict."""
    if not isinstance(call, dict):
        return None

    name = call.get("name") or call.get("tool")
    args = call.get("args") or call.get("arguments") or call.get("parameters")

    fn = call.get("function")
    if isinstance(fn, dict):
        fn_name = fn.get("name")
        if isinstance(fn_name, str) and fn_name.strip():
            name = fn_name
        args = args or fn.get("arguments")
    if not isinstance(name, str) or not name.strip():
        fallback = call.get("type")
        if isinstance(fallback, str) and fallback.strip() and fallback.strip().lower() != "function":
            name = fallback

    args_dict = _parse_json_object(args)
    if not isinstance(name, str) or not name.strip():
        return None

    op: dict[str, Any] = {"type": name.strip()}
    op.update(args_dict)
    op.update(
        {
            k: v
            for k, v in call.items()
            if k not in {"name", "tool", "type", "function", "args", "arguments", "parameters"}
        }
    )
    return op


def _op_to_tool_call(op: Any, *, call_id: str) -> dict[str, Any] | None:
    """Convert one op-style entry into an OpenAI-style tool_call object."""
    normalized = _normalize_op_dict(op)
    if not isinstance(normalized, dict):
        return None

    op_type = str(normalized.get("type") or "").strip()
    if not op_type:
        return None

    raw_args = normalized.get("args") or normalized.get("arguments") or normalized.get("parameters")
    args: dict[str, Any] = _parse_json_object(raw_args)
    for key, value in normalized.items():
        if key in {
            "type",
            "op",
            "action",
            "name",
            "tool",
            "function",
            "id",
            "args",
            "arguments",
            "parameters",
        }:
            continue
        args[key] = value

    try:
        args_json = json.dumps(args, ensure_ascii=False)
    except TypeError:
        args_json = json.dumps({k: str(v) for k, v in args.items()}, ensure_ascii=False)

    raw_id = normalized.get("id")
    resolved_id = str(raw_id).strip() if isinstance(raw_id, str) and raw_id.strip() else call_id
    tool_call: dict[str, Any] = {
        "id": resolved_id,
        "type": "function",
        "function": {
            "name": op_type,
            "arguments": args_json,
        },
    }
    depends_on = normalized.get("depends_on")
    if isinstance(depends_on, list):
        tool_call["depends_on"] = depends_on
    policy = normalized.get("policy")
    if isinstance(policy, dict):
        tool_call["policy"] = policy
    protocol_version = normalized.get("protocol_version")
    if isinstance(protocol_version, str) and protocol_version.strip():
        tool_call["protocol_version"] = protocol_version.strip()
    return tool_call


def normalize_plan_shape(plan: Any) -> dict:
    """Normalize plan into both OpenAI `tool_calls` and internal `ops` lists."""
    if not isinstance(plan, dict):
        return {}

    normalized = dict(plan)
    normalized_ops: list[dict[str, Any]] = []

    def _append_from_tool_calls(items: list[Any]) -> None:
        for raw_call in items:
            op = _tool_call_to_op(raw_call)
            if op:
                normalized_ops.append(op)

    def _append_from_ops(items: list[Any]) -> None:
        for raw_op in items:
            op = _normalize_op_dict(raw_op)
            if op:
                normalized_ops.append(op)

    tool_calls = normalized.get("tool_calls")
    ops = normalized.get("ops")
    calls = normalized.get("calls")
    if isinstance(tool_calls, list) and tool_calls:
        _append_from_tool_calls(tool_calls)
    elif isinstance(ops, list) and ops:
        _append_from_ops(ops)
    elif isinstance(calls, list) and calls:
        _append_from_tool_calls(calls)

    normalized["ops"] = normalized_ops

    normalized_calls: list[dict[str, Any]] = []
    for idx, op in enumerate(normalized_ops, start=1):
        call = _op_to_tool_call(op, call_id=f"call_{idx}")
        if call:
            normalized_calls.append(call)
    normalized["tool_calls"] = normalized_calls

    return normalized


# ===================================================================
# 2. Skill context helpers
# ===================================================================

def _coerce_skill_context(plan: dict, fallback_skill: str) -> dict[str, str]:
    """Build/coerce _skill_context metadata on a plan dict."""
    raw = plan.get("_skill_context")
    ctx: dict[str, str] = dict(raw) if isinstance(raw, dict) else {}
    raw_name, skill_dir = _read_skill_context(ctx)
    name = raw_name or fallback_skill.strip()
    ctx["name"] = name

    if skill_dir is None:
        skill_dir = _resolve_skill_dir(name)
    if skill_dir is not None:
        ctx["dir"] = str(skill_dir)

    return ctx


def _extract_skill_context(plan: dict) -> tuple[str | None, Path | None, bool]:
    """Extract (skill_name, skill_dir, prefer_skill_paths) from plan."""
    skill_name, skill_dir = _read_skill_context(plan.get("_skill_context"))

    if skill_dir is None and skill_name:
        skill_dir = _resolve_skill_dir(skill_name)

    prefer_skill_paths = bool(skill_name and skill_name not in BUILTIN_BRIDGE_SKILLS)
    return skill_name, skill_dir, prefer_skill_paths


# ===================================================================
# 3. Skill creator executor
# ===================================================================

def _execute_skill_creator_plan(plan: dict) -> str:
    """Execute a skill-creator plan (create/update a skill directory)."""
    action = plan.get("action")
    skill_name = plan.get("skill_name")
    ops = plan.get("ops", [])

    if action not in {"create", "update"}:
        return f"Invalid action: {action}"
    if not isinstance(skill_name, str) or not skill_name.strip():
        return "Missing skill_name"
    if not isinstance(ops, list):
        return "Invalid tool_calls"

    raw_skills_dir = str(plan.get("skills_dir") or "").strip()
    if action == "create":
        skills_dir = "skill_extra"
    else:
        skills_dir = raw_skills_dir or "skill_extra"

    base = (Path(skills_dir) / skill_name.strip()).resolve()
    report: list[str] = []

    if action == "create":
        base.mkdir(parents=True, exist_ok=True)
        report.append(f"ensure_dir OK: {base}")
    elif action == "update" and not base.exists():
        return f"Skill not found: {base}"

    for op in ops:
        if not isinstance(op, dict):
            report.append("SKIP: op is not a dict")
            continue
        op_type = str(op.get("type") or "").strip()
        rel_path = str(op.get("path") or "").strip()
        if op_type in {"mkdir", "write_file", "append_file", "replace_text"} and not rel_path:
            report.append(f"{op_type} SKIP: missing path")
            continue

        try:
            if op_type == "mkdir":
                p = _safe_subpath(base, rel_path)
                p.mkdir(parents=True, exist_ok=True)
                report.append(f"mkdir OK: {p}")
            elif op_type == "write_file":
                p = _safe_subpath(base, rel_path)
                overwrite = _parse_bool(op.get("overwrite"), True)
                if p.exists() and not overwrite:
                    report.append(f"write_file SKIP (exists): {p}")
                    continue
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(str(op.get("content", "")), encoding="utf-8")
                report.append(f"write_file OK: {p}")
            elif op_type == "append_file":
                p = _safe_subpath(base, rel_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open("a", encoding="utf-8") as f:
                    f.write(str(op.get("content", "")))
                report.append(f"append_file OK: {p}")
            elif op_type == "replace_text":
                p = _safe_subpath(base, rel_path)
                if not p.exists():
                    report.append(f"replace_text SKIP (missing): {p}")
                    continue
                old = str(op.get("old", ""))
                new = str(op.get("new", ""))
                max_n = _parse_int(op.get("max"), 1, minimum=1)
                text = p.read_text(encoding="utf-8")
                if old not in text:
                    report.append(f"replace_text NOOP (not found): {p}")
                    continue
                p.write_text(text.replace(old, new, max_n), encoding="utf-8")
                report.append(f"replace_text OK: {p}")
            else:
                report.append(f"unknown op: {op_type}")
        except Exception as exc:
            report.append(f"{op_type} ERR: {exc}")

    return "\n".join(report) if report else "No tool_calls"


# ===================================================================
# 4. Delegated executor wrappers
# ===================================================================

def _filesystem_tree(
    path: Path,
    prefix: str = "",
    depth: int = 3,
    current_depth: int = 0,
) -> list[str]:
    return _delegate_filesystem_tree(
        path,
        prefix=prefix,
        depth=depth,
        current_depth=current_depth,
    )


def _execute_filesystem_op(
    op: dict,
    base_dir: Path,
    *,
    skill_dir: Path | None = None,
    prefer_skill_paths: bool = False,
) -> str:
    return _delegate_execute_filesystem_op(
        op,
        base_dir,
        skill_dir=skill_dir,
        prefer_skill_paths=prefer_skill_paths,
    )


def _execute_filesystem_ops(plan: dict) -> str:
    return _delegate_execute_filesystem_ops(plan)


def _convert_pip_to_uv(command: str, working_dir: Path) -> str:
    return _delegate_convert_pip_to_uv(command, working_dir)


def _execute_terminal_ops(plan: dict) -> str:
    return _delegate_execute_terminal_ops(plan)


def _run_uv_pip(
    args: list[str],
    working_dir: Path,
    venv_dir: Path,
) -> tuple[int, str, str]:
    return _delegate_run_uv_pip(args, working_dir, venv_dir)


def _execute_uv_pip_ops(plan: dict) -> str:
    return _delegate_execute_uv_pip_ops(plan)


def _web_google_search(query: str, num_results: int = 10) -> list[dict]:
    return _delegate_web_google_search(query, num_results=num_results)


async def _fetch_async(url: str, max_length: int = 50000, raw: bool = False) -> str:
    return await _delegate_fetch_async(url, max_length=max_length, raw=raw)


def _web_fetch(url: str, max_length: int = 50000, raw: bool = False) -> str:
    return _delegate_web_fetch(url, max_length=max_length, raw=raw)


def _execute_web_ops(plan: dict) -> str:
    return _delegate_execute_web_ops(plan)


# ===================================================================
# 9. Bridge dispatcher & execute_skill_plan
# ===================================================================

SkillPlanHandler = Callable[[dict], str]


_SKILL_HANDLERS: dict[str, SkillPlanHandler] = {
    "skill-creator": _execute_skill_creator_plan,
    "filesystem": _execute_filesystem_ops,
    "terminal": _execute_terminal_ops,
    "web-search": _execute_web_ops,
    "uv-pip-install": _execute_uv_pip_ops,
}


_TOOL_REGISTRY = build_tool_registry()


def _coerce_call_stack(raw_stack: Any, current_skill: str | None = None) -> list[str]:
    return _bridge_coerce_call_stack(raw_stack, current_skill)


def _dispatch_bridge_op(
    op: dict,
    parent_plan: dict,
    caller_skill: str,
    *,
    call_id: str,
) -> str:
    return _bridge_dispatch_bridge_op(
        op,
        parent_plan,
        caller_skill,
        call_id=call_id,
        registry=_TOOL_REGISTRY,
        canonicalize_op_type=_canonicalize_op_type,
        parse_json_object=_parse_json_object,
        op_to_tool_call=lambda payload: _op_to_tool_call(payload, call_id=""),
        normalize_plan_shape=normalize_plan_shape,
        execute_skill_plan=execute_skill_plan,
    )


def _build_execution_result(
    *,
    ok: bool,
    output: str,
    skill_name: str,
    code: str = "",
    normalized_plan: dict | None = None,
) -> dict[str, Any]:
    return build_execution_result_payload(
        ok=ok,
        output=output,
        code=code,
        skill_name=skill_name,
        normalized_plan=normalized_plan if isinstance(normalized_plan, dict) else {},
    )


def execute_skill_plan_result(skill_name: str, plan: dict) -> dict[str, Any]:
    """Structured execution result for programmatic callers."""
    normalized = normalize_plan_shape(plan)
    skill = str(skill_name or "").strip()
    normalized["_call_stack"] = _coerce_call_stack(normalized.get("_call_stack"), skill)
    if skill:
        normalized["_skill_context"] = _coerce_skill_context(normalized, skill)
    log_event("execute_skill_plan_input", skill_name=skill, normalized_plan=normalized)
    ops = normalized.get("ops")
    if not isinstance(ops, list) or not ops:
        result = format_error("no tool_calls provided", code="missing_tool_calls")
        log_event("execute_skill_plan_output", skill_name=skill, result=result)
        return _build_execution_result(
            ok=False,
            output=result,
            code="missing_tool_calls",
            skill_name=skill,
            normalized_plan=normalized,
        )

    handler = _SKILL_HANDLERS.get(skill)
    if handler is not None:
        result = handler(normalized)
        log_event("execute_skill_plan_output", skill_name=skill, result=result)
        ok = infer_ok_from_output(result, default=True)
        return _build_execution_result(
            ok=ok,
            output=result,
            code="ok" if ok else "handler_error",
            skill_name=skill,
            normalized_plan=normalized,
        )

    # Generic skill: dispatch each op individually through the bridge
    outputs: list[str] = []
    for idx, raw_op in enumerate(ops, start=1):
        op = _normalize_op_dict(raw_op)
        if not op:
            outputs.append(f"[op#{idx}] SKIP: op is not a dict")
            continue
        op_type = str(op.get("type") or "unknown")
        out = _dispatch_bridge_op(op, normalized, skill, call_id=f"op#{idx}")
        outputs.append(f"[op#{idx}:{op_type}]\n{out}")

    result = "\n\n".join(outputs) if outputs else format_error("no executable tool_calls", code="dispatch_error")
    log_event("execute_skill_plan_output", skill_name=skill, result=result)
    ok = infer_ok_from_output(result, default=True)
    return _build_execution_result(
        ok=ok,
        output=result,
        code="ok" if ok else "dispatch_error",
        skill_name=skill,
        normalized_plan=normalized,
    )


def execute_skill_plan(skill_name: str, plan: dict) -> str:
    """Top-level entry point: execute a skill plan by name."""
    return str(execute_skill_plan_result(skill_name, plan).get("output") or "")


__all__ = [
    "_normalize_op_dict",
    "_tool_call_to_op",
    "_op_to_tool_call",
    "normalize_plan_shape",
    "_coerce_skill_context",
    "_extract_skill_context",
    "_execute_skill_creator_plan",
    "_filesystem_tree",
    "_execute_filesystem_op",
    "_execute_filesystem_ops",
    "_convert_pip_to_uv",
    "_execute_terminal_ops",
    "_run_uv_pip",
    "_execute_uv_pip_ops",
    "_web_google_search",
    "_fetch_async",
    "_web_fetch",
    "_execute_web_ops",
    "_dispatch_bridge_op",
    "execute_skill_plan_result",
    "execute_skill_plan",
]
