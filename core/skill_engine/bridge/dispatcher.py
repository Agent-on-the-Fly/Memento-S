"""Typed bridge dispatch extracted from skill_executor.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from core.config import BUILTIN_BRIDGE_SKILLS
from core.skill_engine.error_model import format_error

from .registry import ToolSpec

TOOL_PROTOCOL_VERSION = "1.0"
_MAX_SKILL_CALL_DEPTH = 12


@dataclass(frozen=True)
class ToolCall:
    id: str
    tool: str
    args: dict[str, Any]
    depends_on: tuple[str, ...] = ()
    policy: dict[str, Any] = field(default_factory=dict)
    protocol_version: str = TOOL_PROTOCOL_VERSION


@dataclass(frozen=True)
class ToolCallResult:
    ok: bool
    data: str = ""
    error_code: str | None = None
    retryable: bool = False


def _tool_error(message: str, *, code: str, retryable: bool = False) -> ToolCallResult:
    return ToolCallResult(
        ok=False,
        data=format_error(message, code=code),
        error_code=code,
        retryable=retryable,
    )


def coerce_call_stack(raw_stack: Any, current_skill: str | None = None) -> list[str]:
    stack: list[str] = []
    if isinstance(raw_stack, list):
        for item in raw_stack:
            if isinstance(item, (str, int, float)):
                name = str(item).strip()
                if name:
                    stack.append(name)

    current = str(current_skill or "").strip()
    if current:
        if not stack:
            stack.append(current)
        elif stack[-1] != current:
            stack.append(current)
    return stack


def _normalize_bridge_op_type(
    op: dict[str, Any],
    *,
    canonicalize_op_type: Callable[[Any], str],
) -> tuple[dict[str, Any], str]:
    normalized_op = dict(op)
    canonical = canonicalize_op_type(normalized_op.get("type"))
    if canonical:
        normalized_op["type"] = canonical
    return normalized_op, canonical


def _extract_tool_args(
    op: dict[str, Any],
    *,
    parse_json_object: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    parsed_args = parse_json_object(op.get("args") or op.get("arguments") or op.get("parameters"))
    args = dict(parsed_args)
    for key, value in op.items():
        if key in {
            "type",
            "op",
            "action",
            "tool",
            "id",
            "depends_on",
            "policy",
            "protocol_version",
            "args",
            "arguments",
            "parameters",
        }:
            continue
        args[key] = value
    return args


def _legacy_op_to_tool_call(
    op: dict[str, Any],
    *,
    call_id: str,
    canonicalize_op_type: Callable[[Any], str],
    parse_json_object: Callable[[Any], dict[str, Any]],
) -> ToolCall:
    normalized_op, canonical_tool = _normalize_bridge_op_type(
        op,
        canonicalize_op_type=canonicalize_op_type,
    )
    depends_on_raw = normalized_op.get("depends_on")
    depends_on: tuple[str, ...] = ()
    if isinstance(depends_on_raw, list):
        depends_on = tuple(
            str(item).strip()
            for item in depends_on_raw
            if isinstance(item, (str, int, float)) and str(item).strip()
        )
    policy = dict(normalized_op.get("policy")) if isinstance(normalized_op.get("policy"), dict) else {}
    raw_id = normalized_op.get("id")
    resolved_id = str(raw_id).strip() if isinstance(raw_id, str) and raw_id.strip() else call_id
    raw_protocol = normalized_op.get("protocol_version")
    protocol_version = (
        str(raw_protocol).strip()
        if isinstance(raw_protocol, str) and raw_protocol.strip()
        else TOOL_PROTOCOL_VERSION
    )
    return ToolCall(
        id=resolved_id,
        tool=canonical_tool,
        args=_extract_tool_args(normalized_op, parse_json_object=parse_json_object),
        depends_on=depends_on,
        policy=policy,
        protocol_version=protocol_version,
    )


def _validate_tool_call(
    call: ToolCall,
    *,
    registry: dict[str, ToolSpec],
) -> ToolCallResult | None:
    tool = str(call.tool or "").strip()
    if not tool:
        return _tool_error("ERR: tool_call missing required function.name", code="missing_tool")

    spec = registry.get(tool)
    if spec is None:
        return _tool_error(f"unknown op type: {tool}", code="unknown_tool")

    if not isinstance(call.args, dict):
        return _tool_error(f"{tool} ERR: args must be an object", code="invalid_args")

    for key in spec.schema.required:
        value = call.args.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            return _tool_error(
                f"{tool} ERR: missing required arg '{key}'",
                code="invalid_args",
            )

    for key, allowed_types in spec.schema.typed.items():
        if key not in call.args or call.args.get(key) is None:
            continue
        value = call.args.get(key)
        if not isinstance(value, allowed_types):
            expected = "/".join(t.__name__ for t in allowed_types)
            return _tool_error(
                f"{tool} ERR: arg '{key}' must be {expected}",
                code="invalid_args",
            )

    return None


def _dispatch_typed_tool_call(
    call: ToolCall,
    parent_plan: dict[str, Any],
    caller_skill: str,
    *,
    registry: dict[str, ToolSpec],
    parse_json_object: Callable[[Any], dict[str, Any]],
    op_to_tool_call: Callable[[Any], dict[str, Any] | None],
    normalize_plan_shape: Callable[[Any], dict],
    execute_skill_plan: Callable[[str, dict], str],
) -> ToolCallResult:
    invalid = _validate_tool_call(call, registry=registry)
    if invalid is not None:
        return invalid

    tool = call.tool
    if tool == "call_skill":
        target = call.args.get("skill") or call.args.get("name")
        if not isinstance(target, str) or not target.strip():
            return _tool_error("call_skill ERR: missing 'skill' name", code="invalid_args")
        target_name = target.strip()
        call_stack = coerce_call_stack(parent_plan.get("_call_stack"), caller_skill)
        if target_name in set(call_stack):
            cycle = " -> ".join([*call_stack, target_name])
            return _tool_error(
                f"call_skill ERR: recursive call cycle detected: {cycle}",
                code="recursive_call",
            )
        if len(call_stack) >= _MAX_SKILL_CALL_DEPTH:
            return _tool_error(
                f"call_skill ERR: max call depth {_MAX_SKILL_CALL_DEPTH} reached",
                code="max_call_depth",
            )

        sub_plan = call.args.get("plan") or parse_json_object(
            call.args.get("args") or call.args.get("arguments")
        )
        if isinstance(sub_plan, list):
            sub_plan = {"tool_calls": sub_plan}
        if not isinstance(sub_plan, dict):
            sub_plan = {}
        if "tool_calls" not in sub_plan and isinstance(call.args.get("tool_calls"), list):
            sub_plan["tool_calls"] = call.args.get("tool_calls")
        if "ops" not in sub_plan and isinstance(call.args.get("ops"), list):
            sub_plan["ops"] = call.args.get("ops")
        if "working_dir" not in sub_plan and parent_plan.get("working_dir"):
            sub_plan["working_dir"] = parent_plan.get("working_dir")
        sub_plan["_call_stack"] = [*call_stack, target_name]
        if (
            target_name in BUILTIN_BRIDGE_SKILLS
            and "_skill_context" not in sub_plan
            and isinstance(parent_plan.get("_skill_context"), dict)
        ):
            sub_plan["_skill_context"] = dict(parent_plan["_skill_context"])
        if not sub_plan:
            return _tool_error("call_skill ERR: missing plan/tool_calls payload", code="invalid_args")
        output = execute_skill_plan(target_name, normalize_plan_shape(sub_plan))
        return ToolCallResult(ok=True, data=output)

    spec = registry.get(tool)
    if spec is None or not spec.target_skill:
        return _tool_error(f"{tool} ERR: no target skill registered", code="dispatch_config", retryable=True)

    forwarded_op: dict[str, Any] = {"type": tool}
    forwarded_op.update(call.args)
    forwarded_call = op_to_tool_call(forwarded_op)
    if forwarded_call is None:
        return _tool_error(f"{tool} ERR: failed to build forwarded tool_call", code="dispatch_config", retryable=True)
    if not str(forwarded_call.get("id") or "").strip():
        forwarded_call["id"] = f"{call.id}:1"
    forwarded: dict[str, Any] = {"tool_calls": [forwarded_call]}
    if parent_plan.get("working_dir") and spec.forward_working_dir:
        forwarded["working_dir"] = parent_plan.get("working_dir")
    if isinstance(parent_plan.get("_skill_context"), dict):
        forwarded["_skill_context"] = dict(parent_plan["_skill_context"])
    output = execute_skill_plan(spec.target_skill, normalize_plan_shape(forwarded))
    return ToolCallResult(ok=True, data=output)


def _format_tool_call_result(result: ToolCallResult) -> str:
    if result.ok:
        return result.data or "OK"
    return result.data or format_error(result.error_code or "tool_error", code=result.error_code or "tool_error")


def dispatch_bridge_op(
    op: dict[str, Any],
    parent_plan: dict[str, Any],
    caller_skill: str,
    *,
    call_id: str,
    registry: dict[str, ToolSpec],
    canonicalize_op_type: Callable[[Any], str],
    parse_json_object: Callable[[Any], dict[str, Any]],
    op_to_tool_call: Callable[[Any], dict[str, Any] | None],
    normalize_plan_shape: Callable[[Any], dict],
    execute_skill_plan: Callable[[str, dict], str],
) -> str:
    call = _legacy_op_to_tool_call(
        op,
        call_id=call_id,
        canonicalize_op_type=canonicalize_op_type,
        parse_json_object=parse_json_object,
    )
    result = _dispatch_typed_tool_call(
        call,
        parent_plan,
        caller_skill,
        registry=registry,
        parse_json_object=parse_json_object,
        op_to_tool_call=op_to_tool_call,
        normalize_plan_shape=normalize_plan_shape,
        execute_skill_plan=execute_skill_plan,
    )
    return _format_tool_call_result(result)


__all__ = [
    "TOOL_PROTOCOL_VERSION",
    "ToolCall",
    "ToolCallResult",
    "coerce_call_stack",
    "dispatch_bridge_op",
]
