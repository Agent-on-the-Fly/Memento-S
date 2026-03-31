"""ToolBridge orchestrator."""

from __future__ import annotations

import json
from typing import Any

from core.skill.config import SkillConfig
from core.skill.execution.tool_bridge.context import ToolContext
from core.skill.execution.tool_bridge.result_processor import ToolResultProcessor
from core.skill.execution.tool_bridge.args_processor import ToolArgsProcessor
from core.skill.execution.policy.tool_gate import ToolGate
from core.skill.execution.policy.path_validator import validate_path
from core.skill.execution.tool_bridge.runner import ToolRunner
from core.skill.schema import ErrorType, Skill, SkillExecutionOutcome
from utils.debug_logger import log_sandbox_exec
from utils.logger import get_logger

logger = get_logger(__name__)


class ToolBridge:
    """Executor 与 Builtin Tools 的中间层（总控）。"""

    def __init__(self, config: "SkillConfig", *, policy_manager):
        self._config = config
        self._policy_manager = policy_manager
        self._args_processor = ToolArgsProcessor()
        self._tool_gate = ToolGate(policy_manager=self._policy_manager)
        self._runner = ToolRunner()
        self._result_processor = ToolResultProcessor()

    async def execute(
        self,
        *,
        skill: Skill,
        tool_calls: list,
        workspace_dir=None,
        session_id: str | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> tuple[SkillExecutionOutcome, str]:
        """Execute tool calls with optional environment variables.

        Args:
            env_vars: Environment variables to inject into tool execution context.
                     Used for PRIMARY_ARTIFACT_PATH and other configuration.
        """
        from core.shared.tools_facade import is_builtin_tool, get_tool_schema

        results: list[dict[str, Any]] = []
        blocked_reason = None
        blocked_error_type: ErrorType | None = None
        first_error_type: ErrorType | None = None
        first_error_detail: dict[str, Any] | None = None

        effective_workspace_dir = workspace_dir or self._config.workspace_dir
        context = ToolContext.from_skill(
            config=self._config,
            skill=skill,
            workspace_dir=effective_workspace_dir,
            session_id=session_id,
        )

        for idx, tc in enumerate(tool_calls, start=1):
            tool_name = (
                tc.name
                if hasattr(tc, "name")
                else tc.get("function", {}).get("name", "")
            )
            if hasattr(tc, "arguments"):
                raw_args = tc.arguments
            else:
                arguments_str = tc.get("function", {}).get("arguments", "{}")
                raw_args = (
                    json.loads(arguments_str)
                    if isinstance(arguments_str, str)
                    else arguments_str
                )

            if not tool_name:
                results.append({"skipped": True, "reason": "no_tool_name"})
                continue

            if not is_builtin_tool(tool_name):
                results.append(
                    {
                        "index": idx,
                        "tool": tool_name,
                        "skipped": True,
                        "reason": "not_builtin_tool",
                    }
                )
                continue

            schema = get_tool_schema(tool_name) or {}
            props = schema.get("properties", {})
            args, mapping_warnings = self._args_processor.process(
                tool_name=tool_name,
                raw_args=raw_args,
                props=props,
                context=context,
            )

            # Validate path boundaries using dynamic allow_roots from context
            path_check = validate_path(
                tool_name=tool_name,
                args=args,
                allow_roots=list(context.allow_roots),
            )
            if not path_check.valid:
                logger.warning(
                    "Path validation failed: {} -> {}", tool_name, path_check.reason
                )
                blocked_reason = path_check.reason
                blocked_error_type = ErrorType.PATH_VALIDATION_FAILED
                first_error_detail = {
                    "tool": tool_name,
                    "category": "path",
                    "message": path_check.reason,
                    "retryable": False,
                    "hint": "Use @ROOT-relative paths and avoid absolute paths outside workspace.",
                    "raw_args": raw_args,
                    "resolved_args": args,
                }
                break

            gate = self._tool_gate.check(tool_name, args)
            if not gate.allowed:
                logger.warning(
                    "Tool call blocked by policy: {} -> {}", tool_name, gate.reason
                )
                blocked_reason = gate.reason
                blocked_error_type = ErrorType.POLICY_BLOCKED
                first_error_detail = {
                    "tool": tool_name,
                    "category": "policy",
                    "message": gate.reason,
                    "retryable": False,
                    "hint": "Adjust tool choice/arguments to satisfy policy constraints.",
                    "raw_args": raw_args,
                    "resolved_args": args,
                }
                break

            logger.info(
                "!Tool call start: #{} tool={} session_id={} args={}",
                idx,
                tool_name,
                session_id,
                args,
            )

            if tool_name == "bash":
                cmd = args.get("command", "")
                log_sandbox_exec(cmd, args.get("work_dir"), {"tool": tool_name})

            try:
                # ENV VAR JAIL: Pass environment variables to tool execution
                tool_result = await self._runner.run(tool_name, args, env_vars=env_vars)

                processed_result, processed = await self._result_processor.process(
                    tool_name=tool_name,
                    tool_result=tool_result,
                    args=args,
                    runner=self._runner,
                )
                tool_result = processed_result
                logger.info(
                    "Tool call done: #{} tool={} result={}",
                    idx,
                    tool_name,
                    processed.summary,
                )
                logger.info(
                    "Tool result decision: #{} tool={} basis={}",
                    idx,
                    tool_name,
                    processed.decision_basis,
                )

                entry = {
                    "index": idx,
                    "tool": tool_name,
                    "args": args,
                    "result": tool_result,
                }
                if mapping_warnings:
                    entry["arg_warnings"] = mapping_warnings
                if processed.warning:
                    entry["warning"] = processed.warning
                results.append(entry)

                if processed.classified_error and first_error_type is None:
                    first_error_type = processed.classified_error[0]
                    first_error_detail = {
                        **processed.classified_error[1],
                        "raw_args": raw_args,
                        "resolved_args": args,
                        "decision_basis": processed.decision_basis,
                    }
            except Exception as e:
                logger.warning(
                    "Tool call failed: #{} tool={} err={}", idx, tool_name, e
                )
                err_msg = str(e)
                results.append(
                    {"index": idx, "tool": tool_name, "args": args, "error": err_msg}
                )
                if first_error_type is None:
                    first_error_type = ErrorType.INTERNAL_ERROR
                    first_error_detail = {
                        "tool": tool_name,
                        "category": "internal",
                        "message": err_msg,
                        "retryable": False,
                        "hint": "Check tool arguments, execution environment, and recent changes.",
                        "raw_args": raw_args,
                        "resolved_args": args,
                    }

        output = f"[Executed {len(tool_calls)} tool call(s)]\n"
        output += json.dumps(results, ensure_ascii=False, indent=2, default=str)

        success = blocked_reason is None and first_error_type is None
        final_error_type = blocked_error_type or first_error_type
        final_error = (
            f"Blocked by policy: {blocked_reason}"
            if blocked_reason
            else (first_error_detail.get("message") if first_error_detail else None)
        )

        logger.info(
            "Tool bridge outcome: success={} error_type={} error={} reason_basis={}",
            success,
            final_error_type.value if final_error_type else None,
            final_error,
            first_error_detail,
        )

        return SkillExecutionOutcome(
            success=success,
            result=output,
            error=final_error,
            error_type=final_error_type,
            error_detail=first_error_detail,
            skill_name=skill.name,
            operation_results=results,
        ), ""
