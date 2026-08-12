# SPDX-License-Identifier: Apache-2.0
"""SkillToolAdapter: bridges SkillAgent to tools/registry.

Migrated from tool_bridge/ — now includes ENV VAR JAIL injection
(ToolRunner logic) alongside the 4-stage argument pipeline.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from shared.schema import SkillConfig
from shared.hooks import HookExecutor
from shared.hooks.types import HookEvent, HookPayload
from core.skill.execution.tool_args_processor import ToolArgsProcessor
from core.skill.execution.tool_context import RuntimeToolContext, ToolContext
from core.skill.execution.tool_result_processor import ToolResultProcessor
from core.skill.schema import ErrorType, Skill
from utils.log_config import log_preview_long
from utils.logger import get_logger

logger = get_logger(__name__)


class SkillToolAdapter:
    """Bridges SkillAgent to the tools/registry for atomic tool execution."""

    def __init__(
        self,
        config: SkillConfig,
        *,
        result_cache=None,
        hook_executor: HookExecutor | None = None,
    ):
        """Initialize the adapter.

        Args:
            config: SkillConfig for the executing skill.
            result_cache: Optional result cache for storing results.
            hook_executor: HookExecutor for lifecycle hooks (policy gates now in hooks).
        """
        self._config = config
        self._result_cache = result_cache
        self._args_processor = ToolArgsProcessor()
        self._result_processor = ToolResultProcessor()
        self._context: RuntimeToolContext | None = None
        self._hook_executor: HookExecutor | None = hook_executor

    def set_context(
        self,
        skill: Skill,
        workspace_dir: Path,
        session_id: str = "",
    ) -> None:
        """Set the execution context for tool calls.

        Args:
            skill: The skill being executed.
            workspace_dir: The workspace directory for the skill run.
            session_id: Skill execution session identifier.
        """
        self._context = RuntimeToolContext.from_skill(
            config=self._config,
            skill=skill,
            workspace_dir=workspace_dir,
            session_id=session_id,
        )

    async def execute(
        self,
        tool_name: str,
        raw_args: dict[str, Any],
        tool_props: dict[str, Any] | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        """Execute an atomic tool through the tools registry.

        Args:
            tool_name: Name of the tool to execute.
            raw_args: Raw arguments for the tool.
            tool_props: Optional tool properties (env vars, timeout, etc.).
            env_vars: Optional environment variables for ENV VAR JAIL.

        Returns:
            Tuple of (observation string, error dict or None).
        """
        from tools import get_registry

        registry = get_registry()

        logger.debug(
            f"[SkillToolAdapter.execute] ENTRY: tool={tool_name}, "
            f"raw_args_keys={list(raw_args.keys())}, "
            f"env_vars_keys={list((env_vars or {}).keys())}"
        )

        # ── Inject skill_deps from tool_props (set by SkillAgent) ─────────────
        # SkillAgent queries tool schema and injects skill_deps into tool_props
        # if the tool declares a skill_deps parameter. Adapter performs transparent
        # merge without any knowledge of specific tool names.
        final_args = dict(raw_args)
        tp_skill_deps = (tool_props or {}).get("skill_deps") or []
        if tp_skill_deps:
            existing = final_args.get("skill_deps") or []
            final_args["skill_deps"] = list(existing) + [
                d for d in tp_skill_deps if d not in existing
            ]

        # Check if tool is registered
        if not registry.is_registered(tool_name):
            logger.warning(
                f"[SkillToolAdapter.execute] TOOL NOT FOUND: tool={tool_name}"
            )
            return (
                f"ERR: tool '{tool_name}' not found in registry.",
                {
                    "error_type": ErrorType.TOOL_NOT_FOUND,
                    "tool": tool_name,
                },
            )

        logger.debug(f"[SkillToolAdapter.execute] tool={tool_name} found in registry")

        # Skip hooks flag — used by execute_with_retry's run() shim to prevent
        # double hook execution (BEFORE is already fired by outer SkillAgent.run())
        skip_hooks = (tool_props or {}).pop("_skip_hooks", False)

        # ── Step 1: _args_processor 始终调用（主流程）──────────────────────────
        # hook 作为补充层，在 processor 之后执行
        args, processor_warnings = self._args_processor.process(
            tool_name=tool_name,
            raw_args=final_args,
            props=tool_props or {},
            context=self._context,
        )

        # 合并 processor 产生的警告
        warnings = processor_warnings

        # ── Step 2: BEFORE_TOOL_EXEC hook（补充层）────────────────────────────
        # PolicyGateHook: 执行策略检查（可阻止）
        # ToolArgsValidationHook: 补充分析 processor 产出的 args
        # skip_hooks=True: retry loop 通过 run() shim 调用 execute()，避免重复执行
        if self._hook_executor is not None and not skip_hooks:
            hook_result = await self._hook_executor.execute(
                HookEvent.BEFORE_TOOL_EXEC,
                HookPayload(
                    event=HookEvent.BEFORE_TOOL_EXEC,
                    tool_name=tool_name,
                    args=args,  # 传入 processor 处理后的 args，hook 做补充分析
                    context=self._context,
                ),
            )
            if not hook_result.allowed:
                logger.warning(
                    f"[SkillToolAdapter] tool={tool_name} blocked by hook: {hook_result.reason}"
                )
                return (
                    f"ERR: tool '{tool_name}' blocked by hook: {hook_result.reason}",
                    {
                        "error_type": ErrorType.POLICY_BLOCKED,
                        "tool": tool_name,
                        "reason": hook_result.reason,
                        "blocked_by": "hook",
                    },
                )

            # 合并 hook 产生的补充警告（processor 仍作为主流程）
            if hook_result.metadata and hook_result.metadata.get("warnings"):
                hook_warnings = hook_result.metadata.get("warnings")
                if warnings:
                    warnings = warnings + hook_warnings
                else:
                    warnings = hook_warnings

        logger.info(
            f"[SkillToolAdapter] args_pipeline: tool={tool_name}, "
            f"raw_keys={list(raw_args.keys())}, "
            f"final_keys={list(args.keys())}, "
            f"skill_name={args.get('skill_name')}, "
            f"context_skill_name={self._context.skill_name if self._context else 'None'}"
        )

        # Extract timeout from tool_props (default 300s, cap at 600s)
        timeout = None
        if tool_props:
            raw_timeout = tool_props.get("timeout")
            if raw_timeout is not None:
                try:
                    timeout = min(int(raw_timeout), 600)
                except (ValueError, TypeError):
                    pass

        # Execute tool with ENV VAR JAIL (with optional timeout)
        execution_result: Any = None
        execution_error: Exception | None = None
        try:
            if timeout:
                execution_result = await asyncio.wait_for(
                    self._run_with_env_vars(tool_name, args, env_vars),
                    timeout=timeout,
                )
            else:
                execution_result = await self._run_with_env_vars(tool_name, args, env_vars)
            logger.debug(
                f"[SkillToolAdapter.execute] registry.execute completed: "
                f"tool={tool_name}, result_type={type(execution_result).__name__}, "
                f"result_preview='{log_preview_long(str(execution_result))}'"
            )
        except asyncio.TimeoutError:
            execution_error = None  # Represented via error dict below
            execution_result = (
                f"ERR: tool '{tool_name}' timed out after {timeout}s",
            )
        except Exception as e:
            execution_error = e
            logger.error(
                f"[SkillToolAdapter.execute] registry.execute EXCEPTION: "
                f"tool={tool_name}, error={e}"
            )
            execution_result = (
                f"ERR: tool '{tool_name}' execution failed: {e}",
            )

        # Handle timeout as a classified error
        if isinstance(execution_result, str) and "timed out" in execution_result:
            timeout_error_dict = {
                "error_type": ErrorType.TIMEOUT,
                "tool": tool_name,
                "message": execution_result,
                "hint": "Increase timeout, split task, or retry with lighter command.",
                "retryable": True,
                "category": "timeout",
            }
            # Still process through result_processor to get the observation format
            result_output, decision = await self._result_processor.process(
                tool_name=tool_name,
                tool_result=execution_result,
                args=args,
                runner=self,
            )
            logger.debug(
                f"[SkillToolAdapter.execute] result processed: "
                f"tool={tool_name}, decision={decision.decision_basis}"
            )
            # NOTE: AFTER_TOOL_EXEC hook 在 agent.py 外层统一调用，不在 adapter 内部重复
            final_obs = str(execution_result)
            if decision.warning:
                final_obs += f"\n[Warning: {decision.warning}]"
            return final_obs, timeout_error_dict

        if execution_error is not None:
            # NOTE: AFTER_TOOL_EXEC hook 在 agent.py 外层统一调用，不在 adapter 内部重复
            return (
                f"ERR: tool '{tool_name}' execution failed: {execution_error}",
                {
                    "error_type": ErrorType.EXECUTION_ERROR,
                    "tool": tool_name,
                    "message": str(execution_error),
                },
            )

        # ── Step 1: _result_processor 始终调用（主流程）────────────────────────
        result_output, decision = await self._result_processor.process(
            tool_name=tool_name,
            tool_result=str(execution_result),
            args=args,
            runner=self,
        )

        logger.debug(
            f"[SkillToolAdapter.execute] result processed: "
            f"tool={tool_name}, decision={decision.decision_basis}, "
            f"warning='{decision.warning or ''}'"
        )

        # ── Step 2: 重试决策（基于 processor 的 decision）────────────────────
        # 注意：AFTER_TOOL_EXEC hook 在 agent.py 外层统一调用，不在 adapter 内部重复执行
        # （避免双重执行导致 FileChangeHook 等状态性 hook 的 before/after 配对错乱）
        classified = decision.classified_error
        retry_err: dict[str, Any] | None = None
        if classified is not None:
            error_type, error_detail = classified
            if error_detail.get("retryable", False):
                logger.info(
                    f"[SkillToolAdapter.execute] retryable error detected for "
                    f"tool={tool_name}: error_type={error_type.value}"
                )
                retry_result, retry_err = await self._result_processor.execute_with_retry(
                    runner=self,
                    tool_name=tool_name,
                    args=args,
                    error_detail=error_detail,
                    max_retries=2,
                )
                if retry_err is None:
                    logger.info(
                        f"[SkillToolAdapter.execute] retry succeeded for tool={tool_name}"
                    )
                    retry_output, retry_decision = await self._result_processor.process(
                        tool_name=tool_name,
                        tool_result=retry_result,
                        args=args,
                        runner=self,
                    )
                    if self._result_cache and retry_decision.decision_basis.get("state") == "succeeded":
                        self._register_result(self._result_cache, tool_name, retry_result)
                else:
                    logger.warning(
                        f"[SkillToolAdapter.execute] all retries exhausted for tool={tool_name}"
                    )
                    execution_result = retry_result
                    decision = retry_decision  # type: ignore[assignment]

        # Cache result if successful
        if self._result_cache and decision.decision_basis.get("state") == "succeeded":
            self._register_result(self._result_cache, tool_name, str(execution_result))

        # Build observation
        obs_parts = [str(execution_result)]
        if warnings:
            if isinstance(warnings, list):
                for w in warnings:
                    msg = w.get("message") if isinstance(w, dict) else str(w)
                    obs_parts.append(f"[Note: {msg}]")
            else:
                obs_parts.append(f"[Note: {warnings}]")
        if decision.warning:
            obs_parts.append(f"[Warning: {decision.warning}]")

        final_obs = "\n".join(obs_parts)
        logger.debug(
            f"[SkillToolAdapter.execute] EXIT: tool={tool_name}, "
            f"obs_len={len(final_obs)}, obs_preview='{final_obs[:150]}'"
        )

        return final_obs, None

    async def _run_with_env_vars(
        self,
        tool_name: str,
        args: dict[str, Any],
        env_vars: dict[str, str] | None,
    ) -> Any:
        """Execute tool with ENV VAR JAIL — inject env vars into subprocess context.

        Mirrors ToolRunner.run() from the deprecated tool_bridge/ module.
        For python_repl tool, env_vars are made available via os.environ.
        """
        from tools import get_registry

        registry = get_registry()

        if not env_vars:
            return await registry.execute(tool_name, args)

        old_values: dict[str, str | None] = {}
        try:
            for key, value in env_vars.items():
                old_values[key] = os.environ.get(key)
                os.environ[key] = value

            result = await registry.execute(tool_name, args)

        finally:
            for key, old_value in old_values.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value

        return result

    def _register_result(self, cache, tool_name: str, result: str) -> None:
        """Register a successful tool result in the result cache.

        Args:
            cache: Result cache to register into.
            tool_name: Name of the tool that produced the result.
            result: JSON string result to parse and cache.
        """
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict):
                for key in ["uuid", "id", "result", "data", "value"]:
                    if key in parsed and parsed[key]:
                        cache.register(f"{tool_name}.{key}", parsed[key])

                for key, value in parsed.items():
                    if isinstance(value, list):
                        for i, item in enumerate(value[:50]):
                            if isinstance(item, dict):
                                id_k = next(
                                    (k for k in ["id", "uuid", "name"] if k in item),
                                    None,
                                )
                                if id_k:
                                    cache.register(
                                        f"{tool_name}.{key}",
                                        {"index": i, id_k: item[id_k]},
                                        list_item=True,
                                    )
            elif isinstance(parsed, list):
                for i, item in enumerate(parsed[:100]):
                    if isinstance(item, dict):
                        id_val = item.get("uuid") or item.get("id") or str(i)
                        cache.register(
                            f"{tool_name}.results",
                            {"index": i, "id": id_val},
                            list_item=True,
                        )
        except (json.JSONDecodeError, TypeError):
            pass

    async def run(self, tool_name: str, args: dict[str, Any]) -> tuple[Any, dict[str, Any] | None]:
        """Compatibility shim so SkillToolAdapter can be passed as `runner` to
        ToolResultProcessor.execute_with_retry(), which calls runner.run().

        execute_with_retry expects run() to return (observation_str, error_dict),
        matching the return signature of .execute().

        CRITICAL: Skips BEFORE/AFTER hooks to prevent double-execution.
        The outer SkillAgent.run() call is responsible for all hook lifecycle.
        """
        return await self.execute(tool_name, raw_args=args, tool_props={"_skip_hooks": True}, env_vars=None)
