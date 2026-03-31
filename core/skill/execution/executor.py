"""Unified Skill executor in controlled ReAct mode."""

from __future__ import annotations

import json
import platform
import re
from pathlib import Path
from typing import Any

from core.shared import PolicyManager
from core.shared.tools_facade import BUILTIN_TOOL_SCHEMAS
from core.skill.config import SkillConfig
from core.skill.execution.content_analyzer import InfoSaturationDetector
from core.skill.execution.error_recovery import StatefulErrorPatternDetector
from core.skill.execution.loop_detector import LoopDetector
from core.skill.execution.prompts import SKILL_REACT_PROMPT
from core.skill.execution.state import (
    ReActState,
    action_signature,
    infer_preferred_extension,
    state_fingerprint,
)
from core.skill.execution.tool_bridge import ToolBridge
from core.skill.schema import ErrorType, Skill, SkillExecutionOutcome
from middleware.llm import LLMClient
from utils.debug_logger import log_skill_exec
from utils.logger import get_logger

logger = get_logger(__name__)


class SkillExecutor:
    def __init__(self, config: SkillConfig, *, llm: Any = None):
        self._config = config
        self._llm = llm if llm is not None else LLMClient()
        self._policy_manager = PolicyManager()

    def _create_tool_bridge(self) -> ToolBridge:
        return ToolBridge(self._config, policy_manager=self._policy_manager)

    async def execute(
        self,
        skill: Skill,
        query: str,
        params: dict[str, Any] | None = None,
        run_dir: Path | None = None,
        session_id: str | None = None,
        on_step: Any | None = None,
    ) -> tuple[SkillExecutionOutcome, str]:
        """Execute a skill with optional real-time step callback.

        Args:
            skill: The skill to execute
            query: User query/request
            params: Optional parameters
            run_dir: Working directory
            session_id: Session identifier
            on_step: Optional callback function(step_number, tool_name, status, signal, summary)
                    Called after each tool execution for real-time UI updates.
        """
        _ = session_id
        log_skill_exec(skill.name, query, phase="start")
        generated_code = ""

        state = ReActState(
            query=query,
            params=params,
            max_turns=30,
            preferred_core_extension=infer_preferred_extension(query, params),
        )

        self._step_counter = 0  # Track step numbers for callback

        # Track action history for error pattern detection
        action_history: list[dict] = []

        # Generic loop detector
        loop_detector = LoopDetector(
            max_observation_chain=6,
            min_effect_ratio=0.15,
            window_size=10,
        )

        # Information saturation detector
        saturation_detector = InfoSaturationDetector(
            similarity_threshold=0.6,
            entity_overlap_threshold=0.7,
            min_results_for_analysis=3,
        )

        workspace_root = (run_dir or self._config.workspace_dir).resolve()
        tool_bridge = self._create_tool_bridge()

        try:
            for turn in range(state.max_turns):
                state.turn_count = turn + 1
                messages = self._build_messages(skill, state, workspace_root)
                filtered_tools = self._filter_tools(skill.allowed_tools)

                response = await self._llm.async_chat(
                    messages=messages,
                    tools=filtered_tools,
                    tool_choice="auto",
                )

                state.messages.append(self._message_from_response(response))

                if not response.has_tool_calls:
                    if not self._goal_met(state, workspace_root):
                        state.messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "[System] VERIFICATION_FAILED: You claimed completion, "
                                    "but required artifact/observable progress is not verified in physical workspace. "
                                    "Re-check files and finish with concrete evidence."
                                ),
                            }
                        )
                        continue
                    return self._build_success_outcome(
                        skill, response.text, state, generated_code
                    )

                state.tool_calls_count += len(response.tool_calls)

                for tool_call in response.tool_calls:
                    tool_name, args, tool_call_id = self._extract_tool_call_parts(
                        tool_call
                    )
                    sig = action_signature(tool_name, args)

                    if sig == state.last_action_signature:
                        state.repeated_action_count += 1
                    else:
                        state.repeated_action_count = 0
                    state.last_action_signature = sig

                    if state.repeated_action_count > state.max_repeated_actions:
                        return self._build_failure_outcome(
                            skill,
                            "Stopped due to repeated identical tool calls",
                            state,
                            generated_code,
                        )

                    # SCRATCHPAD: Intercept update_scratchpad calls
                    if tool_name == "update_scratchpad":
                        content = args.get("content", "")
                        state.update_scratchpad(content)
                        state.messages.append(
                            {
                                "role": "tool",
                                "content": f"Scratchpad updated. Current notes:\n{state.scratchpad}",
                                "tool_call_id": tool_call_id,
                            }
                        )
                        continue

                    observation = await self._execute_tool_with_observation(
                        tool_bridge=tool_bridge,
                        tool_name=tool_name,
                        tool_call_id=tool_call_id,
                        normalized_args=args,
                        skill=skill,
                        workspace_root=workspace_root,
                        state=state,
                    )

                    if observation.get("generated_code"):
                        generated_code = observation["generated_code"]

                    state.update_from_observation(observation)

                    # Increment step counter and call real-time callback if provided
                    self._step_counter += 1
                    if on_step and callable(on_step):
                        try:
                            await on_step(
                                step_number=self._step_counter,
                                tool_name=tool_name,
                                status=observation.get("exec_status", "unknown"),
                                signal=observation.get("task_signal", "none"),
                                summary=observation.get("summary", "")[:200],
                            )
                        except Exception:
                            # Callback errors should not break execution
                            pass

                    task_signal = str(observation.get("task_signal") or "none")
                    if task_signal in {"strong", "medium"}:
                        state.no_progress_count = 0
                    else:
                        state.no_progress_count += 1

                    fp = state_fingerprint(observation)
                    if fp == state.last_state_fingerprint:
                        state.repeated_state_fingerprint_count += 1
                    else:
                        state.repeated_state_fingerprint_count = 0
                    state.last_state_fingerprint = fp

                    state.messages.append(
                        {
                            "role": "tool",
                            "content": json.dumps(observation, ensure_ascii=False),
                            "tool_call_id": observation["tool_call_id"],
                        }
                    )

                    # Track action for error pattern detection
                    action_history.append({"tool": tool_name, "arguments": args})

                    # Track for loop detection
                    tool_category = self._tool_category(tool_name)
                    new_entities = len(
                        observation.get("state_delta", {}).get("result_entities", [])
                    )
                    created = len(
                        observation.get("state_delta", {}).get("created_files", [])
                    )
                    updated = len(
                        observation.get("state_delta", {}).get("updated_files", [])
                    )

                    loop_detector.record(
                        tool_name=tool_name,
                        category=tool_category,
                        turn=state.turn_count,
                        new_entities=new_entities,
                        created_artifacts=created + updated,
                    )

                    # Check for execution loops
                    loop_info = loop_detector.detect()
                    if loop_info:
                        state.messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"[System] ⚠️ LOOP_DETECTED [{loop_info['type']}]:\n"
                                    f"{loop_info['message']}\n\n"
                                    f"Severity: {loop_info['severity']}"
                                ),
                            }
                        )
                        state.update_scratchpad(
                            f"[LOOP] {loop_info['type']}: {loop_info['message'][:80]}..."
                        )

                    # Check for information saturation (for search/fetch tools)
                    if tool_name in {"search_web", "fetch_webpage"}:
                        query = (
                            args.get("query", "")
                            if tool_name == "search_web"
                            else args.get("url", "")
                        )
                        summary = observation.get("summary", "")

                        saturation_detector.record(
                            tool_name=tool_name,
                            query=query,
                            content=summary,
                            turn=state.turn_count,
                        )

                        saturation_info = saturation_detector.check_saturation()
                        if saturation_info:
                            stats = saturation_detector.get_stats()
                            state.messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        f"[System] ⚠️ INFO_SATURATION [{saturation_info['type']}]:\n"
                                        f"{saturation_info['message']}\n\n"
                                        f"Stats: {stats['total_searches']} searches, "
                                        f"{stats['unique_entities']} unique entities, "
                                        f"{stats['entity_reuse_rate']:.0%} reuse rate\n"
                                        f"Recommendation: {saturation_info['recommendation']}"
                                    ),
                                }
                            )
                            state.update_scratchpad(
                                f"[SATURATION] {saturation_info['type']}: "
                                f"Information collection complete. Proceed to creation."
                            )

                    # ERROR RECOVERY: Detect error patterns and inject hints
                    raw_error = str(observation.get("raw", {}).get("error", ""))
                    if raw_error:
                        # Record error for pattern detection
                        state.record_error(
                            error=raw_error,
                            tool_name=tool_name,
                            hint_injected=False,
                        )

                        # Check if we should inject recovery hint
                        if (
                            state.repeated_error_count >= 1
                            and state.should_inject_recovery_hint()
                        ):
                            recovery_hints = StatefulErrorPatternDetector.analyze(
                                state.error_history,
                                action_history=action_history,
                            )

                            for hint in recovery_hints:
                                state.messages.append(
                                    {
                                        "role": "user",
                                        "content": (
                                            f"[System] ERROR_RECOVERY_HINT [{hint['pattern']}]:\n"
                                            f"{hint['hint']}\n\n"
                                            f"Severity: {hint['severity']}, "
                                            f"Repeated: {hint['match_count']} times"
                                        ),
                                    }
                                )
                                state.mark_recovery_hint_injected()
                                # Also add to scratchpad for persistence
                                state.update_scratchpad(
                                    f"[RECOVERY] {hint['pattern']}: {hint['hint'][:100]}..."
                                )

                    if state.no_progress_count > state.max_no_progress:
                        # TOPOLOGY LIGHT: Before giving up, inject recon hint
                        if state.no_progress_count == state.max_no_progress + 1:
                            state.messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "[System] PROGRESS_STALLED: No effective progress detected. "
                                        "Before continuing, run workspace reconnaissance:\n"
                                        "1. Call list_dir to check current file state\n"
                                        "2. If editing code, read_file to verify imports/dependencies\n"
                                        "3. Then retry with corrected approach."
                                    ),
                                }
                            )
                            state.no_progress_count = 0  # Give one more chance
                            continue

                        return self._build_failure_outcome(
                            skill,
                            "Stopped due to no effective task progress across tool calls",
                            state,
                            generated_code,
                        )

                    if (
                        state.repeated_state_fingerprint_count
                        > state.max_repeated_state_fingerprint
                    ):
                        return self._build_failure_outcome(
                            skill,
                            "Stopped due to repeated equivalent execution states",
                            state,
                            generated_code,
                        )

            # YIELD & RESUME: Instead of hard failure, return partial outcome
            # if meaningful progress was made
            if self._goal_met(state, workspace_root):
                logger.info(
                    "[ReAct] Max turns reached but goal met — returning success"
                )
                return self._build_success_outcome(
                    skill,
                    f"Task completed at turn limit ({state.max_turns}). "
                    f"Primary artifact: {state.get_primary_artifact() or 'N/A'}",
                    state,
                    generated_code,
                )

            return self._build_partial_outcome(
                skill,
                state,
                generated_code,
            )
        finally:
            log_skill_exec(skill.name, query, phase="end")

    def _build_messages(
        self, skill: Skill, state: ReActState, workspace_root: Path
    ) -> list[dict[str, Any]]:
        turn_warning = ""
        remaining = state.max_turns - state.turn_count
        if remaining <= 0:
            # YIELD: Final turn - must wrap up
            turn_warning = (
                "- **SYSTEM ALERT: This is your FINAL turn.**\n"
                "- If the task is COMPLETE: output a final summary with key files.\n"
                "- If the task is INCOMPLETE: output what is done and what remains, "
                "so execution can resume later."
            )
        elif remaining <= 2:
            turn_warning = (
                f"- Warning: Only {remaining} turn(s) remaining. "
                "Prioritize stabilizing current output and finalizing."
            )

        system_prompt = SKILL_REACT_PROMPT.format(
            skill_name=skill.name,
            description=skill.description or "",
            skill_source_dir=skill.source_dir or "<none>",
            existing_scripts=self._list_existing_scripts(skill),
            skill_content=self._get_skill_content(skill),
            workspace_root=str(workspace_root),
            progress_projection=state.build_progress_projection(),
            physical_world_fact=self._get_real_file_tree_limited(workspace_root),
            turn_warning=turn_warning,
            query=state.query,
            params=json.dumps(state.params or {}, ensure_ascii=False, indent=2),
            platform_info=self._platform_info(),
        )
        history = self._get_optimized_history(state.messages)
        msgs: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        msgs.append({"role": "user", "content": state.query})
        if history:
            msgs.extend(history)

        return msgs

    @staticmethod
    def _platform_info() -> str:
        system = platform.system() or "Unknown"
        release = platform.release() or ""
        machine = platform.machine() or ""
        return " ".join(part for part in [system, release, machine] if part).strip()

    @staticmethod
    def _get_skill_content(skill: Skill) -> str:
        if skill.source_dir:
            p = Path(skill.source_dir) / "SKILL.md"
            if p.exists():
                return p.read_text(encoding="utf-8")
        return skill.content or ""

    @staticmethod
    def _list_existing_scripts(skill: Skill) -> str:
        if not skill.source_dir:
            return "- <none>"
        root = Path(skill.source_dir)
        if not root.exists() or not root.is_dir():
            return "- <none>"

        exts = {".py", ".sh", ".js", ".ts", ".rb", ".pl"}
        scripts: list[str] = []
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            if p.name == "SKILL.md":
                continue
            if p.suffix.lower() in exts:
                try:
                    scripts.append(str(p.relative_to(root)))
                except Exception:
                    scripts.append(p.name)

        if not scripts:
            return "- <none>"

        return "\n".join(f"- {s}" for s in scripts[:30])

    @staticmethod
    def _get_real_file_tree_limited(workspace_root: Path, limit: int = 40) -> str:
        items: list[str] = []
        try:
            for p in sorted(workspace_root.rglob("*")):
                if len(items) >= limit:
                    break
                if not p.is_file():
                    continue
                try:
                    rel = p.relative_to(workspace_root)
                except Exception:
                    rel = p.name
                items.append(str(rel))
        except Exception:
            return "REAL_EXISTING_FILES: <unavailable>"

        if not items:
            return "REAL_EXISTING_FILES: []"
        suffix = " ..." if len(items) >= limit else ""
        return "REAL_EXISTING_FILES: [" + ", ".join(items) + "]" + suffix

    @staticmethod
    def _get_optimized_history(
        messages: list[dict[str, Any]], keep_recent: int = 8
    ) -> list[dict[str, Any]]:
        if len(messages) <= keep_recent:
            return messages

        head = messages[:-keep_recent]
        tail = messages[-keep_recent:]
        compacted: list[dict[str, Any]] = []

        for msg in head:
            role = msg.get("role")
            if role == "assistant":
                tc = msg.get("tool_calls")
                if tc:
                    compacted.append(
                        {"role": "assistant", "content": "", "tool_calls": tc}
                    )
                continue
            if role == "tool":
                content = str(msg.get("content", ""))
                tc_id = msg.get("tool_call_id", "")
                compacted.append(
                    {"role": "tool", "content": content[:220], "tool_call_id": tc_id}
                )
                continue

        compacted.extend(tail)
        return compacted

    # SCRATCHPAD: Tool schema injected into every tool list
    _SCRATCHPAD_TOOL_SCHEMA: dict = {
        "type": "function",
        "function": {
            "name": "update_scratchpad",
            "description": (
                "Save important notes, constraints, or sub-goals to a persistent scratchpad. "
                "Use this when you need to remember critical information across turns "
                "(e.g., user requirements, chapter titles, key parameters). "
                "The scratchpad is never compressed and always visible in every turn."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The note to save. Be concise but complete.",
                    }
                },
                "required": ["content"],
            },
        },
    }

    @staticmethod
    def _filter_tools(allowed_tools: list[str] | None) -> list[dict]:
        """根据 allowed_tools 过滤工具

        语义：
        - None: 允许所有内置工具（向后兼容）
        - []: 允许所有内置工具（向后兼容，但不推荐）
        - ["tool1", "tool2"]: 只允许列表中的工具，无效工具名会记录警告

        始终追加 update_scratchpad 虚拟工具。
        """
        if allowed_tools is None:
            base = list(BUILTIN_TOOL_SCHEMAS)
        elif not allowed_tools:  # 空列表
            logger.warning(
                "allowed_tools is empty list, using all builtin tools. "
                "Consider using None to explicitly indicate 'all tools allowed'."
            )
            base = list(BUILTIN_TOOL_SCHEMAS)
        else:
            # 构建可用工具映射
            builtin_map = {t["function"]["name"]: t for t in BUILTIN_TOOL_SCHEMAS}

            base = []
            unknown_tools = []

            for tool_name in allowed_tools:
                if tool_name in builtin_map:
                    base.append(builtin_map[tool_name])
                else:
                    unknown_tools.append(tool_name)

            # 记录无效工具名
            if unknown_tools:
                logger.warning(
                    "Unknown tools in allowed_tools: {}. Available builtin tools: {}.",
                    unknown_tools,
                    list(builtin_map.keys()),
                )

            # 如果没有匹配到任何工具，回退到全部工具
            if not base:
                logger.warning(
                    "No valid tools found in allowed_tools {}, using all builtin tools",
                    allowed_tools,
                )
                base = list(BUILTIN_TOOL_SCHEMAS)

        # SCRATCHPAD: Always inject update_scratchpad tool
        base.append(SkillExecutor._SCRATCHPAD_TOOL_SCHEMA)
        return base

    async def _execute_tool_with_observation(
        self,
        *,
        tool_bridge: ToolBridge,
        tool_name: str,
        tool_call_id: str,
        normalized_args: dict[str, Any],
        skill: Skill,
        workspace_root: Path,
        state: ReActState,
    ) -> dict[str, Any]:
        arguments = self._inject_artifact_path_if_needed(normalized_args, state)

        # ENV VAR JAIL: Build environment variables for tool execution
        env_vars = self._build_env_vars(state, workspace_root)

        safe_call = {
            "function": {
                "name": tool_name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            }
        }

        try:
            outcome, _ = await tool_bridge.execute(
                skill=skill,
                tool_calls=[safe_call],
                workspace_dir=workspace_root,
                env_vars=env_vars,
            )
        except Exception as e:
            return {
                "tool_call_id": tool_call_id,
                "status": "error",
                "summary": f"PHYSICAL_EXECUTION_ERROR: {e}",
                "state_delta": {},
                "raw": str(e),
                "generated_code": arguments.get("code"),
            }

        observation = self._build_observation(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            outcome=outcome,
            state=state,
        )

        # ENV VAR JAIL: Append warning if model hardcoded paths
        env_var_warning = arguments.get("_env_var_warning")
        if env_var_warning:
            observation["summary"] = (
                f"{observation.get('summary', '')}\n\n{env_var_warning}"
            )

        self._apply_artifact_lock(observation, state)
        return observation

    def _build_observation(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        outcome: SkillExecutionOutcome,
        state: ReActState,
    ) -> dict[str, Any]:
        artifacts = list(outcome.artifacts or [])
        created_files: list[str] = []
        updated_files: list[str] = []

        for a in artifacts:
            if isinstance(a, str):
                created_files.append(a)
            elif isinstance(a, dict):
                p = a.get("path") or a.get("file_path")
                if p:
                    created_files.append(str(p))

        operation_results = list(outcome.operation_results or [])
        op_signals = self._extract_operation_signals(operation_results)
        created_files.extend(op_signals["created_files"])
        updated_files.extend(op_signals["updated_files"])

        if tool_name in {"write_file", "create_file"}:
            candidate = self._extract_path_from_result_text(str(outcome.result or ""))
            if candidate:
                created_files.append(candidate)
        if tool_name in {"edit_file", "str_replace", "delete_file"}:
            candidate = self._extract_path_from_result_text(str(outcome.result or ""))
            if candidate:
                updated_files.append(candidate)

        created_files = self._dedupe_nonempty(created_files)
        updated_files = self._dedupe_nonempty(updated_files)

        if created_files:
            updated_files = [p for p in updated_files if p not in created_files]

        summary_source = ""
        if operation_results:
            first = (
                operation_results[0] if isinstance(operation_results[0], dict) else {}
            )
            summary_source = str(first.get("result") or first.get("output") or "")
        if not summary_source:
            summary_source = str(outcome.result or "")

        summary = summary_source if outcome.success else f"Error: {outcome.error}"
        summary = summary[:500]

        result_entities = self._extract_result_entities(summary)
        installed_deps = self._extract_installed_deps_structured(
            operation_results=operation_results,
            result_text=summary,
            tool_name=tool_name,
        )

        urls = self._extract_urls(summary)
        new_urls = [u for u in urls if u not in state.seen_urls]
        for u in new_urls:
            state.seen_urls.add(u)

        new_entities = [e for e in result_entities if e not in state.seen_entities]
        for e in new_entities:
            state.seen_entities.add(e)

        repeated_action = state.repeated_action_count > 0
        raw_error = str(outcome.error or "").strip()
        current_error_hash = self._error_fingerprint(raw_error)
        error_changed = (
            bool(current_error_hash) and current_error_hash != state.last_error_hash
        )
        if raw_error:
            state.last_error = raw_error
            state.last_error_hash = current_error_hash

        task_signal = self._evaluate_task_signal(
            tool_name=tool_name,
            created_files=created_files,
            updated_files=updated_files,
            installed_deps=installed_deps,
            result_entities=result_entities,
            new_urls_count=len(new_urls),
            new_entities_count=len(new_entities),
            success=outcome.success,
            error=outcome.error,
            repeated_action=repeated_action,
            error_changed=error_changed,
            assistant_text_hint=summary,
        )

        return {
            "tool_call_id": tool_call_id,
            "tool": tool_name,
            "exec_status": "success" if outcome.success else "error",
            "status": "success" if outcome.success else "error",
            "task_signal": task_signal,
            "summary": summary,
            "state_delta": {
                "created_files": created_files,
                "updated_files": updated_files,
                "installed_deps": installed_deps,
                "result_entities": result_entities,
                "new_urls": new_urls,
                "new_entities": new_entities,
                "operation_count": len(operation_results),
                "artifacts_count": len(artifacts),
            },
            "raw": {
                "error": outcome.error,
                "error_type": outcome.error_type.value if outcome.error_type else None,
                "operation_count": len(operation_results),
                "error_changed": error_changed,
            },
            "generated_code": "",
            "artifacts": artifacts,
        }

    @staticmethod
    def _normalize_arguments(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    @staticmethod
    def _inject_artifact_path_if_needed(
        arguments: dict[str, Any], state: ReActState
    ) -> dict[str, Any]:
        """ENV VAR JAIL: 不再修改代码，而是通过环境变量传递产物路径。

        保留此方法用于：
        1. 向后兼容检查（验证模型是否使用了环境变量）
        2. 返回警告信息（如果模型未使用环境变量）

        实际的路径传递通过环境变量 PRIMARY_ARTIFACT_PATH 完成。
        """
        if "code" not in arguments or not isinstance(arguments["code"], str):
            return arguments

        target_path = state.get_primary_artifact()
        if not target_path:
            return arguments

        code = arguments["code"]

        # ENV VAR JAIL: 检查模型是否使用了环境变量
        # 如果模型硬编码了路径，返回警告但不修改代码
        has_hardcoded_save = re.search(r'\.save\s*\(\s*["\']', code)
        uses_env_var = "os.environ.get" in code or "os.getenv" in code

        if has_hardcoded_save and not uses_env_var:
            arguments["_env_var_warning"] = (
                f"WARNING: You are hardcoding file paths. "
                f"Use os.environ.get('PRIMARY_ARTIFACT_PATH') instead. "
                f"Expected path: {target_path}"
            )

        return arguments

    @staticmethod
    def _build_env_vars(state: ReActState, workspace_root: Path) -> dict[str, str]:
        """ENV VAR JAIL: Build environment variables for tool execution.

        Injects into subprocess environment:
        - PRIMARY_ARTIFACT_PATH: Locked primary artifact path
        - WORKSPACE_ROOT: Workspace root directory
        """
        env_vars: dict[str, str] = {
            "WORKSPACE_ROOT": str(workspace_root),
        }

        primary = state.get_primary_artifact()
        if primary:
            env_vars["PRIMARY_ARTIFACT_PATH"] = primary

        return env_vars

        target_path = state.get_primary_artifact()
        if not target_path:
            return arguments

        code = arguments["code"]
        code = re.sub(
            r"prs\.save\([\'\"][^\'\"]+[\'\"]\)", f'prs.save("{target_path}")', code
        )
        if "Presentation()" in code:
            code = re.sub(
                r"prs\s*=\s*Presentation\(\s*\)",
                f'prs = Presentation("{target_path}")',
                code,
            )
        arguments["code"] = code
        return arguments

    @staticmethod
    def _has_state_signal(state_delta: dict[str, Any]) -> bool:
        if not state_delta:
            return False
        if state_delta.get("created_files"):
            return True
        if state_delta.get("updated_files"):
            return True
        if state_delta.get("installed_deps"):
            return True
        if int(state_delta.get("operation_count") or 0) > 0:
            return True
        if int(state_delta.get("artifacts_count") or 0) > 0:
            return True
        return False

    @staticmethod
    def _dedupe_nonempty(paths: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for p in paths:
            p_norm = str(p or "").strip()
            if not p_norm or p_norm in seen:
                continue
            seen.add(p_norm)
            out.append(p_norm)
        return out

    @staticmethod
    def _extract_path_from_result_text(text: str) -> str | None:
        if not text:
            return None
        patterns = [
            r"(?:at|to|path[:\s])\s+([\w\-/\\.@]+\.[a-zA-Z0-9]+)",
            r"([\w\-/\\.@]+\.[a-zA-Z0-9]+)",
        ]
        for pattern in patterns:
            m = re.search(pattern, text)
            if m:
                return m.group(1)
        return None

    @staticmethod
    def _extract_installed_deps_structured(
        *,
        operation_results: list[dict[str, Any]],
        result_text: str,
        tool_name: str,
    ) -> list[str]:
        _ = result_text
        _ = tool_name
        deps: list[str] = []

        for op in operation_results:
            if not isinstance(op, dict):
                continue

            for key in ("installed_deps", "deps", "dependencies"):
                value = op.get(key)
                if isinstance(value, list):
                    for dep in value:
                        if isinstance(dep, str) and re.match(
                            r"^[A-Za-z0-9_.\-]+$", dep
                        ):
                            deps.append(dep)

            maybe_result = op.get("result")
            if isinstance(maybe_result, dict):
                for key in ("installed_deps", "deps", "dependencies"):
                    value = maybe_result.get(key)
                    if isinstance(value, list):
                        for dep in value:
                            if isinstance(dep, str) and re.match(
                                r"^[A-Za-z0-9_.\-]+$", dep
                            ):
                                deps.append(dep)

        seen: set[str] = set()
        clean: list[str] = []
        for dep in deps:
            dep = dep.strip()
            if not dep or dep in seen:
                continue
            seen.add(dep)
            clean.append(dep)
        return clean[:15]

    @staticmethod
    def _extract_result_entities(summary: str) -> list[str]:
        text = (summary or "").strip()
        if not text:
            return []
        entities = re.findall(r"[A-Za-z0-9_\-/]{4,}", text)
        seen: set[str] = set()
        out: list[str] = []
        for e in entities:
            if e in seen:
                continue
            seen.add(e)
            out.append(e)
        return out[:8]

    @staticmethod
    def _tool_category(tool_name: str) -> str:
        if tool_name in {
            "search_web",
            "fetch_webpage",
            "read_file",
            "list_dir",
            "grep",
        }:
            return "observation"
        if tool_name in {"file_create", "edit_file_by_lines", "bash", "python_repl"}:
            return "effect"
        return "mixed"

    @staticmethod
    def _extract_urls(text: str) -> list[str]:
        if not text:
            return []
        urls = re.findall(r"https?://[^\s\]\)\"']+", text)
        dedup: list[str] = []
        seen: set[str] = set()
        for u in urls:
            if u in seen:
                continue
            seen.add(u)
            dedup.append(u)
        return dedup[:20]

    @staticmethod
    def _is_conclusion_like(text: str) -> bool:
        t = (text or "").lower()
        hints = [
            "综上",
            "总结",
            "结论",
            "final",
            "in summary",
            "overall",
            "完成",
            "done",
        ]
        return any(h in t for h in hints)

    @staticmethod
    def _error_fingerprint(error_text: str) -> str | None:
        text = (error_text or "").strip()
        if not text:
            return None
        normalized = text.lower()
        normalized = re.sub(r"line\s+\d+", "line <n>", normalized)
        normalized = re.sub(r"/[^\s:\"']+", "<path>", normalized)
        normalized = re.sub(r"\b\d+\b", "<n>", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return action_signature("error", normalized)

    @staticmethod
    def _evaluate_task_signal(
        *,
        tool_name: str,
        created_files: list[str],
        updated_files: list[str],
        installed_deps: list[str],
        result_entities: list[str],
        new_urls_count: int,
        new_entities_count: int,
        success: bool,
        error: str | None,
        repeated_action: bool,
        error_changed: bool,
        assistant_text_hint: str = "",
    ) -> str:
        if repeated_action:
            return "none"
        if not success:
            return "none"

        category = SkillExecutor._tool_category(tool_name)

        if created_files or updated_files:
            return "strong"
        if installed_deps:
            return "medium"
        if error_changed:
            return "medium"
        if SkillExecutor._is_conclusion_like(assistant_text_hint):
            return "medium"

        if category == "observation":
            if new_urls_count >= 2:
                return "strong"
            if new_urls_count >= 1 or new_entities_count >= 2:
                return "medium"
            if result_entities and not error:
                return "weak"
            return "none"

        if result_entities and not error:
            return "weak"
        return "none"

    @staticmethod
    def _extract_operation_signals(
        operation_results: list[dict[str, Any]],
    ) -> dict[str, list[str]]:
        created: list[str] = []
        updated: list[str] = []

        for op in operation_results:
            if not isinstance(op, dict):
                continue
            op_name = str(op.get("operation") or op.get("tool") or "").lower()
            payload = str(op.get("result") or op.get("output") or "")
            path = (
                op.get("path")
                or op.get("file_path")
                or op.get("target_path")
                or SkillExecutor._extract_path_from_result_text(payload)
            )
            if not path:
                continue
            p = str(path)

            if any(x in op_name for x in ["create", "write", "save"]):
                created.append(p)
            elif any(
                x in op_name for x in ["edit", "replace", "update", "append", "delete"]
            ):
                updated.append(p)

        return {"created_files": created, "updated_files": updated}

    @staticmethod
    def _apply_artifact_lock(observation: dict[str, Any], state: ReActState) -> None:
        artifacts = observation.get("artifacts") or []
        for artifact in artifacts:
            if isinstance(artifact, str):
                state.lock_artifact(artifact)
            elif isinstance(artifact, dict):
                p = artifact.get("path") or artifact.get("file_path")
                if p:
                    state.lock_artifact(str(p))

    @staticmethod
    def _extract_tool_call_parts(tool_call: Any) -> tuple[str, dict[str, Any], str]:
        if hasattr(tool_call, "function") and tool_call.function is not None:
            name = str(getattr(tool_call.function, "name", "") or "")
            raw_args = getattr(tool_call.function, "arguments", {})
            call_id = str(getattr(tool_call, "id", "tool_call") or "tool_call")
            args = SkillExecutor._normalize_arguments(raw_args)
            return name, args, call_id

        if isinstance(tool_call, dict):
            fn = tool_call.get("function", {}) or {}
            name = str(fn.get("name", "") or tool_call.get("name", ""))
            raw_args = fn.get("arguments", tool_call.get("arguments", {}))
            call_id = str(tool_call.get("id", "tool_call"))
            args = SkillExecutor._normalize_arguments(raw_args)
            return name, args, call_id

        # Fallback for atypical SDK shapes
        name = str(getattr(tool_call, "name", "") or "")
        raw_args = getattr(tool_call, "arguments", {})
        call_id = str(getattr(tool_call, "id", "tool_call") or "tool_call")
        args = SkillExecutor._normalize_arguments(raw_args)
        return name, args, call_id

    @staticmethod
    def _message_from_response(response: Any) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": response.text or ""}
        if response.has_tool_calls:
            tool_calls_payload = []
            for tc in response.tool_calls:
                name, args, call_id = SkillExecutor._extract_tool_call_parts(tc)
                tool_calls_payload.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                    }
                )
            msg["tool_calls"] = tool_calls_payload
        return msg

    @staticmethod
    def _goal_met(state: ReActState, workspace_root: Path) -> bool:
        primary = state.get_primary_artifact()
        if primary:
            try:
                p = Path(primary)
                if p.exists() and p.is_file():
                    return True
            except Exception:
                pass
        if state.created_files or state.updated_files:
            return True
        try:
            return any(workspace_root.rglob("*"))
        except Exception:
            return False

    @staticmethod
    def _apply_progress_policy(
        *,
        state: ReActState,
        observation: dict[str, Any],
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        signal = str(observation.get("task_signal") or "none")
        category = SkillExecutor._tool_category(tool_name)

        if category == "observation":
            threshold = 4
        elif category == "effect":
            threshold = 2
        else:
            threshold = 3

        state.max_no_progress = threshold

        if signal in {"strong", "medium"}:
            state.no_progress_count = 0
        elif signal == "weak":
            state.no_progress_count += 1
        else:
            state.no_progress_count += 2

        if category == "observation":
            # reward information gain with extra horizon, bounded
            if signal in {"strong", "medium"} and state.max_turns < 12:
                state.max_turns += 1

        # same query repetition should be treated as higher risk stalling
        if tool_name in {"search_web", "fetch_webpage"}:
            query = str(arguments.get("query") or arguments.get("url") or "").strip()
            if query and state.last_action_signature:
                # repeated_action_count is already maintained by action signature check
                if state.repeated_action_count > 0:
                    state.no_progress_count += 1

    @staticmethod
    def _build_success_outcome(
        skill: Skill,
        result_text: str,
        state: ReActState,
        generated_code: str,
    ) -> tuple[SkillExecutionOutcome, str]:
        projection = state.build_outcome_projection()
        result_payload = {
            "final_response": result_text
            or f"Task completed in {state.turn_count} turns.",
            "execution_summary": projection,
        }
        return SkillExecutionOutcome(
            success=True, result=result_payload, skill_name=skill.name
        ), generated_code

    @staticmethod
    def _build_partial_outcome(
        skill: Skill,
        state: ReActState,
        generated_code: str,
    ) -> tuple[SkillExecutionOutcome, str]:
        """YIELD & RESUME: Build a partial outcome when max turns exceeded.

        Instead of a hard failure, returns a structured partial result
        with resume hints so upper-layer Router can continue later.
        """
        projection = state.build_outcome_projection()

        # Build resume hint from scratchpad + progress
        resume_hint = {
            "status": "partial",
            "completed": {
                "created_files": state.created_files,
                "updated_files": state.updated_files,
                "installed_deps": state.installed_deps,
                "primary_artifact": state.get_primary_artifact(),
            },
            "remaining": state.scratchpad or "No scratchpad notes available.",
            "turn_count": state.turn_count,
        }

        return (
            SkillExecutionOutcome(
                success=False,
                result={
                    "final_response": (
                        f"Task partially completed after {state.turn_count} turns. "
                        f"Created files: {[Path(p).name for p in state.created_files]}. "
                        f"Primary artifact: {state.get_primary_artifact() or 'N/A'}."
                    ),
                    "execution_summary": projection,
                    "resume_hint": resume_hint,
                },
                error=f"Execution exceeded maximum turns ({state.max_turns})",
                error_type=ErrorType.EXECUTION_ERROR,
                error_detail={
                    "turn_count": state.turn_count,
                    "tool_calls": state.tool_calls_count,
                    "last_error": state.last_error,
                    "artifacts": state.all_artifacts,
                    "observation_stats": projection.get("observation_stats", {}),
                    "resume_hint": resume_hint,
                },
                skill_name=skill.name,
            ),
            generated_code,
        )

    @staticmethod
    def _build_failure_outcome(
        skill: Skill,
        error: str,
        state: ReActState,
        generated_code: str,
    ) -> tuple[SkillExecutionOutcome, str]:
        projection = state.build_outcome_projection()
        return (
            SkillExecutionOutcome(
                success=False,
                result={
                    "final_response": "",
                    "execution_summary": projection,
                },
                error=error,
                error_type=ErrorType.EXECUTION_ERROR,
                error_detail={
                    "turn_count": state.turn_count,
                    "tool_calls": state.tool_calls_count,
                    "last_error": state.last_error,
                    "artifacts": state.all_artifacts,
                    "observation_stats": projection.get("observation_stats", {}),
                },
                skill_name=skill.name,
            ),
            generated_code,
        )
