# SPDX-License-Identifier: Apache-2.0
"""SkillAgent — ReAct agent for skill execution.

Delegates atomic tool execution to SkillToolAdapter, keeping the agent
focused on LLM orchestration and state management.

Detection architecture (long-term fix):
  ALL loop/stall detection is routed through the Hook system.
  No hard-coded detection in agent.py (Layer 1/2/3 removed).
  The Hook system provides:
    - LoopDetector: patterns (chain, ratio, diminishing, sequence)
    - InfoSaturationDetector: web-tool result similarity
    - The Hook's recovery_action is the single exit decision point.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
from pathlib import Path
from typing import Any

from shared.schema import SkillConfig
from core.skill.execution.adapter import SkillToolAdapter
from core.skill.execution.artifact_registry import ArtifactRegistry
from core.skill.execution.error_recovery import StatefulErrorPatternDetector
from core.skill.execution.loop_detector import LoopDetector
from core.skill.execution.prompts import SKILL_REACT_PROMPT
from core.skill.execution.compaction import (
    TokenBudgetPolicy,
    SummarizerRegistry,
    get_default_registry,
    make_budget_policy,
)
from core.skill.execution.state import (
    ReActState,
    infer_preferred_extension,
    ContextCompactor,
)
from core.skill.schema import ErrorType, Skill, SkillExecutionOutcome
from middleware.llm import LLMClient
from utils.debug_logger import log_skill_exec
from utils.log_config import log_preview, log_preview_long, log_preview_medium
from utils.logger import get_logger

logger = get_logger(__name__)


class SkillAgent:
    """ReAct agent that executes skills using atomic tools.

    The agent manages the LLM orchestration loop while delegating
    actual tool execution to SkillToolAdapter.
    """

    def __init__(self, config: SkillConfig, *, llm=None, policy_manager=None):
        """Initialize the skill agent.

        Args:
            config: SkillConfig for skill execution.
            llm: Optional LLM client. Defaults to LLMClient().
            policy_manager: Deprecated. Policy checks are now via global HookExecutor (PolicyGateHook).
                          Kept for backward compatibility — will be ignored.
        """
        self._config = config
        self._llm = llm if llm is not None else LLMClient()
        self._context = None  # 由 adapter.set_context() 设置

    async def run(
        self,
        skill: Skill,
        query: str,
        params: dict[str, Any] | None,
        run_dir: Path | None,
        session_id: str,
        on_step: Any | None,
        max_turns: int = 30,
    ) -> tuple[SkillExecutionOutcome, str]:
        """Run the ReAct loop for skill execution.

        Args:
            skill: The skill to execute.
            query: User query/request.
            params: Optional parameters.
            run_dir: Working directory.
            session_id: Session identifier.
            on_step: Optional callback for step updates.
            max_turns: Maximum number of turns (default 30).

        Returns:
            Tuple of (SkillExecutionOutcome, generated_code).
        """
        log_skill_exec(skill.name, query, phase="start")
        generated_code = ""

        budget = self._context_budget()
        policy = make_budget_policy(
            context_window=getattr(self._llm, "context_window", 128000),
            max_output_tokens=getattr(self._llm, "max_tokens", 4096),
        )

        state = ReActState(
            query=query,
            params=params,
            max_turns=max_turns,
            preferred_core_extension=infer_preferred_extension(query, params),
            compact_threshold_tokens=policy.compact_threshold,
            _compactor=ContextCompactor(
                threshold=policy.compact_threshold,
                llm=self._llm,
            ),
        )
        state._budget_policy = policy
        state._summarizer_registry = get_default_registry()

        workspace_root = (run_dir or self._config.workspace_dir).resolve()

        logger.debug(
            f"[SkillAgent.run] INITIAL STATE: "
            f"skill={skill.name}, "
            f"query='{log_preview_long(query)}', "
            f"workspace_root={workspace_root}, "
            f"run_dir={run_dir}, "
            f"allowed_tools={skill.allowed_tools}, "
            f"execution_mode={skill.execution_mode}, "
            f"max_turns={max_turns}"
        )

        # 使用全局 HookExecutor（单例，按事件类型自动过滤）
        # PolicyGateHook, PathPolicyHook 在系统初始化时已注册到全局
        # FileChangeHook（需要 workspace_root + artifact_registry）在下方动态注册
        from core.skill.execution.hooks import global_skill_agent_hook_executor, SandboxAuditHook, FileChangeHook
        from shared.hooks.types import HookEvent, HookPayload

        hook_executor = global_skill_agent_hook_executor()

        # 注册 SandboxAuditHook（后置文件创建审计）
        hook_executor.register(HookEvent.BEFORE_TOOL_EXEC, SandboxAuditHook(workspace_root=workspace_root))
        hook_executor.register(HookEvent.AFTER_TOOL_EXEC, SandboxAuditHook(workspace_root=workspace_root))

        # 注册 FileChangeHook（文件变化检测 + 生命周期管理）
        file_change_hook = FileChangeHook(workspace_root=workspace_root)
        # 绑定 artifact_registry，自动注册产物
        file_change_hook.bind_artifact_registry(state.artifact_registry)
        hook_executor.register(HookEvent.BEFORE_TOOL_EXEC, file_change_hook)
        hook_executor.register(HookEvent.AFTER_TOOL_EXEC, file_change_hook)

        adapter = SkillToolAdapter(
            config=self._config,
            result_cache=state.result_cache,
            hook_executor=hook_executor,
        )
        adapter.set_context(skill, workspace_root, session_id=session_id)
        self._skill = skill  # 供 tool_props 注入 skill_deps 使用

        # ── Per-execution 层：注册状态性监督 Hook ─────────────────────────────
        # 这些 Hook 在每次 SkillAgent.run() 时创建新实例，
        # 确保 LoopDetector / InfoSaturationDetector 在每次 skill 执行时重置状态。
        # （注意：全局永久层中的 LoopSupervisionHook / SaturationSupervisionHook 已移除）
        #
        # LoopSupervisionHook: 持有 hook_executor 用于触发 ON_LOOP_DETECTED
        # SaturationSupervisionHook: 仅对 search_web/fetch_webpage 有效
        # ErrorPatternSupervisionHook: 错误模式检测（无状态，可保留全局，但放这里更清晰）
        from core.skill.execution.hooks.loop_supervision import LoopSupervisionHook
        from core.skill.execution.hooks.saturation_supervision import SaturationSupervisionHook
        from core.skill.execution.hooks.stall_supervision import StallSupervisionHook
        from core.skill.execution.hooks.error_pattern_supervision import ErrorPatternSupervisionHook

        loop_hook = LoopSupervisionHook(hook_executor=hook_executor)
        hook_executor.register(HookEvent.AFTER_TOOL_EXEC, loop_hook)

        saturation_hook = SaturationSupervisionHook()
        hook_executor.register(HookEvent.AFTER_TOOL_EXEC, saturation_hook)

        stall_hook = StallSupervisionHook()
        hook_executor.register(HookEvent.AFTER_TOOL_EXEC, stall_hook)

        error_hook = ErrorPatternSupervisionHook()
        hook_executor.register(HookEvent.AFTER_TOOL_EXEC, error_hook)

        # action_history 必须在 bind_state_context 之前声明（error_hook.bind_state_context 引用了它）
        action_history: list[dict] = []

        loop_hook.bind_state_context({
            "turn_count": lambda: state.turn_count,
            "artifact_registry": state.artifact_registry,
            "update_scratchpad": state.update_scratchpad,
            "scratchpad": lambda: state.scratchpad,
        })
        saturation_hook.bind_state_context({
            "turn_count": lambda: state.turn_count,
            "update_scratchpad": state.update_scratchpad,
        })
        stall_hook.bind_state_context({
            "turn_count": lambda: state.turn_count,
            "artifact_registry": state.artifact_registry,
            "scratchpad": lambda: state.scratchpad,
            "update_scratchpad": state.update_scratchpad,
        })
        error_hook.bind_state_context({
            "turn_count": lambda: state.turn_count,
            "error_history": state.error_history,
            "action_history": action_history,
            "record_error": state.record_error,
            "should_inject_recovery_hint": state.should_inject_recovery_hint,
            "mark_recovery_hint_injected": state.mark_recovery_hint_injected,
            "update_scratchpad": state.update_scratchpad,
        })

        step_counter = 0
        last_verification_failed: bool = False  # 防止 VERIFICATION_FAILED 震荡
        last_bash_cmd: str = ""  # 防止相同 bash 命令重复执行
        last_bash_result_hash: int = 0
        bash_repeat_count: int = 0
        last_llm_usage: Any = None  # 追踪最后 LLM 调用的 usage 信息
        last_finish_reason: str = None  # 追踪最后 LLM 调用的 finish_reason

        try:
            for turn in range(state.max_turns):
                state.turn_count = turn + 1
                messages = self._build_messages(skill, state, workspace_root)
                tool_schemas = self._get_tool_schemas(skill)

                logger.info(
                    f"[SkillAgent] turn={turn + 1}/{state.max_turns}, "
                    f"message_count={len(messages)}, "
                    f"tools={[t['function']['name'] for t in tool_schemas]}"
                )

                try:
                    response = await self._llm.async_chat(
                        messages=messages,
                        tools=tool_schemas,
                        tool_choice="auto",
                    )
                except Exception as e:
                    # 捕获 LLM 调用时的网络错误(如 ECONNRESET、超时等)
                    logger.error(
                        f"[SkillAgent] LLM call failed: {type(e).__name__}: {e}"
                    )
                    return self._build_failure_outcome(
                        skill=skill,
                        reason=f"LLM call failed: {type(e).__name__}: {str(e)}",
                        state=state,
                        generated_code=generated_code,
                        llm_usage=last_llm_usage,
                        finish_reason=last_finish_reason,
                    )

                # 追踪 LLM 调用信息用于错误诊断
                last_llm_usage = getattr(response, 'usage', None)
                last_finish_reason = getattr(response, 'finish_reason', None)

                # 记录 token 使用情况到日志
                if last_llm_usage:
                    logger.info(
                        f"[SkillAgent] LLM usage: prompt_tokens={getattr(last_llm_usage, 'prompt_tokens', '?')}, "
                        f"completion_tokens={getattr(last_llm_usage, 'completion_tokens', '?')}, "
                        f"total_tokens={getattr(last_llm_usage, 'total_tokens', '?')}"
                    )

                logger.debug(
                    f"[SkillAgent] LLM response: turn={turn + 1}, "
                    f"finish_reason={getattr(response, 'finish_reason', '?')}, "
                    f"has_tool_calls={response.has_tool_calls}, "
                    f"content_len={len(response.text) if response.text else 0}"
                )
                logger.debug(
                    f"[SkillAgent] LLM content preview (first 300 chars): "
                    f"{response.text[:300] if response.text else '(empty)'}"
                )

                if response.tool_calls:
                    logger.info(
                        f"[SkillAgent] LLM made {len(response.tool_calls)} tool call(s): "
                        f"{[tc.get('function', {}).get('name', '?') if isinstance(tc, dict) else getattr(tc, 'name', '?') for tc in response.tool_calls]}"
                    )
                    for tc in response.tool_calls:
                        tc_name = (
                            tc.get("function", {}).get("name", "?")
                            if isinstance(tc, dict)
                            else getattr(tc, "name", "?")
                        )
                        tc_args = (
                            tc.get("function", {}).get("arguments", "")
                            if isinstance(tc, dict)
                            else getattr(tc, "arguments", "")
                        )
                        logger.info(
                            f"  -> tool_call: name={tc_name}, "
                            f"args_preview='{log_preview_medium(str(tc_args))}'"
                        )

                # 统一消息总线：所有消息通过 state.context 管理
                state.context.append_assistant(
                    text=response.text or "",
                    tool_calls=response.tool_calls,
                )

                # ── 无 tool_calls 分支 ──────────────────────────────────────
                if not response.has_tool_calls:
                    state.consecutive_final_answer_count += 1
                    goal_met = self._goal_met(state, workspace_root)
                    logger.info(
                        f"[SkillAgent] No tool calls. "
                        f"consecutive_final_answer_count={state.consecutive_final_answer_count}, "
                        f"goal_met={goal_met}, "
                        f"created_files={list(state.created_files)}, "
                        f"updated_files={list(state.updated_files)}, "
                        f"primary_artifact={state.get_primary_artifact()}, "
                        f"last_tool_name={tool_name}"
                    )

                    # Phase 1: 当 goal_met=True 时，注入明确的完成信号
                    if goal_met:
                        primary = state.get_primary_artifact()
                        primary_name = Path(primary).name if primary else "unknown"
                        state.context.append_user(
                            role="user",
                            content=(
                                f"[System] GOAL_MET: Primary artifact '{primary_name}' "
                                f"has been created and verified. "
                                f"Task is complete. Do NOT make any more tool calls. Provide Final Answer now."
                            ),
                        )

                    if state.consecutive_final_answer_count > 3:
                        logger.info(
                            "[SkillAgent] STOPPED: too many consecutive Final Answer"
                        )
                        state.update_scratchpad(
                            f"[LOOP] Consecutive Final Answer x{state.consecutive_final_answer_count}"
                        )
                        return self._build_failure_outcome(
                            skill,
                            f"Stopped: {state.consecutive_final_answer_count} consecutive Final Answer responses",
                            state,
                            generated_code,
                            llm_usage=last_llm_usage,
                            finish_reason=last_finish_reason,
                        )

                    if not goal_met:
                        # 防止震荡：如果上一轮已经是 VERIFICATION_FAILED，这次直接失败
                        if last_verification_failed:
                            logger.info(
                                "[SkillAgent] VERIFICATION_FAILED again — giving up"
                            )
                            state.update_scratchpad(
                                "[LOOP] VERIFICATION_FAILED twice in a row"
                            )
                            return self._build_failure_outcome(
                                skill,
                                "VERIFICATION_FAILED: claimed completion but no progress verified",
                                state,
                                generated_code,
                                llm_usage=last_llm_usage,
                                finish_reason=last_finish_reason,
                            )
                        last_verification_failed = True
                        state.context.append_user(
                            role="user",
                            content=(
                                "[System] VERIFICATION_FAILED: 你声称任务完成但系统未检测到进展。"
                                "使用 file_create 工具将结果写入文件（如 notes.md）。"
                                "不要重复执行 ls/file 命令验证已确认存在的文件。"
                                "完成工作后调用 respond_final_answer 工具。"
                            ),
                        )
                        continue

                    last_verification_failed = False
                    logger.info("[SkillAgent] Goal met — returning success")
                    return self._build_success_outcome(
                        skill, response.text, state, generated_code
                    )

                state.tool_calls_count += len(response.tool_calls)
                state.consecutive_final_answer_count = 0
                last_verification_failed = False

                deferred_msgs: list[dict[str, Any]] = []
                early_exit_outcome: SkillExecutionOutcome | None = None

                # FIX 2: Turn 级别进度快照 — 在本 turn 处理前记录文件状态
                _turn_files_before: set[str] = set(state.created_files)
                _turn_updated_before: set[str] = set(state.updated_files)

                for tool_call in response.tool_calls:
                    tool_name, args, tool_call_id = self._extract_tool_call_parts(
                        tool_call
                    )
                    sig = f"{tool_name}:{json.dumps(args, sort_keys=True)}"

                    sig = (tool_name, json.dumps(args, sort_keys=True))
                    if sig == state.last_action_signature:
                        state.repeated_action_count += 1
                    else:
                        state.repeated_action_count = 0
                    state.last_action_signature = sig

                    # CRITICAL: 重复 action 立即退出（阈值 1）
                    if state.repeated_action_count > 1:
                        logger.info(
                            f"[SkillAgent] EARLY EXIT: repeated action={tool_name} x{state.repeated_action_count + 1}"
                        )
                        early_exit_outcome = self._build_failure_outcome(
                            skill,
                            f"Stopped: repeated identical tool call '{tool_name}'",
                            state,
                            generated_code,
                            llm_usage=last_llm_usage,
                            finish_reason=last_finish_reason,
                        )
                        state.context.append_tool_result(
                            tool_call_id=tool_call_id,
                            tool_name=tool_name,
                            content="skipped: repeated identical tool call",
                        )
                        break  # turn 内 early break


                    if tool_name == "update_scratchpad":
                        content = args.get("content", "")
                        state.update_scratchpad(content)
                        state.context.append_tool_result(
                            tool_call_id=tool_call_id,
                            tool_name=tool_name,
                            content=f"Updated: {log_preview_medium(content)}",
                        )
                        continue

                    logger.info(
                        f"[SkillAgent] Tool call: tool={tool_name}, "
                        f"raw_args_keys={list(args.keys())}, "
                        f"raw_args_preview='{str(args)[:200]}'"
                    )

                    # Inject skill_deps if tool schema declares a skill_deps parameter.
                    # This avoids hardcoding tool names in the adapter layer.
                    tp: dict[str, Any] = {}
                    if hasattr(self, "_skill") and self._skill is not None:
                        from tools import get_registry
                        registry = get_registry()
                        schemas = registry.get_schemas_by_names([tool_name])
                        schema_props = {}
                        if schemas:
                            schema = schemas[0]
                            if "function" in schema:
                                schema_props = schema["function"].get("parameters", {}).get("properties", {})
                            else:
                                schema_props = schema.get("properties", {})
                        if "skill_deps" in schema_props:
                            deps = getattr(self._skill, "dependencies", None) or []
                            if deps:
                                tp["skill_deps"] = deps

                    observation_str, error_dict = await adapter.execute(
                        tool_name=tool_name,
                        raw_args=args,
                        tool_props=tp,
                        env_vars=self._build_env_vars(workspace_root, state),
                    )

                    logger.debug(
                        f"[SkillAgent] adapter.execute result: "
                        f"tool={tool_name}, "
                        f"error_dict={error_dict is not None}, "
                        f"result_len={len(observation_str)}, "
                        f"result_preview='{observation_str[:150]}'"
                    )

                    if error_dict:
                        error_type = error_dict.get("error_type", "UNKNOWN")
                        logger.info(
                            f"[SkillAgent] TOOL ERROR: tool={tool_name}, "
                            f"error_type={error_type}, "
                            f"error_preview='{str(error_dict.get('message', ''))[:100]}'"
                        )
                        # Policy-blocked commands are not agent failures; treat as weak signal
                        if error_type == "POLICY_BLOCKED":
                            task_signal = "weak"
                        else:
                            task_signal = "none"

                    if error_dict:
                        observation = {
                            "tool": tool_name,
                            "tool_call_id": tool_call_id,
                            "summary": observation_str,
                            "exec_status": "error",
                            "state_delta": {},
                            "task_signal": task_signal,
                            "raw": error_dict,
                        }
                    else:
                        observation = {
                            "tool": tool_name,
                            "tool_call_id": tool_call_id,
                            "summary": observation_str,
                            "exec_status": "success",
                            "state_delta": {},
                            "task_signal": self._evaluate_task_signal(
                                tool_name, observation_str
                            ),
                            "raw": {},
                        }
                        logger.info(
                            "observation built (pre-parse): tool={}, "
                            "exec_status=success, state_delta_initial={}, "
                            "observation_str_preview='{}'",
                            tool_name,
                            {},
                            observation_str[:200] if observation_str else "(empty)",
                        )

                        # python_repl 工具：从 JSON payload 提取 artifacts（降级为辅助补充，
                        # 主要来源已改为 AFTER_TOOL_EXEC Hook 的 fs_changes）
                        try:
                            payload = json.loads(observation_str)
                            artifacts = payload.get("artifacts") or []
                            if artifacts:
                                observation["state_delta"] = {"created_files": artifacts}
                                names = ", ".join(str(a) for a in artifacts)
                                observation["summary"] = (
                                    f"[Artifacts created: {names}]\n"
                                    f"{observation_str}"
                                )
                        except (json.JSONDecodeError, TypeError):
                            pass

                    if observation.get("generated_code"):
                        generated_code = observation["generated_code"]

                    state.update_from_observation(observation)

                    _delta_after = observation.get("state_delta") or {}
                    logger.info(
                        "state.update_from_observation: "
                        "state.created_files={}, state.updated_files={}, "
                        "state.installed_deps={}, delta_keys={}, delta_created={}, delta_updated={}",
                        list(state.created_files),
                        list(state.updated_files),
                        list(state.installed_deps),
                        list(_delta_after.keys()),
                        _delta_after.get("created_files", []),
                        _delta_after.get("updated_files", []),
                    )

                    # ── P1-4: 注册 artifact 到 ArtifactRegistry ───────────────────
                    self._register_artifacts(
                        tool_name=tool_name,
                        observation=observation,
                        state=state,
                        tool_args=args,
                    )

                    # ── P0-3c: AFTER_TOOL_EXEC Hook — 统一产物注册 + fs_changes 注入 ──
                    # FileChangeHook 在 AFTER_TOOL_EXEC 时：
                    #   1. detected_artifacts → 直接注册到 ArtifactRegistry
                    #   2. fs_changes → 填充 observation["state_delta"]（直接文件系统快照，
                    #      不依赖 regex 解析，消除 bash 输出格式不确定性）
                    # 其他 Hook（LoopSupervisionHook 等）返回 deferred_messages / recovery_action
                    hook_result = await hook_executor.execute(
                        HookEvent.AFTER_TOOL_EXEC,
                        HookPayload(
                            event=HookEvent.AFTER_TOOL_EXEC,
                            tool_name=tool_name,
                            args=args,
                            context=self._context,
                            result=observation_str,
                            error=error_dict,
                        ),
                    )

                    # 用 fs_changes（直接快照结果）填充 state_delta — 优先于 regex 解析
                    _hook_fs = hook_result.fs_changes
                    if _hook_fs:
                        _hc = _hook_fs.get("created", [])
                        _hu = _hook_fs.get("modified", [])
                        # 合并到现有 state_delta（python_repl JSON 可能已设置 created_files）
                        _existing_c = set(observation["state_delta"].get("created_files", []))
                        _existing_u = set(observation["state_delta"].get("updated_files", []))
                        observation["state_delta"] = {
                            "created_files": list(_existing_c | set(_hc)),
                            "updated_files": list(_existing_u | set(_hu)),
                        }
                        logger.info(
                            "[ANALYSIS-LOG] agent: fs_changes merged → state_delta "
                            "created={}, updated={}",
                            observation["state_delta"]["created_files"],
                            observation["state_delta"]["updated_files"],
                        )

                    if hook_result.detected_artifacts:
                        for path in hook_result.detected_artifacts:
                            if not state.artifact_registry.is_registered(path):
                                state.artifact_registry.register(
                                    path=path,
                                    tool=tool_name,
                                    turn=state.turn_count,
                                    source="hook_detection",
                                )
                    if hook_result.deferred_messages:
                        deferred_msgs.extend(hook_result.deferred_messages)

                    # P1-2: 处理 Hook 返回的 recovery_action（LoopSupervisionHook 优先级最高）
                    if hook_result.recovery_action == "RECOMMEND_ABORT":
                        loop_type = (hook_result.metadata or {}).get('loop_info') or {}
                        early_exit_outcome = self._build_failure_outcome(
                            skill,
                            f"Stopped: {loop_type.get('type', 'loop')}",
                            state,
                            generated_code,
                            llm_usage=last_llm_usage,
                            finish_reason=last_finish_reason,
                        )

                    # FIX 2+3: 将本轮新创建的文件加入快照，使 created_this_turn 正确检测
                    for p in state.created_files:
                        if p not in _turn_files_before:
                            _turn_files_before.add(p)
                    for p in state.updated_files:
                        if p not in _turn_updated_before:
                            _turn_updated_before.add(p)

                    step_counter += 1
                    if on_step and callable(on_step):
                        try:
                            await on_step(
                                step_number=step_counter,
                                tool_name=tool_name,
                                status=observation.get("exec_status", "unknown"),
                                signal=observation.get("task_signal", "none"),
                                summary=log_preview(
                                    observation.get("summary", ""), default=200
                                ),
                                state_delta=observation.get("state_delta", {}),
                            )
                        except Exception:
                            pass

                    # Note: state_fingerprint tracking (Layer 2) is REMOVED.
                    # Detection is now fully handled by the Hook system (LoopDetector).

                    # 统一消息总线：tool result 通过 context 管理
                    state.context.append_tool_result(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        content=observation.get("summary", ""),
                    )

                    action_history.append({"tool": tool_name, "arguments": args})

                    # 防止相同 bash 命令重复执行（连续 3 次相同命令+相同结果 → 死循环）
                    if tool_name == "bash" and not error_dict:
                        _result_hash = hash(observation_str[:500])
                        if (
                            args.get("command", "") == last_bash_cmd
                            and _result_hash == last_bash_result_hash
                        ):
                            bash_repeat_count += 1
                            if bash_repeat_count >= 3:
                                logger.info(
                                    "[SkillAgent] STOPPED: repeated identical bash "
                                    f"command x{bash_repeat_count}"
                                )
                                state.update_scratchpad(
                                    f"[LOOP] Repeated bash: {last_bash_cmd[:80]}"
                                )
                                return self._build_failure_outcome(
                                    skill,
                                    f"Stopped: repeated identical bash command x{bash_repeat_count}",
                                    state,
                                    generated_code,
                                    llm_usage=last_llm_usage,
                                    finish_reason=last_finish_reason,
                                )
                            logger.info(
                                "[SkillAgent] bash repeat detected: "
                                f"count={bash_repeat_count}, cmd='{last_bash_cmd[:80]}'"
                            )
                            deferred_msgs.append({
                                "role": "user",
                                "content": (
                                    "[System] LOOP_DETECTED [bash_repeat]: "
                                    "You executed the same bash command "
                                    f"{bash_repeat_count} times with identical results. "
                                    "This command already succeeded. "
                                    "Proceed to the next step instead of repeating it."
                                ),
                            })
                        else:
                            bash_repeat_count = 0
                        last_bash_cmd = args.get("command", "")
                        last_bash_result_hash = _result_hash

                    logger.debug(
                        f"[SkillAgent] Progress: "
                        f"created_files={list(state.created_files)}, "
                        f"updated_files={list(state.updated_files)}, "
                        f"total_tool_calls={state.tool_calls_count}"
                    )

                    # FIX 2: 重复状态指纹检查（保留，仍在 tool_call 级别）
                    if (
                        state.repeated_state_fingerprint_count
                        > state.max_repeated_state_fingerprint
                    ):
                        early_exit_outcome = self._build_failure_outcome(
                            skill,
                            "Stopped due to repeated equivalent execution states",
                            state,
                            generated_code,
                            llm_usage=last_llm_usage,
                            finish_reason=last_finish_reason,
                        )

                    # turn 内 early break
                    if early_exit_outcome is not None:
                        break

                # ── FIX 2+3: Turn 级别 stall 检测（修复缺陷2：按 turn 而非 tool_call）
                # FIX 3: 文件创建保护 — 本 turn 有实质产出则不触发 stall
                # 旧方法（快照对比）的问题：file_create 后快照被更新，差集永远为空。
                # 新方法：直接用本轮 state_delta 中有没有 created/updated 文件。
                # FIX 4 的另一面：snapshot 更新后本轮新文件无法用差集检测，
                # 故改为在 stall 检测处累积 _turn_actual_new_files 集合。
                # ── Turn 级别 stall 检测 ───────────────────────────────────────────
                # 唯一可靠的进展指标：文件系统快照对比。
                # _has_actual_new = True → 本 turn 创建/修改了文件 → reset 计数器
                # _has_actual_new = False → 累计 no_progress_count，到阈值触发警告/退出
                _has_actual_new = bool(
                    set(state.created_files) - _turn_files_before
                    or set(state.updated_files) - _turn_updated_before
                )

                if _has_actual_new:
                    state.no_progress_count = 0
                    state.stall_warning_count = 0
                else:
                    state.no_progress_count += 1

                if state.no_progress_count > 6 and not early_exit_outcome:
                    state.stall_warning_count += 1
                    if state.stall_warning_count <= 1:
                        deferred_msgs.append({
                            "role": "user",
                            "content": (
                                "[System] PROGRESS_STALLED: Recent tool calls have not "
                                "produced new files. Complete the task or call task_complete."
                            ),
                        })
                        state.no_progress_count = 0
                    else:
                        early_exit_outcome = self._build_failure_outcome(
                            skill,
                            "Stopped due to no effective task progress",
                            state,
                            generated_code,
                            llm_usage=last_llm_usage,
                            finish_reason=last_finish_reason,
                        )

                logger.debug(
                    f"[SkillAgent] Turn-level stall: "
                    f"has_actual_new={_has_actual_new}, "
                    f"no_progress_count={state.no_progress_count}, "
                    f"stall_warning_count={state.stall_warning_count}"
                )

                # ── 整轮结束：注入 deferred system 消息 ─────────────────────
                for msg in deferred_msgs:
                    state.context.append_user(**msg)

                # ── P4-1: Turn-level microcompact（替代旧的 per-tool-call 粒度）────
                # 每 turn 结束时统一压缩，粒度合理，不重复
                if state._compactor is not None:
                    state._compactor._bind_tool_name_map(
                        state.context._raw_messages
                    )
                    state._compactor.microcompact(state.context._raw_messages)

                logger.debug(
                    f"[SkillAgent] Turn {state.turn_count} end: "
                    f"tool_calls={state.tool_calls_count}, "
                    f"errors={len(state.error_history)}, "
                    f"created={list(state.created_files)}, "
                    f"artifacts={state.artifact_registry.count}, "
                    f"early_exit={'yes' if early_exit_outcome else 'no'}"
                )

                if early_exit_outcome is not None:
                    logger.info(
                        f"[SkillAgent] EARLY EXIT: skill={skill.name}, "
                        f"turn={state.turn_count}/{state.max_turns}, "
                        f"tool_calls={state.tool_calls_count}, "
                        f"created_files={list(state.created_files)}"
                    )
                    return early_exit_outcome

            # ── 循环结束 ────────────────────────────────────────────────────
            # Layer 2 auto_compact：操作 state.context._raw_messages
            if state._compactor is not None:
                if (state._compactor._estimate_tokens(state.context._raw_messages)
                        > state._compactor.threshold):
                    logger.info(
                        f"[SkillAgent] auto_compact: {len(state.context._raw_messages)} messages, "
                        f"threshold={state._compactor.threshold}"
                    )
                    artifact_section = state.artifact_registry.to_prompt_section()
                    state.context._raw_messages[:] = await state._compactor.auto_compact(
                        state.context._raw_messages,
                        artifact_section=artifact_section,
                    )

            if self._goal_met(state, workspace_root):
                logger.info(
                    f"[SkillAgent] Max turns reached but goal met: "
                    f"created_files={list(state.created_files)}, "
                    f"updated_files={list(state.updated_files)}, "
                    f"primary_artifact={state.get_primary_artifact()}"
                )
                return self._build_success_outcome(
                    skill,
                    f"Task completed at turn limit ({state.max_turns}).",
                    state,
                    generated_code,
                )

            logger.info(
                f"[SkillAgent] Max turns reached, goal NOT met: "
                f"created_files={list(state.created_files)}, "
                f"updated_files={list(state.updated_files)}, "
                f"primary_artifact={state.get_primary_artifact()}"
            )
            return self._build_partial_outcome(skill, state, generated_code)
        finally:
            log_skill_exec(skill.name, query, phase="end")

    # ── P1-4: ArtifactRegistry 集成 ──────────────────────────────────────

    def _register_artifacts(
        self,
        tool_name: str,
        observation: dict[str, Any],
        state: ReActState,
        tool_args: dict[str, Any] | None = None,
    ) -> None:
        """从 observation 中提取 created_files/updated_files，注册到 ArtifactRegistry。

        同时补充工具参数层面的语义信息（用于 observation.state_delta 为空的情况）。
        """
        delta = observation.get("state_delta") or {}
        # [ANALYSIS-LOG] Always log at INFO so it appears in logs regardless of DEBUG level
        logger.info(
            "[ANALYSIS-LOG] _register_artifacts: tool={}, turn={}, "
            "delta_keys={}, created_files={}, updated_files={}, "
            "registry_count_before={}, observation_exec_status={}, "
            "observation_summary_len={}",
            tool_name,
            state.turn_count,
            list(delta.keys()),
            delta.get("created_files", []),
            delta.get("updated_files", []),
            state.artifact_registry.count,
            observation.get("exec_status", "unknown"),
            len(observation.get("summary", "")),
        )

        # 补充层：若 state_delta 为空但工具参数明确包含路径信息，补充 delta
        if not delta and tool_args:
            _path = tool_args.get("path")
            _target = tool_args.get("target")
            _source = tool_args.get("source")

            if tool_name == "file_create" and _path:
                delta = {"created_files": [str(_path)]}
                observation["state_delta"] = delta
                logger.info(
                    "_register_artifacts: file_create fallback "
                    "from tool_args, added to delta: created_files={}",
                    delta.get("created_files"),
                )
            elif tool_name == "edit_file_by_lines" and _path:
                delta = {"updated_files": [str(_path)]}
                observation["state_delta"] = delta
                logger.info(
                    "_register_artifacts: edit_file_by_lines fallback "
                    "from tool_args, added to delta: updated_files={}",
                    delta.get("updated_files"),
                )

        # 同步到 state.created_files / state.updated_files
        # （state.update_from_observation 已在此之前调用，delta 当时可能为空）
        if not delta:
            pass  # delta 仍为空，保持不变
        else:
            for p in delta.get("created_files", []):
                if p not in state.created_files:
                    state.created_files.append(p)
                    logger.info(
                        "[ANALYSIS-LOG] _register_artifacts: synced to state.created_files: {}",
                        p,
                    )
                # Register as core artifact so PRIMARY_ARTIFACT_PATH gets set.
                ok, warning = state.lock_artifact(p)
                if ok:
                    logger.info(
                        "[ANALYSIS-LOG] _register_artifacts: locked core artifact: {}",
                        p,
                    )
                elif warning:
                    logger.debug(
                        "[ANALYSIS-LOG] _register_artifacts: lock_artifact skipped for {}: {}",
                        p,
                        warning,
                    )
            for p in delta.get("updated_files", []):
                if p not in state.updated_files:
                    state.updated_files.append(p)
                    logger.info(
                        "[ANALYSIS-LOG] _register_artifacts: synced to state.updated_files: {}",
                        p,
                    )

        for p in delta.get("created_files", []):
            if state.artifact_registry.is_registered(p):
                logger.debug(f"[SkillAgent] Artifact already registered, skip: {p}")
                continue
            content_summary = ArtifactRegistry.read_content_summary(p)
            try:
                size_bytes = Path(p).stat().st_size if Path(p).exists() else None
            except Exception:
                size_bytes = None
            state.artifact_registry.register(
                path=p,
                tool=tool_name,
                turn=state.turn_count,
                content_summary=content_summary,
                size_bytes=size_bytes,
            )
            logger.debug(
                f"[SkillAgent] Artifact registered: {p} (tool={tool_name}, "
                f"turn={state.turn_count}, summary_len={len(content_summary)})"
            )

        for p in delta.get("updated_files", []):
            if not state.artifact_registry.is_registered(p):
                content_summary = ArtifactRegistry.read_content_summary(p)
                try:
                    size_bytes = Path(p).stat().st_size if Path(p).exists() else None
                except Exception:
                    size_bytes = None
                state.artifact_registry.register(
                    path=p,
                    tool=tool_name,
                    turn=state.turn_count,
                    content_summary=content_summary,
                    size_bytes=size_bytes,
                )
                logger.debug(
                    f"[SkillAgent] Artifact updated: {p} (tool={tool_name})"
                )

    # =========================================================================

    def _get_tool_schemas(self, skill: Skill) -> list[dict]:
        """Get atomic + mcp tool schemas for the LLM."""
        all_schemas: list[dict] = []
        try:
            from tools import get_tool_schemas

            atomic_schemas = get_tool_schemas(category="atomic")
            all_schemas.extend(atomic_schemas)
            logger.debug(
                f"[SkillAgent._get_tool_schemas] skill={skill.name}, "
                f"atomic_count={len(atomic_schemas)}, "
                f"atomic_tools={[s['function']['name'] for s in atomic_schemas]}"
            )
        except Exception:
            atomic_schemas = []
            logger.warning(
                "[SkillAgent._get_tool_schemas] Failed to load atomic tool schemas"
            )

        try:
            from tools import get_tool_schemas as gts_mcp

            mcp_schemas = gts_mcp(category="mcp")
            if mcp_schemas:
                all_schemas.extend(mcp_schemas)
                logger.debug(
                    f"[SkillAgent._get_tool_schemas] skill={skill.name}, "
                    f"mcp_count={len(mcp_schemas)}, "
                    f"mcp_tools={[s['function']['name'] for s in mcp_schemas]}"
                )
        except Exception:
            pass

        schemas = all_schemas

        if not schemas:
            logger.warning(
                f"[SkillAgent._get_tool_schemas] skill={skill.name}, "
                f"WARNING: tools registry is empty! "
                f"Verify bootstrap.py calls tools.bootstrap(). "
                f"LLM will have NO tools available — task will fail."
            )

        if not skill.allowed_tools:
            logger.debug(
                f"[SkillAgent._get_tool_schemas] skill={skill.name}, "
                f"allowed_tools=None → passing all {len(schemas)} tools to LLM"
            )
            return schemas

        allowed = set(skill.allowed_tools)
        filtered = [s for s in schemas if s.get("function", {}).get("name") in allowed]
        logger.debug(
            f"[SkillAgent._get_tool_schemas] skill={skill.name}, "
            f"allowed_tools={skill.allowed_tools}, "
            f"matched_count={len(filtered)}/{len(schemas)}, "
            f"matched_tools={[s['function']['name'] for s in filtered]}"
        )
        return filtered

    def _build_env_vars(
        self, workspace_root: Path, state: ReActState
    ) -> dict[str, str]:
        """Build environment variables for tool execution (ENV VAR JAIL)."""
        env_vars: dict[str, str] = {
            "WORKSPACE_ROOT": str(workspace_root),
        }

        primary = state.get_primary_artifact()
        if primary:
            env_vars["PRIMARY_ARTIFACT_PATH"] = primary

        logger.info(
            "[ANALYSIS-LOG] _build_env_vars: primary_artifact={}, "
            "core_artifacts={}, preferred_ext={}, WORKSPACE_ROOT={}",
            primary,
            dict(state.core_artifacts),
            state.preferred_core_extension,
            str(workspace_root),
        )

        env_vars.update(self._get_skill_env())

        return env_vars

    def _get_skill_env(self) -> dict[str, str]:
        """Load LLM config + user env as MEMENTO_* vars.

        Uses the global g_config singleton so that config changes made via the UI
        (which update g_config in-memory via g_config.set()) are immediately
        visible here — no restart required.
        """
        from middleware.config import g_config

        env: dict[str, str] = {}
        try:
            profile = g_config.llm.current_profile
            if profile:
                if profile.model:
                    env["MEMENTO_LLM_MODEL"] = profile.model
                if profile.api_key:
                    env["MEMENTO_LLM_API_KEY"] = profile.api_key
                if profile.base_url:
                    env["MEMENTO_LLM_BASE_URL"] = profile.base_url
                if profile.litellm_provider:
                    env["MEMENTO_LLM_PROVIDER"] = profile.litellm_provider
                if profile.extra_headers:
                    env["MEMENTO_LLM_EXTRA_HEADERS"] = json.dumps(
                        profile.extra_headers, ensure_ascii=False
                    )

            user_env = g_config.get_env()
            if isinstance(user_env, dict):
                for key, value in user_env.items():
                    if value is not None:
                        safe_key = key.upper().replace("-", "_")
                        env[f"MEMENTO_{safe_key}"] = str(value)
        except Exception:
            logger.debug("Failed to load config for skill env injection")

        return env

    def _build_messages(
        self,
        skill: Skill,
        state: ReActState,
        workspace_root: Path,
    ) -> list[dict[str, Any]]:
        """Build LLM messages for a turn.

        P3-1: 统一走 state.context.build_llm_messages()。
        P1-4: 传入 artifact_registry，artifact 清单永远注入到 prompt。
        P3-2: progress_projection 带缓存。
        """
        turn_warning = ""
        remaining = state.max_turns - state.turn_count
        if remaining <= 0:
            turn_warning = "- **FINAL TURN:** Complete the task or summarize progress.\n"
        elif remaining <= 2:
            turn_warning = f"- Warning: Only {remaining} turn(s) remaining.\n"

        # 构建 created_files_list（R4 强制注入）
        created_files_list = self._build_created_files_list(state)

        system_prompt = SKILL_REACT_PROMPT.format(
            skill_name=skill.name,
            description=skill.description or "",
            skill_source_dir=skill.source_dir or "<none>",
            existing_scripts=self._list_existing_scripts(skill),
            skill_content=self._get_skill_content(skill),
            workspace_root=str(workspace_root),
            progress_projection=state.build_progress_projection(),
            artifact_section="",  # artifact 通过 build_llm_messages 的 artifact_registry 参数注入
            physical_world_fact=self._get_real_file_tree_limited(workspace_root),
            turn_warning=turn_warning,
            query=state.query,
            params=json.dumps(state.params or {}, ensure_ascii=False, indent=2),
            created_files_list=created_files_list,
            self_correction_section="",  # 当前阶段暂不注入，后续扩展
            platform_info=self._platform_info(),
        )

        # ── P4-1: 前馈预算感知压缩 ───────────────────────────────────────────
        # 在发消息前检查总 token 量，超 80% budget 时主动压缩
        budget = self._context_budget()
        if state._compactor is not None:
            self._budget_aware_compact(state, system_prompt, budget)

        # 统一消息总线 + artifact_registry 注入
        return state.context.build_llm_messages(
            system_prompt=system_prompt,
            scratchpad=state.scratchpad,
            result_cache=state.result_cache,
            artifact_registry=state.artifact_registry,
        )

    def _build_created_files_list(self, state: ReActState) -> str:
        """生成 R4 强制注入的文件清单（来自 ArtifactRegistry）。"""
        section = state.artifact_registry.to_prompt_section()
        if not section:
            return "<none yet — no files created>"
        # 只返回路径列表，不需要完整 section（section 在 build_llm_messages 中单独注入）
        paths = state.artifact_registry.get_created_paths()
        if not paths:
            return "<none yet — no files created>"
        lines = []
        for p in paths:
            rec = state.artifact_registry.get(p)
            tag = " [verified]" if rec and rec.verified else ""
            lines.append(f"- {p}{tag}")
        return "\n".join(lines)

    # =========================================================================
    # 压缩策略（由 compaction.py 提供，通过 TokenBudgetPolicy 配置）
    # =========================================================================

    def _context_budget(self) -> int:
        """LLM 输入预算 = context_window - max_output_tokens（估算）。"""
        cw = getattr(self._llm, "context_window", 0) or 0
        mt = getattr(self._llm, "max_tokens", 0) or 0
        if cw <= 0:
            return 100_000
        return max(cw - mt, 8192)

    def _budget_aware_compact(
        self,
        state: ReActState,
        system_prompt: str,
        budget: int,
    ) -> None:
        """P4-1: 前馈预算感知压缩 — 发消息前检查并压缩。

        两阶段策略（由 TokenBudgetPolicy 控制阈值）：
        - Stage 1: microcompact（零 LLM，已在 turn 结束时执行过一次，
                   此处作为 guard，若 raw_messages 在 turn 间增长了则再次执行）
        - Stage 2: 超 Stage 2 阈值时，对旧 tool result 做激进截断
          （由 SummarizerRegistry 分发到各工具专用摘要器）

        只压缩 state.context._raw_messages，不动 system_prompt / scratchpad / artifact_registry。
        """
        raw_msgs = state.context._raw_messages
        if not raw_msgs:
            return

        fixed_overhead = len(system_prompt) + len(state.scratchpad)
        raw_chars = sum(
            len(str(c)) for msg in raw_msgs
            for c in ([msg.get("content", "")] if isinstance(msg.get("content"), str)
                      else msg.get("content", []))
        )
        artifact_chars = len(state.artifact_registry.to_prompt_section())
        estimated_tokens = (fixed_overhead + raw_chars + artifact_chars) // 4

        policy: TokenBudgetPolicy = getattr(state, "_budget_policy", None)
        if policy is None:
            policy = TokenBudgetPolicy(budget=budget)

        warn_th = policy.warn_threshold
        urgent_th = policy.urgent_threshold

        logger.debug(
            f"[SkillAgent._budget_aware_compact] "
            f"estimated_tokens={estimated_tokens}/{budget}, "
            f"warn={warn_th}, urgent={urgent_th}, "
            f"raw_msgs={len(raw_msgs)}"
        )

        if estimated_tokens <= warn_th:
            return

        compactor = state._compactor
        if compactor is not None:
            compactor._bind_tool_name_map(raw_msgs)
            compactor.microcompact(raw_msgs)

            raw_chars = sum(
                len(str(c)) for msg in raw_msgs
                for c in ([msg.get("content", "")] if isinstance(msg.get("content"), str)
                          else msg.get("content", []))
            )
            estimated_tokens = (fixed_overhead + raw_chars + artifact_chars) // 4

            if estimated_tokens <= warn_th:
                logger.info(
                    f"[SkillAgent] Stage 1 (microcompact): {len(raw_msgs)} msgs, "
                    f"~{estimated_tokens} tokens"
                )
                return

        if estimated_tokens > urgent_th:
            self._truncate_old_tool_results(state, policy)
            logger.info(
                f"[SkillAgent] Stage 2: {len(state.context._raw_messages)} msgs, "
                f"~{estimated_tokens} tokens"
            )

    def _truncate_old_tool_results(
        self,
        state: ReActState,
        policy: TokenBudgetPolicy,
    ) -> None:
        """Stage 2 激进截断：调用 SummarizerRegistry 对旧 tool results 做智能摘要。

        策略由 TokenBudgetPolicy 配置：
        - 保留最近 N 个 tool result（完整）
        - 已压缩内容（[...] 开头）跳过
        - 其余交给 SummarizerRegistry 分发到工具专用摘要器
        """
        raw_msgs = state.context._raw_messages
        if not raw_msgs:
            return

        compactor = state._compactor
        if compactor:
            compactor._bind_tool_name_map(raw_msgs)

        tool_results = []
        for i, msg in enumerate(raw_msgs):
            if msg.get("role") != "tool":
                continue
            content = str(msg.get("content", ""))
            if content.startswith("[") and "] " in content[:20]:
                continue  # 已被 Stage 1 压缩过，跳过
            tool_call_id = msg.get("tool_call_id", "")
            tool_name = (
                compactor._tool_name_map.get(tool_call_id, "unknown")
                if compactor else msg.get("name", "unknown")
            )
            tool_results.append((i, msg, content, tool_name))

        if not tool_results:
            return

        keep_recent = policy.truncate_keep_recent
        truncate_candidates = tool_results[:-keep_recent]
        if not truncate_candidates:
            return

        summarizer: SummarizerRegistry = getattr(state, "_summarizer_registry", None)
        if summarizer is None:
            summarizer = get_default_registry()

        max_chars = policy.truncate_max_chars
        truncated = 0
        for idx, msg, content, tool_name in truncate_candidates:
            if len(content) <= max_chars:
                continue
            new_content = summarizer.summarize(tool_name, content)
            if new_content != content:
                msg["content"] = new_content
                truncated += 1

        if truncated:
            logger.info(
                f"[SkillAgent] Stage 2: summarized {truncated}/{len(tool_results)} "
                f"old tool results via {type(summarizer).__name__}"
            )

        # Priority 1: Tool pair integrity — 压缩后清理 orphaned call/result 配对
        if compactor is not None:
            state.context._raw_messages[:] = compactor.sanitize_tool_pairs(state.context._raw_messages)

    # =========================================================================

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
        try:
            root = Path(skill.source_dir)
            scripts = [
                f"- {p.relative_to(root)}"
                for p in root.rglob("*.py")
                if p.name != "__init__.py"
            ]
            return "\n".join(scripts) if scripts else "- <none>"
        except Exception:
            return "- <none>"

    def _get_real_file_tree_limited(self, workspace_root: Path) -> str:
        """Get a limited file tree for physical world facts."""
        try:
            parts = ["- File tree:"]
            for p in sorted(workspace_root.rglob("*"))[:30]:
                if p.is_file():
                    parts.append(f"  - {p.relative_to(workspace_root)}")
            return "\n".join(parts) if len(parts) > 1 else "- <empty>"
        except Exception:
            return "- <error reading files>"

    def _goal_met(self, state: ReActState, workspace_root: Path) -> bool:
        """Check if the goal appears to be met."""
        # [ANALYSIS-LOG] Verbose goal_met check
        _created = list(state.created_files)
        _updated = list(state.updated_files)
        _primary = state.get_primary_artifact()
        _primary_exists = Path(_primary).exists() if _primary else False
        _result = bool(_created or _updated or (_primary and _primary_exists))
        logger.info(
            "[ANALYSIS-LOG] _goal_met CHECK: result={}, "
            "created_files={}, updated_files={}, "
            "primary_artifact={!r}, primary_exists={}",
            _result, _created, _updated, _primary, _primary_exists,
        )
        if _result:
            return True

        primary = state.get_primary_artifact()
        if primary and Path(primary).exists():
            return True

        # Fallback: 如果 state.created_files/updated_files 为空，
        # 检查物理文件系统是否有通过 bash 等工具间接创建的文件。
        # 这避免了 bash 工具创建了文件但未被 artifact 追踪体系感知导致的误判。
        if not _created and not _updated:
            try:
                _fs_files = [
                    p for p in workspace_root.rglob("*")
                    if p.is_file()
                    and p.stat().st_size > 1024  # 过滤空文件和配置残留（>1KB）
                ]
                if _fs_files:
                    logger.info(
                        "_goal_met: filesystem fallback triggered — "
                        "found {} file(s) in workspace (not tracked in state). "
                        "Treating goal as met.",
                        len(_fs_files),
                    )
                    return True
            except Exception:
                pass

        return False

    @staticmethod
    def _tool_category(tool_name: str) -> str:
        """Categorize a tool for loop detection."""
        if tool_name in {"python_repl", "bash"}:
            return "code"
        if tool_name in {"file_create", "edit_file_by_lines"}:
            return "write"
        if tool_name in {"read_file", "list_dir", "glob"}:
            return "read"
        if tool_name in {"search_web", "fetch_webpage"}:
            return "web"
        return "other"

    # ── Bash 输出解析 ─────────────────────────────────────────────────────────

    # 常见工具的输出文件路径模式
    _BASH_OUTPUT_PATTERNS = [
        # ffmpeg: Output #0, wav, to '/path/to/file'
        re.compile(
            r"Output\s+#\d+,\s*\w+\s*,\s*to\s+'([^']+)'",
            re.IGNORECASE,
        ),
        # ffmpeg: Output file -> '/path/to/file' (alternative format)
        re.compile(
            r"Output\s+file\s*[:\->]\s*'?([^'>\s]+)'?",
            re.IGNORECASE,
        ),
        # Various tools: 'Created: /path/to/file'
        re.compile(r"(?:Created|Saved|Written|Generated)\s*:\s*'?([^\s'\";]+)'?", re.IGNORECASE),
        # Writing to /path/to/file
        re.compile(r"Writing\s+(?:to\s+)?'?([^\s'\";]+)'?", re.IGNORECASE),
        # Output path in angle brackets: '/path/to/file'
        re.compile(r"'((?:/[a-zA-Z0-9_\-.]+)+)'"),
    ]

    # 绝对路径提取：匹配 / 开头且是文件路径（排除 URLs、注释行的路径）
    _BASH_ABSOLUTE_PATH = re.compile(r"(?<![a-zA-Z0-9_/])(/(?:[^\s\"'`<>\[\]|\\]){3,})")

    @staticmethod
    def _parse_bash_output_files(output: str, workspace_root: Path) -> list[str]:
        """
        从 bash 工具的纯文本输出中解析可能的文件路径。

        Returns:
            在 workspace_root 目录下存在的新文件路径列表（绝对路径）。
        """
        candidates: set[str] = set()

        # 1. 尝试模式匹配
        for pattern in SkillAgent._BASH_OUTPUT_PATTERNS:
            for match in pattern.finditer(output):
                path = match.group(1).strip()
                if path:
                    candidates.add(path)

        # 2. 补充绝对路径提取（排除常见误报）
        for match in SkillAgent._BASH_ABSOLUTE_PATH.finditer(output):
            path = match.group(1).strip()
            if path and len(path) > 5:
                candidates.add(path)

        # 3. 过滤并验证
        valid: list[str] = []
        try:
            ws_str = str(workspace_root.resolve())
            for path in candidates:
                p = Path(path).resolve()
                # 必须在 workspace 内（防止读系统文件）
                if not str(p).startswith(ws_str):
                    continue
                # 必须是文件且存在（不是目录）
                if not p.is_file():
                    continue
                # 排除空文件（通常是命令失败）
                if p.stat().st_size == 0:
                    continue
                valid.append(str(p))
        except Exception:
            pass

        return valid

    @staticmethod
    def _extract_tool_call_parts(
        tool_call: dict | Any,
    ) -> tuple[str, dict[str, Any], str]:
        """Extract tool name, args, and call ID from a tool call."""
        if isinstance(tool_call, dict):
            tc = tool_call.get("function", {})
            name = tc.get("name", "?")
            args_str = tc.get("arguments", "{}")
            call_id = tool_call.get("id", "")
            if isinstance(args_str, str):
                try:
                    args = json.loads(args_str)
                except Exception:
                    args = {"raw": args_str}
            else:
                args = args_str or {}
        else:
            name = getattr(tool_call, "name", "?")
            args = getattr(tool_call, "arguments", {}) or {}
            call_id = getattr(tool_call, "id", "")
        return name, args, call_id

    @staticmethod
    def _evaluate_task_signal(tool_name: str, summary: str) -> str:
        """纯工具类型驱动的 signal 评估（summary 已废弃，字符串匹配不可靠）。

        分层策略：
        1. 文件写入类工具 → 强 signal（它们直接修改文件系统）
        2. 信息获取类工具 → 中 signal（它们提供上下文，但不直接产出）
        3. 其他 → 弱 signal
        4. error_dict 由调用方在构建 observation 时处理（不在此处二次判断）

        stall 检测已不依赖此 signal（直接用文件系统快照对比），
        但 observation["task_signal"] 仍保留用于其他上下文（如 on_step callback）。
        """
        if tool_name in {"file_create", "edit_file_by_lines", "bash"}:
            return "strong"
        if tool_name in {"search_web", "fetch_webpage", "read_file", "grep", "glob", "list_dir"}:
            return "medium"
        return "weak"

    def _build_success_outcome(
        self,
        skill: Skill,
        text: str,
        state: ReActState,
        generated_code: str,
    ) -> tuple[SkillExecutionOutcome, str]:
        """Build a success outcome."""
        projection = state.build_outcome_projection()
        result_payload = {
            "final_response": text or f"Task completed in {state.turn_count} turns.",
            "execution_summary": projection,
        }
        return (
            SkillExecutionOutcome(
                success=True,
                result=result_payload,
                skill_name=skill.name,
                artifacts=list(state.artifact_registry.all_paths),
            ),
            generated_code,
        )

    def _build_failure_outcome(
        self,
        skill: Skill,
        reason: str,
        state: ReActState,
        generated_code: str,
        llm_usage: Any = None,
        finish_reason: str = None,
    ) -> tuple[SkillExecutionOutcome, str]:
        """Build a failure outcome."""
        projection = state.build_outcome_projection()

        # 构建错误详情，包含 token 统计信息
        error_detail: dict[str, Any] = {}
        if llm_usage:
            # 从 usage 对象中提取 token 信息
            if hasattr(llm_usage, 'prompt_tokens'):
                error_detail["input_tokens"] = llm_usage.prompt_tokens
            if hasattr(llm_usage, 'completion_tokens'):
                error_detail["output_tokens"] = llm_usage.completion_tokens
            if hasattr(llm_usage, 'total_tokens'):
                error_detail["total_tokens"] = llm_usage.total_tokens
        if finish_reason:
            error_detail["finish_reason"] = finish_reason
        if state.last_error:
            error_detail["last_error"] = state.last_error

        return (
            SkillExecutionOutcome(
                success=False,
                result={"final_response": "", "execution_summary": projection},
                error=reason,
                error_type=ErrorType.INTERNAL_ERROR,
                error_detail=error_detail if error_detail else None,
                skill_name=skill.name,
                artifacts=list(state.artifact_registry.all_paths),
            ),
            generated_code,
        )

    def _build_partial_outcome(
        self,
        skill: Skill,
        state: ReActState,
        generated_code: str,
    ) -> tuple[SkillExecutionOutcome, str]:
        """Build a partial (max turns) outcome."""
        projection = state.build_outcome_projection()
        return (
            SkillExecutionOutcome(
                success=False,
                result={
                    "final_response": (
                        f"Task partially completed after {state.turn_count} turns. "
                        f"Files: {[Path(p).name for p in state.created_files]}"
                    ),
                    "execution_summary": projection,
                },
                error=f"Max turns ({state.max_turns}) exceeded",
                error_type=ErrorType.EXECUTION_ERROR,
                skill_name=skill.name,
                artifacts=list(state.artifact_registry.all_paths),
            ),
            generated_code,
        )
