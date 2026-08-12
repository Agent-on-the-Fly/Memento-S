# SPDX-License-Identifier: Apache-2.0
"""LoopSupervisionHook — LoopDetector 的 Hook 封装。

职责：
  1. Result-aware 重复检测（action sig + result hash，等效于原 Layer 1）
  2. LoopDetector.record() + detect()（通用 loop 模式检测）
  3. 构建 deferred_messages（loop/saturation 信号）
  4. 决定 recovery_action（RECOMMEND_ABORT / RECOMMEND_RETRY / CONTINUE）
  5. 更新 scratchpad

Detection architecture (long-term fix):
  agent.py 中的硬编码 Layer 1/2/3 检测全部移除。
  所有 loop/stall 检测统一路由到 Hook 系统。
  本 Hook 是唯一出口决策点，recovery_action 是单一决策信号。

Result-aware 重复检测逻辑：
  - 相同 action sig（tool_name + args）：
    - 如果 result hash 变化 → Agent 在正常收集不同信息 → 重置计数器，继续
    - 如果 result hash 不变且次数 >= 3 → 死循环 → RECOMMEND_ABORT
  - 这解决了"正常重复获取不同信息"vs"死循环"的区分问题。
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from shared.hooks.executor import HookDefinition
from shared.hooks.types import HookEvent, HookPayload, HookResult
from core.skill.execution.loop_detector import LoopDetector
from utils.log_config import log_preview
from utils.logger import get_logger

if TYPE_CHECKING:
    from shared.hooks.executor import HookExecutor

logger = get_logger(__name__)

RECOMMEND_ABORT = "RECOMMEND_ABORT"
RECOMMEND_RETRY = "RECOMMEND_RETRY"
CONTINUE = "CONTINUE"

# Result-aware repeated detection 阈值：
# 连续相同 action sig + 相同 result hash 超过此次数 → 死循环
REPEATED_ACTION_THRESHOLD = 3


class LoopSupervisionHook(HookDefinition):
    """
    Loop 监督 Hook — 封装 result-aware 检测 + LoopDetector + 统一决策。

    在 AFTER_TOOL_EXEC 时执行，职责：
      1. Result-aware 重复检测（action sig + result hash）
      2. LoopDetector.record() / detect()
      3. 构建 deferred_messages
      4. 决定 recovery_action
    """

    def __init__(
        self,
        hook_executor: "HookExecutor | None" = None,
        max_observation_chain: int = 6,
        min_effect_ratio: float = 0.15,
        window_size: int = 10,
        repeated_action_threshold: int = REPEATED_ACTION_THRESHOLD,
    ):
        super().__init__()
        self._loop_detector = LoopDetector(
            max_observation_chain=max_observation_chain,
            min_effect_ratio=min_effect_ratio,
            window_size=window_size,
            hook_executor=hook_executor,
        )
        # Result-aware repeated action tracking（等效于原 agent.py Layer 1）
        self._last_action_sig: str | None = None
        self._last_result_hash: int | None = None
        self._repeated_action_count: int = 0
        # Config
        self._repeated_action_threshold = repeated_action_threshold
        # State context
        self._state_ctx: dict[str, Any] | None = None

    def bind_state_context(self, state_ctx: dict) -> None:
        self._state_ctx = state_ctx

    async def execute(self, payload: HookPayload) -> HookResult:
        if payload.event != HookEvent.AFTER_TOOL_EXEC:
            return HookResult(allowed=True)

        tool_name = payload.tool_name
        result = payload.result or ""
        args = payload.args or {}
        state_ctx = self._state_ctx or {}

        # ── 0. 读取 fs_changes（由 FileChangeHook 写入）──────────────
        _hook_fs = self.hook_context.get("fs_changes") or {}
        _hook_created = _hook_fs.get("created", [])
        _hook_modified = _hook_fs.get("modified", [])
        _hook_has_effect = bool(_hook_created or _hook_modified)

        # ── 1. Result-aware 重复检测 ────────────────────────────────
        sig = self._build_sig(tool_name, args)
        result_hash = hash(result[:2000]) if result else None

        if sig == self._last_action_sig:
            if result_hash is not None and self._last_result_hash is not None:
                if result_hash == self._last_result_hash:
                    self._repeated_action_count += 1
                else:
                    self._repeated_action_count = 0
            else:
                self._repeated_action_count += 1
        else:
            self._repeated_action_count = 0

        self._last_action_sig = sig
        self._last_result_hash = result_hash

        _abort_signal: dict[str, Any] | None = None
        if (
            self._repeated_action_count >= self._repeated_action_threshold - 1
            and self._last_result_hash is not None
            and result_hash is not None
            and self._last_result_hash == result_hash
        ):
            _abort_signal = {
                "type": "repeated_action",
                "count": self._repeated_action_count + 1,
                "tool": tool_name,
            }

        _repeated_info: dict[str, Any] | None = None
        if _abort_signal:
            _repeated_info = {
                "type": "repeated_action",
                "count": _abort_signal["count"],
                "tool": tool_name,
                "result_changed": False,
            }

        # ── 2. LoopDetector record ─────────────────────────────────
        if self._should_record(tool_name):
            tool_category = self._tool_category(tool_name)
            new_entities = self._extract_entities(result)
            created = self._extract_created_count(result)

            _effective_created = created
            if _hook_has_effect:
                _effective_created = len(_hook_created) + len(_hook_modified)

            self._loop_detector.record(
                tool_name=tool_name,
                category=tool_category,
                turn=self._get_turn(state_ctx),
                new_entities=len(new_entities),
                created_artifacts=_effective_created,
                artifact_registry=state_ctx.get("artifact_registry"),
            )

        # ── 3. LoopDetector detect ──────────────────────────────────
        _loop_info = self._loop_detector.detect()

        # ── 4. 构建 deferred_messages ────────────────────────────────
        deferred_messages: list[dict[str, Any]] = []

        # 4a. Result-aware repeated signal
        if _abort_signal:
            deferred_messages.append({
                "role": "user",
                "content": (
                    f"[System] LOOP_DETECTED [repeated_action]: "
                    f"You've called `{tool_name}` {_abort_signal['count']} times "
                    f"with identical results. This tool already succeeded. "
                    f"Stop fetching and immediately use a creation tool "
                    f"(e.g., file_create, python_repl) to save your findings. "
                    f"Use respond_final_answer when done."
                ),
            })
            self._update_scratchpad(
                state_ctx,
                f"[LOOP] repeated_action: {tool_name} x{_abort_signal['count']}",
            )

        # 4b. LoopDetector patterns
        if _loop_info:
            deferred_messages.append({
                "role": "user",
                "content": (
                    f"[System] LOOP_DETECTED [{_loop_info['type']}]: "
                    f"{_loop_info['message']}"
                ),
            })
            self._update_scratchpad(
                state_ctx,
                f"[LOOP] {_loop_info['type']}: "
                f"{log_preview(_loop_info['message'], default=80)}",
            )

        # ── 5. 决定 recovery_action ────────────────────────────────
        # 优先级：abort_signal > loop_info > None
        recovery_action = self._decide_recovery_action(_abort_signal, _loop_info)

        return HookResult(
            allowed=True,
            deferred_messages=deferred_messages if deferred_messages else None,
            recovery_action=recovery_action,
            metadata={
                "loop_info": _loop_info,
                "repeated_info": _repeated_info,
                "abort_signal": _abort_signal,
            },
        )

    # ── 内部辅助方法 ──────────────────────────────────────────────

    def _build_sig(self, tool_name: str, args: dict[str, Any]) -> str:
        """构建 action signature（与 agent.py 原 action_signature 逻辑一致）。"""
        try:
            args_str = json.dumps(args, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            args_str = str(args)
        sig_raw = f"{tool_name}|{args_str}"
        return hashlib.sha1(sig_raw.encode("utf-8")).hexdigest()

    def _should_record(self, tool_name: str) -> bool:
        return bool(tool_name)

    def _tool_category(self, tool_name: str) -> str:
        if tool_name in {"python_repl", "bash", "js_repl"}:
            return "code"
        if tool_name in {"file_create", "edit_file", "edit_file_by_lines", "write_file"}:
            return "write"
        if tool_name in {"read_file", "list_dir", "glob"}:
            return "read"
        if tool_name in {"search_web", "fetch_webpage"}:
            return "web"
        return "other"

    def _extract_entities(self, result: str) -> list:
        try:
            parsed = json.loads(result)
            return parsed.get("result_entities", [])
        except Exception:
            pass

        lines = result.strip().split("\n")
        search_results = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("---"):
                continue
            if stripped.startswith("- [") or stripped.startswith("* ["):
                search_results.append(stripped)
            elif stripped and stripped[0].isdigit() and "." in stripped[:3]:
                search_results.append(stripped)
        if len(search_results) >= 2:
            return search_results

        content_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("---") or stripped.startswith("***"):
                continue
            if stripped.startswith("http://") or stripped.startswith("https://"):
                if len(stripped.split()) <= 3:
                    continue
            skip_prefixes = ("Source:", "访问", "Click here", "Read more", "Jump to")
            if any(stripped.startswith(p) for p in skip_prefixes):
                continue
            if len(stripped) >= 20:
                content_lines.append(stripped)
        return content_lines

    def _extract_created_count(self, result: str) -> int:
        try:
            parsed = json.loads(result)
            artifacts = parsed.get("artifacts") or []
            created = parsed.get("created_files") or []
            return len(artifacts) + len(created)
        except Exception:
            return 0

    def _get_turn(self, state_ctx: dict) -> int:
        v = state_ctx.get("turn_count", 0)
        return v() if callable(v) else v

    def _update_scratchpad(self, state_ctx: dict, text: str) -> None:
        update_scratchpad = state_ctx.get("update_scratchpad")
        if update_scratchpad and callable(update_scratchpad):
            try:
                update_scratchpad(text)
            except Exception:
                pass

    def _decide_recovery_action(
        self,
        abort_signal: dict[str, Any] | None,
        loop_info: dict[str, Any] | None,
    ) -> str | None:
        """根据检测结果决定恢复动作。优先级：abort_signal > loop_info"""
        # 最高优先：result-aware repeated action
        if abort_signal:
            return RECOMMEND_ABORT

        # LoopDetector 模式
        if not loop_info:
            return None

        t = loop_info.get("type", "")
        # 真正的死循环（相同 action sig + 相同结果）：ABORT
        if t in {
            "repeating_action",
            "repeated_action",
            "repeated_state",
            "repeating_state",
        }:
            return RECOMMEND_ABORT
        # repeating_sequence：正常迭代工作流已被智能跳过，
        # 剩余情况只发 RETRY 警告，不打断执行
        if t in {
            "repeating_sequence",
            "no_progress",
            "stall",
            "observation_chain",
            "low_effect_ratio",
            "diminishing_returns",
        }:
            return RECOMMEND_RETRY
        return None
