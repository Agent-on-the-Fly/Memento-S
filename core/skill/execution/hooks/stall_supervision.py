# SPDX-License-Identifier: Apache-2.0
"""StallSupervisionHook — 多维进展检测 Hook。

职责：
  替代 agent.py 中的硬编码 Layer 3 (no_progress_count / stall_warning_count)。
  在 Hook 层面统一检测多个维度的"进展"，解决"纯文件检测"的盲区。

进展维度：
  1. 文件进展：created_files / updated_files 有新增
  2. 信息进展：InfoSaturationDetector 检测到新实体或新内容指纹
  3. Scratchpad 进展：scratchpad 内容有实质增长

检测逻辑：
  - 每次 execute() 记录各维度状态
  - 与上次记录对比：有任一维度有实质进展 → reset
  - 连续 N 次无实质进展 → RECOMMEND_RETRY（发 warning 消息）
  - 连续 N+M 次仍无实质进展 → RECOMMEND_ABORT

修复的问题：
  - 原 Layer 3 只看文件 → fetch_webpage 收集到大量信息但无文件 → 误判为 stall
  - 原 Layer 1 在第 3 次重复时拦截 → 绕过了本应介入的 stall 检测
  - 现在所有检测统一通过 Hook 架构协同工作
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from shared.hooks.executor import HookDefinition
from shared.hooks.types import HookEvent, HookPayload, HookResult
from core.skill.execution.content_analyzer import InfoSaturationDetector
from utils.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

RECOMMEND_ABORT = "RECOMMEND_ABORT"
RECOMMEND_RETRY = "RECOMMEND_RETRY"

# 配置
DEFAULT_STALL_WARNING_THRESHOLD = 3  # N 次无进展 → 发 warning
DEFAULT_STALL_ABORT_THRESHOLD = 2  # warning 后再 N 次仍无进展 → abort


class StallSupervisionHook(HookDefinition):
    """
    多维 stall 检测 Hook。

    检测 3 个维度的进展：文件、信息、scratchpad。
    任一维度有实质进展即视为有效推进。
    """

    def __init__(
        self,
        stall_warning_threshold: int = DEFAULT_STALL_WARNING_THRESHOLD,
        stall_abort_threshold: int = DEFAULT_STALL_ABORT_THRESHOLD,
    ):
        super().__init__()
        self._stall_warning_threshold = stall_warning_threshold
        self._stall_abort_threshold = stall_abort_threshold

        # 状态追踪
        self._last_created_files: set[str] = set()
        self._last_updated_files: set[str] = set()
        self._last_scratchpad_len: int = 0
        self._last_artifact_registry_count: int = 0

        # 计数器
        self._consecutive_stall_count: int = 0
        self._stall_warning_issued: bool = False

        # InfoSaturationDetector（共享给 SaturationSupervisionHook？不，各自独立）
        self._saturation_detector = InfoSaturationDetector(
            similarity_threshold=0.7,
            entity_overlap_threshold=0.8,
            min_results_for_analysis=3,
        )

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

        # ── 1. 读取 fs_changes（由 FileChangeHook 写入）──────────────
        _hook_fs = self.hook_context.get("fs_changes") or {}
        _hook_created = set(_hook_fs.get("created", []))
        _hook_modified = set(_hook_fs.get("modified", []))

        # ── 2. 收集 3 个维度的当前状态 ──────────────────────────
        # 2a. 文件维度
        _new_created = _hook_created - self._last_created_files
        _new_updated = _hook_modified - self._last_updated_files
        _has_file_progress = bool(_new_created or _new_updated)

        # 2b. 信息维度（通过 InfoSaturationDetector）
        _query = self._extract_query(tool_name, args)
        self._saturation_detector.record(
            tool_name=tool_name,
            query=_query,
            content=result,
            turn=self._get_turn(state_ctx),
        )
        _saturation = self._saturation_detector.check_saturation()
        # 注意：saturation 检测到的是"信息饱和"，反过来说明之前在收集信息
        # 真正的信息进展 = 最近 1-2 次有新实体或新内容指纹
        _has_info_progress = self._check_info_progress(result)

        # 2c. Scratchpad 维度
        _scratchpad = state_ctx.get("scratchpad", "") or ""
        _current_scratchpad_len = len(_scratchpad)
        _scratchpad_delta = _current_scratchpad_len - self._last_scratchpad_len
        # 实质增长：至少 50 字符的增量（避免换行等微小变化）
        _has_scratchpad_progress = _scratchpad_delta >= 50

        # 2d. ArtifactRegistry 维度（如果有接入）
        _registry = state_ctx.get("artifact_registry")
        if _registry is not None:
            _current_count = len(getattr(_registry, "all_paths", []))
            _has_registry_progress = _current_count > self._last_artifact_registry_count
            self._last_artifact_registry_count = _current_count
        else:
            _has_registry_progress = False

        # ── 3. 综合判断是否有实质进展 ─────────────────────────────
        _has_progress = (
            _has_file_progress
            or _has_info_progress
            or _has_scratchpad_progress
            or _has_registry_progress
        )

        # ── 4. 更新状态快照 ───────────────────────────────────────
        self._last_created_files |= _hook_created
        self._last_updated_files |= _hook_modified
        self._last_scratchpad_len = _current_scratchpad_len

        # ── 5. 计数器管理 ─────────────────────────────────────────
        if _has_progress:
            self._consecutive_stall_count = 0
            self._stall_warning_issued = False
        else:
            self._consecutive_stall_count += 1

        # ── 6. 决策 ───────────────────────────────────────────────
        deferred_messages: list[dict[str, Any]] = []
        recovery_action: str | None = None

        _warn_threshold = self._stall_warning_threshold
        _abort_threshold = _warn_threshold + self._stall_abort_threshold

        if self._consecutive_stall_count >= _abort_threshold:
            recovery_action = RECOMMEND_ABORT
            deferred_messages.append({
                "role": "user",
                "content": (
                    "[System] STALL_ABORT: After multiple turns with no effective progress, "
                    "this skill is stuck. Stop executing and return control to the planner "
                    "to try a different approach."
                ),
            })
            logger.info(
                f"[StallSupervision] ABORT: stall_count={self._consecutive_stall_count}"
            )
        elif (
            self._consecutive_stall_count >= _warn_threshold
            and not self._stall_warning_issued
        ):
            recovery_action = RECOMMEND_RETRY  # warning，不 abort
            self._stall_warning_issued = True
            deferred_messages.append({
                "role": "user",
                "content": (
                    "[System] PROGRESS_STALLED: Recent tool calls have not produced new files "
                    "or new information. Complete the task using available tools, or call "
                    "task_complete / respond_final_answer to end. Do not repeat the same "
                    "information-gathering actions."
                ),
            })
            logger.info(
                f"[StallSupervision] WARNING: stall_count={self._consecutive_stall_count}, "
                f"file_progress={_has_file_progress}, info_progress={_has_info_progress}, "
                f"scratchpad_progress={_has_scratchpad_progress}"
            )

        return HookResult(
            allowed=True,
            deferred_messages=deferred_messages if deferred_messages else None,
            recovery_action=recovery_action,
            metadata={
                "stall_count": self._consecutive_stall_count,
                "file_progress": _has_file_progress,
                "info_progress": _has_info_progress,
                "scratchpad_progress": _has_scratchpad_progress,
                "new_created": list(_new_created),
                "new_updated": list(_new_updated),
            },
        )

    def _check_info_progress(self, result: str) -> bool:
        """检查最近是否有实质性的信息增量。"""
        stats = self._saturation_detector.get_stats()
        _total_calls = stats.get("total_calls", 0)
        _unique_content = stats.get("unique_content_pieces", 0)

        if _total_calls >= 1 and _unique_content >= 1:
            if _unique_content >= _total_calls:
                return True
        return False

    def _extract_query(self, tool_name: str, args: dict) -> str:
        if tool_name == "search_web":
            return args.get("query", "") if args else ""
        if tool_name == "fetch_webpage":
            return args.get("url", "") if args else ""
        return ""

    def _get_turn(self, state_ctx: dict) -> int:
        v = state_ctx.get("turn_count", 0)
        return v() if callable(v) else v

    def reset(self) -> None:
        """重置所有状态（skill 切换时调用）。"""
        self._consecutive_stall_count = 0
        self._stall_warning_issued = False
        self._last_created_files.clear()
        self._last_updated_files.clear()
        self._last_scratchpad_len = 0
        self._last_artifact_registry_count = 0
