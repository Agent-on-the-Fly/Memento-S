# SPDX-License-Identifier: Apache-2.0
"""SaturationSupervisionHook — InfoSaturationDetector 的 Hook 封装。

职责：
  1. InfoSaturationDetector.record() + check_saturation()
  2. 构建 deferred_messages（saturation 部分）
  3. 更新 scratchpad

仅对 search_web / fetch_webpage 工具有效。
"""

from __future__ import annotations

from typing import Any

from shared.hooks.executor import HookDefinition
from shared.hooks.types import HookEvent, HookPayload, HookResult
from core.skill.execution.content_analyzer import InfoSaturationDetector
from utils.logger import get_logger

logger = get_logger(__name__)


class SaturationSupervisionHook(HookDefinition):
    """
    Saturation 监督 Hook — 封装 InfoSaturationDetector 的检测逻辑。

    仅对 search_web / fetch_webpage 工具有效。
    在 AFTER_TOOL_EXEC 时执行，执行顺序：
      1. InfoSaturationDetector.record()
      2. InfoSaturationDetector.check_saturation()
      3. 构建 deferred_messages（saturation 部分）
      4. 更新 scratchpad
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        entity_overlap_threshold: float = 0.8,
        min_results_for_analysis: int = 3,
    ):
        """
        初始化 SaturationSupervisionHook。

        Args:
            similarity_threshold: 内容相似度阈值（传给 InfoSaturationDetector）
            entity_overlap_threshold: 实体重叠阈值（传给 InfoSaturationDetector）
            min_results_for_analysis: 最小结果数（传给 InfoSaturationDetector）
        """
        super().__init__()
        self._saturation_detector = InfoSaturationDetector(
            similarity_threshold=similarity_threshold,
            entity_overlap_threshold=entity_overlap_threshold,
            min_results_for_analysis=min_results_for_analysis,
        )
        self._state_ctx: dict[str, Any] | None = None

    def bind_state_context(self, state_ctx: dict) -> None:
        """由 SkillAgent 在初始化时调用，传入必要的状态引用。

        Args:
            state_ctx: 包含以下键的字典：
                - turn_count: int（当前 turn 编号）
                - update_scratchpad: callable
        """
        self._state_ctx = state_ctx

    async def execute(self, payload: HookPayload) -> HookResult:
        """执行 Saturation 检测逻辑。仅对 search_web/fetch_webpage 有效。"""
        if payload.event != HookEvent.AFTER_TOOL_EXEC:
            return HookResult(allowed=True)

        tool_name = payload.tool_name
        result = payload.result or ""
        args = payload.args or {}
        state_ctx = self._state_ctx or {}

        # 仅对 search_web / fetch_webpage 有效
        if tool_name not in {"search_web", "fetch_webpage"}:
            return HookResult(allowed=True)

        # ── 1. record ─────────────────────────────────────────────────────
        query_text = self._extract_query(tool_name, args)
        self._saturation_detector.record(
            tool_name=tool_name,
            query=query_text,
            content=result,
            turn=self._get_turn(state_ctx),
        )

        # ── 2. check_saturation ──────────────────────────────────────────
        saturation_info = self._saturation_detector.check_saturation()

        # ── 3. deferred_messages ─────────────────────────────────────────
        deferred_messages: list[dict[str, Any]] = []
        if saturation_info:
            stats = self._saturation_detector.get_stats()
            deferred_messages.append({
                "role": "user",
                "content": (
                    f"[System] INFO_SATURATION [{saturation_info['type']}]: "
                    f"{saturation_info['message']} "
                    f"(Stats: {stats.get('total_searches', 0)} searches, "
                    f"{stats.get('unique_entities', 0)} unique entities)"
                ),
            })
            self._update_scratchpad(
                state_ctx,
                f"[SATURATION] {saturation_info['type']}: "
                "Information collection complete. Proceed to creation.",
            )

        return HookResult(
            allowed=True,
            deferred_messages=deferred_messages if deferred_messages else None,
            metadata={"saturation_info": saturation_info},
        )

    # ── 内部辅助方法 ──────────────────────────────────────────────────────

    def _extract_query(self, tool_name: str, args: dict) -> str:
        """从工具参数中提取查询文本。"""
        if tool_name == "search_web":
            return args.get("query", "") if args else ""
        if tool_name == "fetch_webpage":
            return args.get("url", "") if args else ""
        return ""

    def _get_turn(self, state_ctx: dict) -> int:
        """从 state_ctx 提取 turn_count（支持 lambda）。"""
        v = state_ctx.get("turn_count", 0)
        return v() if callable(v) else v

    def _update_scratchpad(self, state_ctx: dict, text: str) -> None:
        """更新 scratchpad。"""
        update_scratchpad = state_ctx.get("update_scratchpad")
        if update_scratchpad and callable(update_scratchpad):
            try:
                update_scratchpad(text)
            except Exception:
                pass
