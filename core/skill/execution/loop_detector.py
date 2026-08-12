# SPDX-License-Identifier: Apache-2.0
"""Generic loop detection for ReAct execution.

Detects various loop patterns based on tool usage behavior,
not hardcoded tool names.

Category 推导策略:
- 优先使用真实 fs_changes（来自 FileChangeHook 共享上下文）：
  hook_context["fs_changes"] 有 created/modified 文件 → effect
- 次选 ArtifactRegistry 注册表路径新增 → effect
- 次选 state_delta created_artifacts → effect
- 两者都没有 → observation（由 _tool_category 决定）

observation_chain 检测的智能跳过条件：
- 最近的 observation 工具返回了实质性信息（new_entities >= 2）→ 跳过警告
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from utils.logger import get_logger

if TYPE_CHECKING:
    from core.skill.execution.artifact_registry import ArtifactRegistry
    from shared.hooks.executor import HookExecutor
    from shared.hooks.types import HookEvent, HookPayload

logger = get_logger(__name__)


@dataclass
class ToolCallRecord:
    """Record of a single tool call."""

    tool_name: str
    category: str  # "observation" / "effect" (统一由 record() 推导)
    turn: int
    new_entities: int = 0  # New URLs, files, etc. discovered
    created_artifacts: int = 0  # Files created/modified (来自 state_delta)
    workspace_changed: bool = False  # ArtifactRegistry 感知到的文件变更


class LoopDetector:
    """Detects execution loops based on behavior patterns.

    数据源: ArtifactRegistry 是文件变更的唯一真实来源。
    每轮 record() 时对比当前注册表与上次快照，
    路径集合扩大即为 workspace_changed = True。

    detect() 检测到 loop 时，直接持有 hook_executor 引用，
    异步触发 ON_LOOP_DETECTED 遥测事件（不经过 ReactSupervisionHook 转发）。
    """

    def __init__(
        self,
        max_observation_chain: int = 6,
        min_effect_ratio: float = 0.15,
        window_size: int = 10,
        hook_executor: "HookExecutor | None" = None,
    ):
        """
        Args:
            max_observation_chain: Max consecutive observation tools before warning
            min_effect_ratio: Minimum ratio of effect tools to total tools
            window_size: Sliding window size for analysis
            hook_executor: HookExecutor 实例（用于 detect() 时直接触发 ON_LOOP_DETECTED）
        """
        self.max_observation_chain = max_observation_chain
        self.min_effect_ratio = min_effect_ratio
        self.window_size = window_size
        self.history: list[ToolCallRecord] = []
        # ArtifactRegistry 路径快照：用于感知文件新增/修改
        # 每次 record() 后更新，与当前注册表对比检测变更
        self._last_registry_paths: frozenset[str] = frozenset()
        self._hook_executor: "HookExecutor | None" = hook_executor

    def record(
        self,
        tool_name: str,
        category: str,
        turn: int,
        new_entities: int = 0,
        created_artifacts: int = 0,
        artifact_registry: ArtifactRegistry | None = None,
    ) -> None:
        """Record a tool call.

        统一推导 category，不依赖 _tool_category() 的工具类型分类。

        决策树:
        1. created_artifacts > 0（来自 FileChangeHook 共享上下文的 fs_changes）→ effect
        2. artifact_registry 有新增路径（相比上次快照）→ workspace_changed = True → effect
        3. 两者都没有 → observation
        """
        # 1. 显式产物兜底：state_delta 提供的 created_files（优先级最高）
        has_explicit_effect = created_artifacts > 0 or new_entities > 0

        # 2. ArtifactRegistry 路径对比（感知文件新增/修改）
        workspace_changed = False
        if artifact_registry is not None:
            current_paths = frozenset(artifact_registry.all_paths)
            if current_paths != self._last_registry_paths:
                workspace_changed = True
            # 快照更新在判断之后：当前轮产生的文件变更也能被感知
            self._last_registry_paths = current_paths

        # 3. 统一推导 category
        effective_category = (
            "effect" if (has_explicit_effect or workspace_changed) else "observation"
        )

        self.history.append(
            ToolCallRecord(
                tool_name=tool_name,
                category=effective_category,
                turn=turn,
                new_entities=new_entities,
                created_artifacts=created_artifacts,
                workspace_changed=workspace_changed,
            )
        )

    def detect(self) -> dict[str, Any] | None:
        """Detect if execution is in a loop.

        检测到 loop 时，直接触发 ON_LOOP_DETECTED 遥测事件（不经过 ReactSupervisionHook 转发）。

        Returns:
            Loop info dict if detected, None otherwise
        """
        if len(self.history) < 5:
            return None

        # Pattern 1: Long chain of observation without effect
        result = self._check_observation_chain()
        if result:
            self._trigger_loop_telemetry(result)
            return result

        # Pattern 2: Low effect ratio in sliding window
        result = self._check_effect_ratio()
        if result:
            self._trigger_loop_telemetry(result)
            return result

        # Pattern 3: Diminishing returns (info collection loop)
        result = self._check_diminishing_returns()
        if result:
            self._trigger_loop_telemetry(result)
            return result

        # Pattern 4: Repeating tool sequences
        result = self._check_repeating_sequence()
        if result:
            self._trigger_loop_telemetry(result)
            return result

        return None

    def _trigger_loop_telemetry(self, loop_info: dict[str, Any]) -> None:
        """异步触发 ON_LOOP_DETECTED 遥测事件。

        不影响检测流程主线程：遥测失败不影响检测结果。
        """
        if self._hook_executor is None:
            return

        # 获取最近一条记录以获取 turn 信息
        turn = self.history[-1].turn if self.history else 0

        payload = {
            "event": "on_loop_detected",
            "tool_name": self.history[-1].tool_name if self.history else "",
            "args": {},
            "context": None,
            "skill": None,
            "skill_params": None,
            "result": None,
            "error": None,
            "metadata": {
                "loop_info": loop_info,
                "turn": turn,
            },
        }

        # 构造 HookPayload（注意 frozen dataclass 需要用 __new__ 或直接构造）
        try:
            from shared.hooks.types import HookEvent, HookPayload

            hp = HookPayload(
                event=HookEvent.ON_LOOP_DETECTED,
                tool_name=payload["tool_name"],
                args=payload["args"],
                context=payload["context"],
                skill=payload["skill"],
                skill_params=payload["skill_params"],
                result=payload["result"],
                error=payload["error"],
            )
            object.__setattr__(hp, "metadata", payload["metadata"])

            # 异步触发，不阻塞检测流程
            loop_ref = asyncio.get_event_loop()
            if loop_ref.is_running():
                asyncio.create_task(
                    self._hook_executor.execute(HookEvent.ON_LOOP_DETECTED, hp)
                )
            else:
                loop_ref.run_until_complete(
                    self._hook_executor.execute(HookEvent.ON_LOOP_DETECTED, hp)
                )
        except Exception:
            # 遥测失败不影响检测流程
            pass

    def _check_observation_chain(self) -> dict[str, Any] | None:
        """Check for long consecutive observation tool chains.

        智能跳过条件：
        1. 最近的 observation 工具读取的是本轮 skill 执行中新创建的文件。
           这是正常的信息整理行为，不是 loop。
        2. 最近的 observation 工具返回了新信息（new_entities > 0）。
        """
        if len(self.history) < 3:
            return None

        # Count trailing observation tools
        obs_records = []
        for record in reversed(self.history):
            if record.category == "observation":
                obs_records.append(record)
            else:
                break

        if len(obs_records) < self.max_observation_chain:
            return None

        # 智能跳过：如果最近的 observation 有实质性信息产出，跳过 loop 警告
        recent_obs = obs_records[:3]
        # 有新信息返回的 observation 工具（如 search_web/fetch_webpage 返回了多个结果）
        # 或者 read_file 读取的是当前执行中新创建的文件
        has_meaningful_obs = any(r.new_entities >= 2 for r in recent_obs)
        if has_meaningful_obs:
            logger.info(
                "[LoopDetector] Skipping observation_chain: recent tools returned "
                "substantial information (new_entities >= 2)"
            )
            return None

        obs_chain = len(obs_records)
        return {
            "type": "observation_chain",
            "severity": "high",
            "message": (
                f"You've used {obs_chain} consecutive observation tools "
                f"without creating or modifying anything. "
                "This is a RESEARCH LOOP. "
                "ACTION REQUIRED: Stop searching/reading and immediately use "
                "an appropriate creation tool (e.g., file_create) to write "
                "your results to a file. Do NOT continue observing."
            ),
            "chain_length": obs_chain,
        }

    def _check_effect_ratio(self) -> dict[str, Any] | None:
        """Check if effect tools ratio is too low."""
        window = self.history[-self.window_size :]
        total = len(window)
        effect_count = sum(1 for r in window if r.category == "effect")
        ratio = effect_count / total if total > 0 else 0

        if total >= self.window_size and ratio < self.min_effect_ratio:
            return {
                "type": "low_effect_ratio",
                "severity": "medium",
                "message": (
                    f"In the last {total} actions, only {effect_count} "
                    f"({ratio:.0%}) created or modified files. "
                    "You're collecting information faster than using it. "
                    "Switch to creation mode."
                ),
                "ratio": ratio,
                "window_size": total,
            }
        return None

    def _check_diminishing_returns(self) -> dict[str, Any] | None:
        """Check if new information discovery is decreasing."""
        # Look at last 6 observation tools
        obs_records = [r for r in self.history if r.category == "observation"][-6:]

        if len(obs_records) < 4:
            return None

        # Check if new entities are decreasing
        entities = [r.new_entities for r in obs_records]
        if all(e <= 1 for e in entities[-3:]) and sum(entities) > 0:
            return {
                "type": "diminishing_returns",
                "severity": "medium",
                "message": (
                    "Your recent searches found very little new information. "
                    "This is a DIMINISHING RETURNS loop. "
                    "ACTION REQUIRED: Stop searching and immediately use "
                    "file_create to save the information you already have. "
                    "Do NOT search again."
                ),
                "recent_entities": entities[-3:],
            }
        return None

    def _check_repeating_sequence(self) -> dict[str, Any] | None:
        """Check for repeating tool call patterns (e.g., A-B-A-B).

        智能跳过条件：
        - pattern 中每个步骤都有实质效果产出（workspace_changed 或 created_artifacts > 0）
          → 这是正常的迭代工作流，不是死循环
        """
        if len(self.history) < 6:
            return None

        # Check for 2-tool or 3-tool repeating patterns
        for pattern_len in [2, 3]:
            if len(self.history) < pattern_len * 2:
                continue

            recent_records = self.history[-pattern_len * 2 :]
            recent = [r.tool_name for r in recent_records]
            pattern = recent[:pattern_len]

            # Check if pattern repeats exactly
            if recent == pattern * 2:
                # 智能跳过：如果 pattern 中每个步骤都有效果产出 → 不是 loop
                has_effect = all(
                    r.workspace_changed or r.created_artifacts > 0 for r in recent_records
                )
                if has_effect:
                    logger.info(
                        "[LoopDetector] Skipping repeating_sequence: all steps "
                        "in pattern produced effects (workspace_changed or created_artifacts)"
                    )
                    return None
                return {
                    "type": "repeating_sequence",
                    "severity": "high",
                    "message": (
                        f"You're repeating the same {pattern_len}-step sequence: "
                        f"{' → '.join(pattern)}. This is a LOOP. "
                        "Break the pattern by using a different approach."
                    ),
                    "sequence": pattern,
                    "repetitions": 2,
                }
        return None

    def get_stats(self) -> dict[str, Any]:
        """Get execution statistics."""
        if not self.history:
            return {}

        total = len(self.history)
        categories = {}
        for r in self.history:
            categories[r.category] = categories.get(r.category, 0) + 1

        return {
            "total_calls": total,
            "categories": categories,
            "effect_ratio": categories.get("effect", 0) / total,
            "observation_ratio": categories.get("observation", 0) / total,
        }
