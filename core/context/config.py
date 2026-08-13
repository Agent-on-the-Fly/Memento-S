# SPDX-License-Identifier: Apache-2.0
"""Context 模块配置。

ContextConfig 是 ContextManager 的唯一配置依赖。
所有 token 预算均为 input_budget 的 **比例**，在 init_budget() 中动态计算。

input_budget = LLMProfile.context_window - LLMProfile.max_tokens
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ContextManagerConfig:
    """ContextManager 可配置参数。

    规则: 不出现任何固定 token 数字。token 相关全部为 ratio，
    运行时 * input_budget 得到实际值。绝对下限由代码常量兜底。
    """

    # ── History Loading ──
    history_load_limit: int = 20
    """load_history() 从 DB 读取的最大条数。"""

    recent_rounds_keep: int = 3
    """两层窗口中保留原文的最近对话轮数。"""

    history_budget_ratio: float = 0.5
    """历史占 input_budget 的上限比例。"""

    summary_ratio: float = 0.15
    """compress/compact 摘要输出上限 = input_budget * ratio。"""

    compaction_trigger_ratio: float = 0.7
    """Compatibility threshold for eager append-time compaction."""

    compress_threshold_ratio: float = 0.5
    """Compatibility threshold for single-message compression."""

    embedding_enabled: bool = False
    """是否启用 embedding 语义相关性过滤（暂不支持）。"""

    # ── Artifact Store (全部 ratio) ──
    persist_ratio: float = 0.15
    """persist 阈值 = remaining_budget * ratio。"""

    extract_ratio: float = 0.05
    """extract_key_content budget = remaining_budget * ratio。"""

    # ── Pipeline / Slim (ratio of input_budget) ──
    preview_ratio: float = 0.005
    """pipeline preview 占 input_budget 的比例。"""

    slim_ratio: float = 0.003
    """历史 tool result 精简 budget 占 input_budget 的比例。"""

    # ── Microcompact ──
    microcompact_keep_recent: int = 5
    """microcompact 保留最近多少个 compactable tool result。"""

    microcompact_compactable_tools: list[str] = field(
        default_factory=lambda: [
            "execute_skill",
            "search_skill",
        ]
    )
    """可被 microcompact 清除的工具名白名单。"""

    # ── Budget Guard ──
    emergency_keep_tail: int = 6
    """group_aware_trim 最后防线保留的尾部消息组数。"""

    max_compact_failures: int = 3
    """连续 compact 失败次数触发 circuit breaker。"""

    sm_compact_min_ratio: float = 0.02
    """SM compact 保留消息最小 token = input_budget * ratio。"""

    sm_compact_max_ratio: float = 0.08
    """SM compact 保留消息最大 token = input_budget * ratio。"""

    breaker_cooldown_s: float = 60.0
    """circuit breaker cooldown 秒数。"""

    # ── L1 Session Memory ──
    sm_llm_update_interval: int = 5
    """每 N 个 react iteration 触发一次 session memory LLM 更新 (step boundary 保底)。"""

    max_entrypoint_lines: int = 200
    """MEMORY.md 索引最大行数 (CC 参考值，非 token 阈值)。"""

    max_entrypoint_bytes: int = 25_000
    """MEMORY.md 索引最大字节数 (CC 参考值，非 token 阈值)。"""

    def __post_init__(self) -> None:
        _ratio_fields = (
            "history_budget_ratio", "summary_ratio",
            "compaction_trigger_ratio", "compress_threshold_ratio",
            "persist_ratio", "extract_ratio",
            "preview_ratio", "slim_ratio",
            "sm_compact_min_ratio", "sm_compact_max_ratio",
        )
        for name in _ratio_fields:
            val = getattr(self, name)
            if not (0.0 < val < 1.0):
                raise ValueError(f"{name} must be in (0, 1), got {val}")
        if self.sm_compact_min_ratio > self.sm_compact_max_ratio:
            raise ValueError(
                f"sm_compact_min_ratio ({self.sm_compact_min_ratio}) "
                f"> sm_compact_max_ratio ({self.sm_compact_max_ratio})"
            )
        if self.history_load_limit < 1:
            raise ValueError(
                f"history_load_limit must be >= 1, got {self.history_load_limit}"
            )
        if self.recent_rounds_keep < 0:
            raise ValueError(
                f"recent_rounds_keep must be >= 0, got {self.recent_rounds_keep}"
            )
