# SPDX-License-Identifier: Apache-2.0
"""压缩模块共享常量。

消除 magic numbers，所有默认值集中定义。
"""

from __future__ import annotations


# ── Microcompact ────────────────────────────────────────────────────────

DEFAULT_MICROCOMPACT_KEEP_RECENT: int = 5
"""Microcompact 保留最近多少个可清除 tool result 不清除。"""

DEFAULT_MICROCOMPACT_TOOLS: tuple[str, ...] = (
    "execute_skill",
    "search_skill",
    "read_file",
    "bash",
)
"""Microcompact 默认可清除工具白名单。"""


# ── Budget Guard / Emergency ─────────────────────────────────────────────

DEFAULT_EMERGENCY_KEEP_TAIL: int = 6
"""group_aware_trim 最后防线保留的尾部消息组数。"""

DEFAULT_MAX_COMPACT_FAILURES: int = 3
"""连续 compact 失败次数触发 circuit breaker。"""

DEFAULT_BREAKER_COOLDOWN_S: float = 60.0
"""circuit breaker cooldown 秒数。"""


# ── Token thresholds ────────────────────────────────────────────────────

DEFAULT_FLOOR_TOKENS: int = 200
"""落盘/提取预算的最低保障（防止极端情况）。"""

DEFAULT_SM_MIN_TOKENS: int = 800
"""SM Compact 保留消息最小 token 数。"""

DEFAULT_SM_MAX_TOKENS: int = 3000
"""SM Compact 保留消息最大 token 数。"""


# ── LLM Compact ─────────────────────────────────────────────────────────

DEFAULT_SUMMARY_TOKENS: int = 4000
"""LLM 摘要的最大 token 数（emergency compact）。"""

DEFAULT_LLM_SINGLE_SUMMARY_TOKENS: int = 800
"""单消息压缩的摘要 token 数。"""

DEFAULT_SM_LLM_UPDATE_INTERVAL: int = 5
"""每 N 个 react iteration 触发一次 LLM 摘要更新。"""


# ── PTL Retry ──────────────────────────────────────────────────────────

DEFAULT_MAX_PTL_RETRIES: int = 3
"""Prompt Too Long 重试最大次数。"""

DEFAULT_PTL_TRUNCATE_RATIO: float = 0.2
"""每次 PTL 重试丢弃的最旧消息组比例。"""


# ── Ratios ─────────────────────────────────────────────────────────────

DEFAULT_HISTORY_BUDGET_RATIO: float = 0.5
"""历史消息 token 上限 = context_window * ratio。"""

DEFAULT_SUMMARY_RATIO: float = 0.15
"""摘要输出上限 = input_budget * ratio。"""

DEFAULT_PERSIST_RATIO: float = 0.15
"""落盘阈值 = remaining_budget * ratio。"""

DEFAULT_EXTRACT_RATIO: float = 0.05
"""提取预算 = remaining_budget * ratio。"""

DEFAULT_PREVIEW_RATIO: float = 0.005
"""Pipeline preview budget = context_budget * ratio。"""

DEFAULT_SLIM_RATIO: float = 0.003
"""Pipeline slim budget = context_budget * ratio。"""

DEFAULT_SM_COMPACT_MIN_RATIO: float = 0.02
"""SM compact 保留消息最小 token = input_budget * ratio。"""

DEFAULT_SM_COMPACT_MAX_RATIO: float = 0.08
"""SM compact 保留消息最大 token = input_budget * ratio。"""
