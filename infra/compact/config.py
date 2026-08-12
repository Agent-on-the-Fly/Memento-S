# SPDX-License-Identifier: Apache-2.0
"""配置模型 — 压缩模块的所有可配置参数。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from infra.compact import constants as _c
from infra.compact.abc import CompactObserver, LLMClient, StorageBackend, SummaryReader, TokenCounter

if TYPE_CHECKING:
    from infra.compact.models import CompactBudget, CompactResult, CompactTrigger


@dataclass
class CompactConfig:
    """压缩模块配置 — 所有参数均为比例或无业务依赖的纯数字。

    使用比例而非绝对数字，确保配置可迁移到不同 context_window 的模型。
    """

    # ── 模块开关 ──────────────────────────────────────────────────────
    enabled: bool = True
    """是否启用压缩模块。"""

    # ── Microcompact (零 LLM) ─────────────────────────────────────────
    microcompact_keep_recent: int = _c.DEFAULT_MICROCOMPACT_KEEP_RECENT
    """保留最近多少个 compactable tool result 不清除。"""

    microcompact_compactable_tools: list[str] = field(
        default_factory=lambda: list(_c.DEFAULT_MICROCOMPACT_TOOLS)
    )
    """可被 microcompact 清除的工具名白名单。"""

    # ── Budget Guard ──────────────────────────────────────────────────
    emergency_keep_tail: int = _c.DEFAULT_EMERGENCY_KEEP_TAIL
    """group_aware_trim 最后防线保留的尾部消息组数。"""

    max_compact_failures: int = _c.DEFAULT_MAX_COMPACT_FAILURES
    """连续 compact 失败次数触发 circuit breaker。"""

    breaker_cooldown_s: float = _c.DEFAULT_BREAKER_COOLDOWN_S
    """circuit breaker cooldown 秒数。"""

    # ── SM Compact (零 LLM) ───────────────────────────────────────────
    sm_compact_min_ratio: float = _c.DEFAULT_SM_COMPACT_MIN_RATIO
    """SM compact 保留消息最小 token = input_budget * ratio。"""

    sm_compact_max_ratio: float = _c.DEFAULT_SM_COMPACT_MAX_RATIO
    """SM compact 保留消息最大 token = input_budget * ratio。"""

    # ── LLM Compact ───────────────────────────────────────────────────
    summary_tokens: int = _c.DEFAULT_SUMMARY_TOKENS
    """LLM 摘要的最大 token 数。"""

    summary_ratio: float = _c.DEFAULT_SUMMARY_RATIO
    """摘要输出上限 = input_budget * ratio (用于计算 summary_tokens)。"""

    llm_single_summary_tokens: int = _c.DEFAULT_LLM_SINGLE_SUMMARY_TOKENS
    """单消息压缩的摘要 token 数。"""

    # ── PTL Retry ────────────────────────────────────────────────────
    max_ptl_retries: int = _c.DEFAULT_MAX_PTL_RETRIES
    """Prompt Too Long 重试最大次数。"""

    ptl_truncate_ratio: float = _c.DEFAULT_PTL_TRUNCATE_RATIO
    """每次 PTL 重试丢弃的最旧消息组比例。"""

    # ── Dependencies (依赖注入) ───────────────────────────────────────
    llm_client: LLMClient | None = None
    """LLM 客户端 (实现 LLMClient 协议)。"""

    token_counter: TokenCounter | None = None
    """Token 计数器 (实现 TokenCounter 协议)。"""

    storage_backend: StorageBackend | None = None
    """存储后端 (实现 StorageBackend 协议)。"""

    summary_reader: SummaryReader | None = None
    """Summary 读取器 (实现 SummaryReader 协议，用于 SM Compact)。"""

    # ── 模型配置 ──────────────────────────────────────────────────────
    model: str = ""
    """默认模型名称。"""

    # ── 回调 ──────────────────────────────────────────────────────────
    observers: list[CompactObserver] = field(default_factory=list)
    """压缩回调列表 (实现 CompactObserver 协议)。"""

    def __post_init__(self) -> None:
        """验证配置合法性。"""
        if self.sm_compact_min_ratio <= 0 or self.sm_compact_min_ratio >= 1:
            raise ValueError(
                f"sm_compact_min_ratio must be in (0, 1), got {self.sm_compact_min_ratio}"
            )
        if self.sm_compact_max_ratio <= 0 or self.sm_compact_max_ratio >= 1:
            raise ValueError(
                f"sm_compact_max_ratio must be in (0, 1), got {self.sm_compact_max_ratio}"
            )
        if self.sm_compact_min_ratio > self.sm_compact_max_ratio:
            raise ValueError(
                f"sm_compact_min_ratio ({self.sm_compact_min_ratio}) > "
                f"sm_compact_max_ratio ({self.sm_compact_max_ratio})"
            )
        if self.max_compact_failures < 0:
            raise ValueError(
                f"max_compact_failures must be >= 0, got {self.max_compact_failures}"
            )
        if self.breaker_cooldown_s < 0:
            raise ValueError(
                f"breaker_cooldown_s must be >= 0, got {self.breaker_cooldown_s}"
            )

    def calculate_budget(self, input_budget: int) -> dict[str, int]:
        """基于 input_budget 动态计算实际阈值。

        Args:
            input_budget: 可用上下文预算 (通常为 context_window - max_tokens)

        Returns:
            包含各阈值计算的字典
        """
        return {
            "total_limit": input_budget,
            "per_message_limit": input_budget // 4,
            "sm_min_tokens": max(100, int(input_budget * self.sm_compact_min_ratio)),
            "sm_max_tokens": max(500, int(input_budget * self.sm_compact_max_ratio)),
            "summary_tokens": min(
                self.summary_tokens,
                max(500, int(input_budget * self.summary_ratio)),
            ),
        }

    def with_overrides(self, **kwargs: Any) -> "CompactConfig":
        """创建配置副本并覆盖指定字段。"""
        import dataclasses

        new_config = dataclasses.replace(self, **kwargs)
        return new_config
