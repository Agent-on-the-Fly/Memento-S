# SPDX-License-Identifier: Apache-2.0
"""Compact 可观测性 — 具体实现 + 指标聚合器。

继承 infra.compact.abc.CompactObserver，提供：
  - CompactMetrics: 指标快照
  - CompactObserverImpl: 具体实现（支持日志、内存事件、回调）
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from infra.compact.abc import CompactObserver
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Metrics snapshot
# ---------------------------------------------------------------------------

@dataclass
class CompactMetrics:
    """压缩相关指标快照 — 用于 API 暴露和监控面板。"""
    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    threshold_tokens: int = 0
    context_length: int = 0
    usage_percent: float = 0.0
    compression_count: int = 0
    consecutive_failures: int = 0
    last_trigger: str | None = None
    last_saved_tokens: int = 0
    last_savings_pct: float = 0.0
    total_saved_tokens: int = 0
    last_compact_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Event log entry
# ---------------------------------------------------------------------------

@dataclass
class CompactEvent:
    """压缩事件记录。"""
    timestamp: str
    type: str  # "compact", "failure", "circuit_open", "circuit_close"
    trigger: str | None
    tokens_before: int | None = None
    tokens_after: int | None = None
    saved_tokens: int | None = None
    savings_pct: float | None = None
    consecutive_failures: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Concrete observer implementation
# ---------------------------------------------------------------------------

class CompactObserverImpl(CompactObserver):
    """压缩 Pipeline 的可观测性聚合器。

    功能:
      - 记录每次压缩的 token 使用和节省量
      - 追踪 circuit breaker 状态
      - 内存事件日志（可写入文件）
      - 暴露指标快照 (get_status)

    用法::

        observer = CompactObserverImpl(event_log_dir=session_dir / "logs")
        pipeline = CompactPipeline(config, observers=[observer])

        # 在 agent 状态端点中暴露
        status = observer.get_status()
    """

    def __init__(
        self,
        *,
        event_log_dir: Path | None = None,
        on_compact: callable | None = None,
        on_failure: callable | None = None,
    ) -> None:
        """
        Args:
            event_log_dir: 可选目录，事件日志写入该目录
            on_compact: 压缩成功时的回调 (CompactMetrics) -> None
            on_failure: 压缩失败时的回调 (str trigger, Exception) -> None
        """
        self._metrics = CompactMetrics()
        self._events: list[CompactEvent] = []
        self._event_log_dir = event_log_dir
        self._on_compact = on_compact
        self._on_failure = on_failure
        self._lock = None  # sync access if needed

    # ── CompactObserver 协议实现 ──────────────────────────────────────────

    async def on_before_compact(
        self,
        messages: list[dict[str, Any]],
        trigger: Any,
        budget: Any | None = None,
    ) -> None:
        """压缩前回调 — 记录触发原因（API 签名兼容）。"""
        trigger_val = getattr(trigger, "value", str(trigger)) if trigger else "unknown"
        logger.debug("Compaction triggered: trigger=%s", trigger_val)

    async def on_after_compact(
        self,
        result: Any,
        tokens_before: int,
        tokens_after: int,
    ) -> None:
        """压缩后回调 — 更新指标和事件日志。"""
        trigger_val = (
            getattr(result, "trigger", None)
            if result is not None
            else None
        )
        trigger_str = (
            getattr(trigger_val, "value", str(trigger_val))
            if trigger_val
            else None
        )
        saved = tokens_before - tokens_after
        savings_pct = (saved / tokens_before * 100) if tokens_before > 0 else 0.0

        # Update metrics
        self._metrics.compression_count += 1
        self._metrics.last_trigger = trigger_str
        self._metrics.last_saved_tokens = saved
        self._metrics.last_savings_pct = round(savings_pct, 1)
        self._metrics.total_saved_tokens += saved
        self._metrics.last_compact_at = datetime.now().isoformat()

        # Record event
        event = CompactEvent(
            timestamp=datetime.now().isoformat(),
            type="compact",
            trigger=trigger_str,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            saved_tokens=saved,
            savings_pct=round(savings_pct, 1),
        )
        self._append_event(event)

        # Callback
        if self._on_compact is not None:
            try:
                self._on_compact(self._metrics)
            except Exception as exc:
                logger.warning("on_compact callback failed: %s", exc)

        logger.info(
            "Compact complete: trigger=%s, saved=%d tokens (%.1f%%), total_saved=%d",
            trigger_str, saved, savings_pct, self._metrics.total_saved_tokens,
        )

    def record_response(self, usage: dict[str, Any]) -> None:
        """记录 LLM token 使用量。"""
        self._metrics.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self._metrics.last_completion_tokens = usage.get("completion_tokens", 0)

    def record_failure(self, trigger: str, exc: Exception | None = None) -> None:
        """记录压缩失败。"""
        self._metrics.consecutive_failures += 1
        event = CompactEvent(
            timestamp=datetime.now().isoformat(),
            type="failure",
            trigger=trigger,
            consecutive_failures=self._metrics.consecutive_failures,
            metadata={"error": str(exc)} if exc else {},
        )
        self._append_event(event)

        if self._on_failure is not None:
            try:
                self._on_failure(trigger, exc)
            except Exception:
                pass

        logger.warning(
            "Compact failed: trigger=%s, consecutive_failures=%d",
            trigger, self._metrics.consecutive_failures,
        )

    def record_circuit_open(self, failures: int) -> None:
        """记录 circuit breaker 打开。"""
        event = CompactEvent(
            timestamp=datetime.now().isoformat(),
            type="circuit_open",
            consecutive_failures=failures,
        )
        self._append_event(event)
        logger.warning("Circuit breaker OPENED after %d failures", failures)

    def record_circuit_close(self) -> None:
        """记录 circuit breaker 关闭。"""
        event = CompactEvent(
            timestamp=datetime.now().isoformat(),
            type="circuit_close",
            consecutive_failures=0,
        )
        self._append_event(event)
        logger.info("Circuit breaker CLOSED")

    # ── Metrics exposure ──────────────────────────────────────────────────

    def update_threshold(self, threshold_tokens: int, context_length: int) -> None:
        """更新阈值（由 Pipeline 每次 run 时调用）。"""
        self._metrics.threshold_tokens = threshold_tokens
        self._metrics.context_length = context_length
        if context_length > 0 and self._metrics.last_prompt_tokens > 0:
            self._metrics.usage_percent = min(
                100.0,
                self._metrics.last_prompt_tokens / context_length * 100,
            )

    def get_status(self) -> dict[str, Any]:
        """返回指标快照字典 — 暴露给 agent 状态 API 或 CLI。"""
        return {
            "last_prompt_tokens": self._metrics.last_prompt_tokens,
            "last_completion_tokens": self._metrics.last_completion_tokens,
            "threshold_tokens": self._metrics.threshold_tokens,
            "context_length": self._metrics.context_length,
            "usage_percent": round(self._metrics.usage_percent, 1),
            "compression_count": self._metrics.compression_count,
            "consecutive_failures": self._metrics.consecutive_failures,
            "last_trigger": self._metrics.last_trigger,
            "last_saved_tokens": self._metrics.last_saved_tokens,
            "last_savings_pct": self._metrics.last_savings_pct,
            "total_saved_tokens": self._metrics.total_saved_tokens,
            "last_compact_at": self._metrics.last_compact_at,
        }

    def get_metrics(self) -> CompactMetrics:
        """返回指标对象。"""
        return self._metrics

    def get_events(self, limit: int = 50) -> list[dict[str, Any]]:
        """返回最近 N 条事件。"""
        events = self._events[-limit:]
        return [asdict(e) for e in events]

    def get_event_log_path(self) -> Path | None:
        """返回事件日志文件路径（如已配置）。"""
        if self._event_log_dir is None:
            return None
        return self._event_log_dir / "compact_events.jsonl"

    # ── Internal helpers ─────────────────────────────────────────────────

    def _append_event(self, event: CompactEvent) -> None:
        self._events.append(event)
        # Persist to file if configured
        if self._event_log_dir is not None:
            try:
                self._event_log_dir.mkdir(parents=True, exist_ok=True)
                log_path = self._event_log_dir / "compact_events.jsonl"
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
            except Exception as exc:
                logger.warning("Failed to write compact event log: %s", exc)

    def reset(self) -> None:
        """重置所有指标和事件日志（调用在 /new 或 /reset）。"""
        self._metrics = CompactMetrics()
        self._events.clear()
        logger.debug("CompactObserverImpl metrics reset")
