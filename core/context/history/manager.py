# SPDX-License-Identifier: Apache-2.0
"""History — history loading, two-tier window, slim, and summary.

Extracted from ContextManager for separation of concerns.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Coroutine

from middleware.config import g_config
from utils.logger import get_logger
from utils.token_utils import count_tokens, count_tokens_messages, estimate_tokens_fast

from infra.compact.strategies.llm_emergency import LLMEmergencyStrategy
from infra.compact.config import CompactConfig
from infra.compact.utils import count_messages_tokens, build_digest, extract_key_content

logger = get_logger(__name__)

# Async callable with no args, returns a list of message dicts.
HistoryLoader = Callable[[], Coroutine[Any, Any, list[dict[str, Any]]]]


# ── Module-level helpers (extracted from ContextManager) ──────────────────────


async def _emergency_compact(
    messages: list[dict[str, Any]],
    model: str,
    summary_tokens: int,
) -> list[dict[str, Any]]:
    """调用 infra/compact 的紧急压缩。"""
    config = CompactConfig(
        model=model,
        summary_tokens=summary_tokens,
        llm_client=_LLMClientAdapterForHistory(),
    )
    strategy = LLMEmergencyStrategy(config)
    budget = CompactBudgetForHistory(
        total_tokens=count_messages_tokens(messages, config),
        total_limit=999999999,
        per_message_limit=summary_tokens,
    )
    result = await strategy.compact(messages, budget)
    return result.messages


class _CompactBudgetForHistory:
    """Minimal budget model for history module — avoids infra/compact import in this module."""

    __slots__ = ("total_tokens", "total_limit", "per_message_limit")

    def __init__(self, total_tokens: int, total_limit: int, per_message_limit: int):
        self.total_tokens = total_tokens
        self.total_limit = total_limit
        self.per_message_limit = per_message_limit


# Alias for backward compat
CompactBudgetForHistory = _CompactBudgetForHistory


class _LLMClientAdapterForHistory:
    """history.py 专用的 LLM 客户端适配器。"""

    async def chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        model: str = "",
    ) -> str:
        from middleware.llm.llm_client import chat_completions_async

        return await chat_completions_async(
            system=system,
            messages=messages,
            max_tokens=max_tokens,
            model=model,
        )


# ── History Loader ─────────────────────────────────────────────────────────────


class HistoryManager:
    """Per-session history loading and slimming."""

    def __init__(
        self,
        config: Any,
        *,
        history_loader: HistoryLoader | None = None,
        session_memory: Any = None,
        model_getter: Any = None,
    ) -> None:
        self._cfg = config
        self._loader = history_loader
        self._session_memory = session_memory
        self._model_getter = model_getter  # () -> str
        self._slim_budget = 100
        self._gateway_threshold = 0.85  # 可通过配置覆盖

    def _model(self) -> str:
        if self._model_getter:
            return self._model_getter()
        try:
            return g_config.llm.current_profile.model
        except Exception:
            return ""

    def update_slim_budget(self, budget: int) -> None:
        self._slim_budget = budget

    async def load_history(
        self,
        mode: str = "agentic",
        intent_shifted: bool = False,
    ) -> list[dict[str, Any]]:
        """从 DB 加载历史，两层窗口 + token-aware 截止。

        Args:
            mode: 模式 (agentic / direct / interrupt)
            intent_shifted: intent 是否发生转移
        """
        if not self._loader:
            return []

        raw = await self._loader()
        if not raw:
            return []

        load_limit = self._cfg.history_load_limit
        raw = raw[:load_limit]
        raw = self._slim_tool_results(raw, mark_historical=True)

        # ── Gateway 安全网：粗略估算，超 85% 强制 head+tail 截断 ──
        input_budget = g_config.llm.current_profile.input_budget
        if input_budget > 0:
            estimated_tokens = len(raw) * 150
            gateway_limit = int(input_budget * self._gateway_threshold)
            if estimated_tokens > gateway_limit:
                keep = max(2, gateway_limit // 150)
                raw = raw[:keep] + raw[-keep:]
                logger.warning(
                    "Gateway safety net triggered: {} msgs -> {} msgs "
                    "(est {} > gateway {} tokens)",
                    load_limit, len(raw), estimated_tokens, gateway_limit,
                )
        if input_budget <= 0:
            return raw

        budget = int(input_budget * self._cfg.history_budget_ratio)
        recent, earlier = self._split_by_rounds(raw, self._cfg.recent_rounds_keep)

        model = self._model()
        recent_selected: list[dict[str, Any]] = []
        recent_tokens = 0
        for msg in reversed(recent):
            t = count_tokens(str(msg.get("content", "")), model=model)
            if recent_tokens + t > budget:
                break
            recent_selected.append(msg)
            recent_tokens += t
        recent_selected.reverse()

        remaining_budget = budget - recent_tokens
        earlier_result: list[dict[str, Any]] = []

        if earlier and remaining_budget > 200:
            earlier_tokens = sum(
                count_tokens(str(m.get("content", "")), model=model)
                for m in earlier
            )
            if earlier_tokens > remaining_budget:
                compacted = await _emergency_compact(
                    earlier,
                    model=model,
                    summary_tokens=min(self._cfg.history_load_limit * 100, remaining_budget),
                )
                earlier_result = compacted
            else:
                earlier_result = earlier

        result = earlier_result + recent_selected

        # mode/intent_shifted 过滤（合并自 _select_history_for_intent）
        if mode in ("direct", "interrupt"):
            return result[-4:]
        if intent_shifted:
            return [m for m in result[-4:] if m.get("role") in {"user", "assistant"}]

        logger.info(
            "Two-tier load: {} raw -> {} earlier + {} recent = {} msgs, budget {}/{}",
            len(raw), len(earlier_result), len(recent_selected),
            len(result), recent_tokens, budget,
        )
        return result

    def build_history_summary(
        self,
        history: list[dict[str, Any]] | None,
        max_rounds: int = 3,
        max_tokens: int = 800,
    ) -> str:
        """构建简短历史摘要 (用于 intent 识别)。"""
        if not history:
            if self._session_memory and not self._session_memory.is_empty():
                content = self._session_memory.get_content()
                return extract_key_content(content, max_tokens, model=self._model())
            return "(no prior context)"

        meaningful = [
            m
            for m in history
            if m.get("role") in ("user", "assistant")
            and (str(m.get("content", "")).strip() or m.get("tool_calls"))
        ]
        if not meaningful:
            return "(no prior context)"

        candidates = meaningful[-(max_rounds * 2):]
        selected: list[str] = []
        remaining_tokens = max_tokens
        model = self._model()

        for m in reversed(candidates):
            content = str(m.get("content", "")).strip()
            tool_calls = m.get("tool_calls")

            line = f"{m['role']}: {content}"
            if tool_calls and isinstance(tool_calls, list):
                tool_names = [
                    tc.get("function", {}).get("name", "unknown")
                    if isinstance(tc, dict)
                    else "unknown"
                    for tc in tool_calls
                ]
                line += f" [called tools: {', '.join(tool_names)}]"

            tokens = count_tokens(line, model=model)
            if tokens <= remaining_tokens:
                selected.append(line)
                remaining_tokens -= tokens
            else:
                extracted = extract_key_content(line, remaining_tokens, model=model)
                if extracted:
                    selected.append(extracted)
                break

        selected.reverse()
        return "\n".join(selected) if selected else "(no prior context)"

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _split_by_rounds(
        messages: list[dict[str, Any]], keep_rounds: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """将消息按对话轮次分割为 (recent, earlier)。"""
        if not messages or keep_rounds <= 0:
            return messages, []

        round_boundaries: list[int] = []
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                round_boundaries.append(i)

        if not round_boundaries or len(round_boundaries) <= keep_rounds:
            return messages, []

        split_idx = round_boundaries[-keep_rounds]
        return messages[split_idx:], messages[:split_idx]

    def _slim_tool_results(
        self,
        messages: list[dict[str, Any]],
        *,
        mark_historical: bool = False,
    ) -> list[dict[str, Any]]:
        """规则精简 tool result 消息，零 LLM 成本。"""
        result: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") != "tool":
                result.append(msg)
                continue

            content = msg.get("content", "")
            if not content or not isinstance(content, str):
                result.append(msg)
                continue

            slim = self._slim_single_tool_result(
                content, budget=self._slim_budget,
                mark_historical=mark_historical,
            )
            slimmed = dict(msg)
            slimmed["content"] = slim
            result.append(slimmed)

        return result

    @staticmethod
    def _slim_single_tool_result(
        content: str,
        budget: int = 300,
        *,
        mark_historical: bool = False,
    ) -> str:
        """精简单条 tool result content using extract_key_content."""
        prefix = "[historical] " if mark_historical else ""

        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            if estimate_tokens_fast(content) > budget:
                return f"{prefix}{extract_key_content(content, budget)}"
            return f"{prefix}{content}" if prefix else content

        if not isinstance(parsed, dict):
            if estimate_tokens_fast(content) > budget:
                return f"{prefix}{extract_key_content(content, budget)}"
            return f"{prefix}{content}" if prefix else content

        if "skill_name" in parsed or ("summary" in parsed and "output" in parsed):
            output_val = parsed.get("output")
            if isinstance(output_val, dict):
                digest = build_digest(parsed, output_val)
                return f"{prefix}{digest}"
            skill = parsed.get("skill_name", "unknown")
            ok = parsed.get("ok")
            status = "OK" if ok else ("FAIL" if ok is False else "?")
            summary = parsed.get("summary", "")
            return f"{prefix}[{skill}: {status}] {summary}"

        results = parsed.get("results")
        if isinstance(results, list) and results:
            lines: list[str] = []
            for r in results[:10]:
                tool = r.get("tool", "unknown")
                err = r.get("error")
                if err:
                    lines.append(f"  - {tool}: FAIL — {str(err)[:80]}")
                else:
                    res_str = str(r.get("result", ""))
                    lines.append(f"  - {tool}: OK — {res_str[:80]}")
            return f"{prefix}[batch results]\n" + "\n".join(lines)

        if estimate_tokens_fast(content) > budget:
            return f"{prefix}{extract_key_content(content, budget)}"
        return f"{prefix}{content}" if prefix else content
