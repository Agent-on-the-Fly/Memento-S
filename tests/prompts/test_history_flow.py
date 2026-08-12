# SPDX-License-Identifier: Apache-2.0
"""集成测试 — core/context/history/manager.py: HistoryManager 两层窗口 + slim。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.context.history.manager import HistoryManager


def _msg(role: str, content: str = "", tid: str = "", name: str = "") -> dict:
    m = {"role": role, "content": content}
    if tid:
        m["tool_call_id"] = tid
    if name:
        m["name"] = name
    return m


@pytest.fixture
def mock_cfg() -> MagicMock:
    cfg = MagicMock()
    cfg.history_load_limit = 20
    cfg.history_budget_ratio = 0.5
    cfg.recent_rounds_keep = 3
    return cfg


@pytest.fixture
def hist_mgr(mock_cfg: MagicMock) -> HistoryManager:
    return HistoryManager(config=mock_cfg)


class TestSplitByRounds:
    """验证 _split_by_rounds 静态方法。"""

    def test_empty(self, hist_mgr: HistoryManager) -> None:
        recent, earlier = hist_mgr._split_by_rounds([], 3)
        assert recent == []
        assert earlier == []

    def test_keep_rounds_zero(self, hist_mgr: HistoryManager) -> None:
        msgs = [_msg("user"), _msg("assistant")]
        recent, earlier = hist_mgr._split_by_rounds(msgs, 0)
        assert recent == msgs
        assert earlier == []

    def test_fewer_than_keep_rounds(self, hist_mgr: HistoryManager) -> None:
        msgs = [_msg("user"), _msg("assistant"), _msg("user")]
        recent, earlier = hist_mgr._split_by_rounds(msgs, 5)
        assert recent == msgs
        assert earlier == []

    def test_splits_correctly(self, hist_mgr: HistoryManager) -> None:
        msgs = [
            _msg("user"), _msg("assistant"),
            _msg("user"), _msg("assistant"),
            _msg("user"), _msg("assistant"),
        ]
        recent, earlier = hist_mgr._split_by_rounds(msgs, 2)
        assert len(recent) == 4
        assert len(earlier) == 2

    def test_non_user_roles_ignored(self, hist_mgr: HistoryManager) -> None:
        msgs = [
            _msg("system"), _msg("assistant"), _msg("tool"),
            _msg("user"), _msg("assistant"),
        ]
        recent, earlier = hist_mgr._split_by_rounds(msgs, 1)
        assert recent == msgs
        assert earlier == []


class TestSlimToolResults:
    """验证 _slim_tool_results 规则精简 tool result。"""

    def test_slim_reduces_tool_content(self, hist_mgr: HistoryManager) -> None:
        msgs = [
            _msg("user", "hi"), _msg("assistant"),
            _msg("tool", "result" * 300, tid="t1", name="read"),
        ]
        with patch("core.context.history.manager.estimate_tokens_fast", return_value=10):
            with patch("core.context.history.manager.build_digest", return_value="slimmed"):
                slim = hist_mgr._slim_tool_results(msgs)
        for orig, s in zip(msgs, slim):
            if orig.get("role") == "tool":
                assert len(s.get("content", "")) <= len(orig.get("content", ""))

    def test_slim_non_tool_pass_through(self, hist_mgr: HistoryManager) -> None:
        msgs = [_msg("user", "hello"), _msg("assistant", "hi")]
        with patch("core.context.history.manager.estimate_tokens_fast", return_value=10):
            slim = hist_mgr._slim_tool_results(msgs)
        assert slim == msgs

    def test_slim_mark_historical(self, hist_mgr: HistoryManager) -> None:
        msgs = [_msg("tool", "short", tid="t1")]
        with patch("core.context.history.manager.estimate_tokens_fast", return_value=10):
            slim = hist_mgr._slim_tool_results(msgs, mark_historical=True)
        assert "[historical]" in slim[0]["content"]


class TestBuildHistorySummary:
    """验证 build_history_summary 摘要构建。"""

    def test_empty_history(self, hist_mgr: HistoryManager) -> None:
        result = hist_mgr.build_history_summary(None)
        assert isinstance(result, str)

    def test_build_history_summary_returns_string(self, hist_mgr: HistoryManager) -> None:
        msgs = [
            _msg("user", "task 1"),
            _msg("assistant", "plan: step1, step2"),
            _msg("tool", "data" * 10, tid="t1", name="read"),
            _msg("assistant", "done step1"),
        ]
        result = hist_mgr.build_history_summary(msgs, max_tokens=500)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_rounds_limit_respected(self, hist_mgr: HistoryManager) -> None:
        msgs = [
            _msg("user", "msg1"), _msg("assistant", "msg2"),
            _msg("user", "msg3"), _msg("assistant", "msg4"),
            _msg("user", "msg5"), _msg("assistant", "msg6"),
        ]
        with patch("core.context.history.manager.count_tokens", return_value=10):
            result = hist_mgr.build_history_summary(msgs, max_rounds=2)
        assert "msg5" in result
        assert "msg6" in result
        assert "msg1" not in result


class TestLoadHistory:
    """验证 load_history 异步加载和过滤逻辑。"""

    @pytest.mark.asyncio
    async def test_no_loader_returns_empty(self, hist_mgr: HistoryManager) -> None:
        hist_mgr._loader = None
        result = await hist_mgr.load_history()
        assert result == []

    @pytest.mark.asyncio
    async def test_load_limit_applied(self, hist_mgr: HistoryManager) -> None:
        hist_mgr._cfg.history_load_limit = 2
        loader = AsyncMock(side_effect=lambda: [_msg("user", f"msg{i}") for i in range(5)])
        hist_mgr._loader = loader
        with patch("core.context.history.manager.g_config") as mock_g:
            mock_profile = MagicMock()
            mock_profile.input_budget = -1
            mock_llm = MagicMock()
            mock_llm.current_profile = mock_profile
            mock_g.llm = mock_llm
            result = await hist_mgr.load_history()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_load_history_returns_list(self, hist_mgr: HistoryManager) -> None:
        loader = AsyncMock(side_effect=lambda: [_msg("user", "hi")])
        hist_mgr._loader = loader
        with patch("core.context.history.manager.g_config") as mock_g:
            mock_profile = MagicMock()
            mock_profile.input_budget = 100000
            mock_llm = MagicMock()
            mock_llm.current_profile = mock_profile
            mock_g.llm = mock_llm
            with patch("core.context.history.manager.count_tokens", return_value=10):
                result = await hist_mgr.load_history()
        assert isinstance(result, list)
        assert all(isinstance(m, dict) for m in result)
