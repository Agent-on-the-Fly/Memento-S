# SPDX-License-Identifier: Apache-2.0
"""单元测试 — core/context/history.py: HistoryManager & helpers."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.context.history import HistoryManager
from core.context.history.manager import CompactBudgetForHistory, _CompactBudgetForHistory


# ── Session fixture: load g_config so g_config.__getattr__ doesn't raise ────

@pytest.fixture(scope="session", autouse=True)
def _ensure_config():
    from middleware.config import g_config
    if not g_config.is_loaded():
        g_config.load()


# ── Test helpers ──────────────────────────────────────────────────────────────

def _make_loader(raw: list):
    """Build an async mock loader returning the given messages."""
    async def loader():
        return raw
    return AsyncMock(side_effect=loader)


def _mock_g_config(input_budget: int) -> MagicMock:
    """Build a minimal mock g_config with a given input_budget."""
    mock_profile = MagicMock()
    mock_profile.input_budget = input_budget
    mock_llm = MagicMock()
    mock_llm.current_profile = mock_profile
    mock_g = MagicMock()
    mock_g.llm = mock_llm
    return mock_g


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_cfg() -> MagicMock:
    """Minimal config mock for HistoryManager."""
    cfg = MagicMock()
    cfg.history_load_limit = 20
    cfg.history_budget_ratio = 0.5
    cfg.recent_rounds_keep = 3
    return cfg


@pytest.fixture
def hm(mock_cfg: MagicMock) -> HistoryManager:
    """HistoryManager with a mock config and no loader."""
    return HistoryManager(config=mock_cfg)


# ── _CompactBudgetForHistory ─────────────────────────────────────────────────

class TestCompactBudgetForHistory:
    def test_attributes(self):
        budget = _CompactBudgetForHistory(total_tokens=5000, total_limit=80000, per_message_limit=200)
        assert budget.total_tokens == 5000
        assert budget.total_limit == 80000
        assert budget.per_message_limit == 200

    def test_alias(self):
        assert CompactBudgetForHistory is _CompactBudgetForHistory


# ── HistoryManager._model ────────────────────────────────────────────────────

class TestHistoryManagerModel:
    def test_model_getter_returns_value(self, hm: HistoryManager):
        hm._model_getter = lambda: "gpt-4o"
        assert hm._model() == "gpt-4o"

    def test_model_getter_returns_empty(self, hm: HistoryManager):
        hm._model_getter = lambda: ""
        assert hm._model() == ""

    def test_model_no_getter_uses_g_config(self, hm: HistoryManager):
        mock_profile = MagicMock()
        mock_profile.model = "claude-3"
        mock_llm = MagicMock()
        mock_llm.current_profile = mock_profile
        mock_g = MagicMock()
        mock_g.llm = mock_llm
        with patch("core.context.history.manager.g_config", mock_g):
            hm._model_getter = None
            assert hm._model() == "claude-3"

    def test_model_g_config_raises(self, hm: HistoryManager):
        """RuntimeError from current_profile is caught, returns empty string."""
        with patch("core.context.history.manager.g_config") as mock_g:
            type(mock_g).llm = property(lambda *_: (_ for _ in ()).throw(RuntimeError("no profile")))
            hm._model_getter = None
            assert hm._model() == ""


# ── HistoryManager.update_slim_budget ────────────────────────────────────────

class TestUpdateSlimBudget:
    def test_set_positive(self, hm: HistoryManager):
        assert hm._slim_budget == 100
        hm.update_slim_budget(200)
        assert hm._slim_budget == 200

    def test_set_zero(self, hm: HistoryManager):
        hm.update_slim_budget(0)
        assert hm._slim_budget == 0


# ── HistoryManager._split_by_rounds ──────────────────────────────────────────

class TestSplitByRounds:
    """Test _split_by_rounds static method.

    Logic: round_boundaries = indices of user-role messages.
    If len(boundaries) <= keep_rounds: no split needed (all recent).
    Otherwise split_idx = round_boundaries[-(keep_rounds)], placing
    everything from the (k-1)-th-from-last user round onward into recent.
    """

    def _msg(self, role: str, content: str = "x") -> dict:
        return {"role": role, "content": content}

    def test_empty(self, hm: HistoryManager):
        recent, earlier = hm._split_by_rounds([], 3)
        assert recent == []
        assert earlier == []

    def test_keep_rounds_zero(self, hm: HistoryManager):
        msgs = [self._msg("user"), self._msg("assistant")]
        recent, earlier = hm._split_by_rounds(msgs, 0)
        assert recent == msgs
        assert earlier == []

    def test_fewer_than_keep_rounds(self, hm: HistoryManager):
        msgs = [self._msg("user"), self._msg("assistant"), self._msg("user")]
        recent, earlier = hm._split_by_rounds(msgs, 5)
        assert recent == msgs
        assert earlier == []

    def test_exact_keep_rounds(self, hm: HistoryManager):
        msgs = [self._msg("user"), self._msg("assistant"), self._msg("user"), self._msg("assistant")]
        recent, earlier = hm._split_by_rounds(msgs, 2)
        assert recent == msgs
        assert earlier == []

    def test_splits_correctly(self, hm: HistoryManager):
        """keep_rounds=2 with 3 rounds: recent = last 2 rounds, earlier = first round."""
        msgs = [
            self._msg("user"),           # 0  ← round 0 boundary
            self._msg("assistant"),      # 1
            self._msg("user"),           # 2  ← round 1 boundary
            self._msg("assistant"),      # 3
            self._msg("user"),           # 4  ← round 2 boundary
            self._msg("assistant"),      # 5
        ]
        # round_boundaries = [0, 2, 4]; len=3 > keep_rounds=2
        # split_idx = round_boundaries[-2] = 2
        # recent = msgs[2:], earlier = msgs[:2]
        recent, earlier = hm._split_by_rounds(msgs, 2)
        assert recent == msgs[2:]
        assert earlier == msgs[:2]
        assert len(recent) == 4
        assert len(earlier) == 2

    def test_keep_rounds_one(self, hm: HistoryManager):
        """keep_rounds=1: recent = last round only."""
        msgs = [
            self._msg("user"),           # 0
            self._msg("assistant"),      # 1
            self._msg("user"),           # 2
            self._msg("assistant"),      # 3
        ]
        recent, earlier = hm._split_by_rounds(msgs, 1)
        assert recent == [msgs[2], msgs[3]]
        assert earlier == [msgs[0], msgs[1]]

    def test_non_user_roles_ignored_for_boundary(self, hm: HistoryManager):
        """Only user-role messages define round boundaries; others are skipped."""
        msgs = [
            self._msg("system"),
            self._msg("assistant"),
            self._msg("tool"),
            self._msg("user"),           # 3
            self._msg("assistant"),      # 4
        ]
        # round_boundaries = [3]; len=1 <= keep_rounds=1 → no split
        recent, earlier = hm._split_by_rounds(msgs, 1)
        assert recent == msgs
        assert earlier == []

    def test_multiple_users_at_start(self, hm: HistoryManager):
        msgs = [
            self._msg("user"), self._msg("user"), self._msg("user"),
            self._msg("assistant"),
        ]
        # round_boundaries = [0, 1, 2]; split_idx = round_boundaries[-2] = 1
        recent, earlier = hm._split_by_rounds(msgs, 2)
        assert recent == [msgs[1], msgs[2], msgs[3]]
        assert earlier == [msgs[0]]


# ── HistoryManager._slim_single_tool_result ───────────────────────────────────

class TestSlimSingleToolResult:
    def _slim(self, content: str, budget: int = 300, mark_historical: bool = False) -> str:
        return HistoryManager._slim_single_tool_result(content, budget=budget, mark_historical=mark_historical)

    def test_short_plain_text(self, hm: HistoryManager):
        with patch("core.context.history.manager.estimate_tokens_fast", return_value=10):
            result = self._slim("hello world", budget=300)
        assert result == "hello world"

    def test_long_plain_text_truncated(self, hm: HistoryManager):
        with patch("core.context.history.manager.estimate_tokens_fast", return_value=500):
            with patch("core.context.history.manager.extract_key_content", return_value="shortened...") as mock_extract:
                result = self._slim("x" * 1000, budget=300)
        assert result == "shortened..."
        mock_extract.assert_called_once()

    def test_long_plain_text_truncated_no_prefix(self, hm: HistoryManager):
        with patch("core.context.history.manager.estimate_tokens_fast", return_value=500):
            with patch("core.context.history.manager.extract_key_content", return_value="trunc"):
                result = self._slim("x" * 1000, budget=300)
        assert result == "trunc"

    def test_json_skill_with_summary_output(self, hm: HistoryManager):
        with patch("core.context.history.manager.estimate_tokens_fast", return_value=10):
            with patch("core.context.history.manager.build_digest", return_value="[my_skill: OK] did stuff") as mock_digest:
                content = '{"skill_name": "my_skill", "summary": "did stuff", "output": {"result": "foo"}}'
                result = self._slim(content)
        mock_digest.assert_called_once()
        assert result == "[my_skill: OK] did stuff"

    def test_json_skill_ok_true(self, hm: HistoryManager):
        with patch("core.context.history.manager.estimate_tokens_fast", return_value=10):
            with patch("core.context.history.manager.build_digest", return_value="[bash_tool: OK] ls succeeded"):
                content = '{"skill_name": "bash_tool", "ok": true, "summary": "ls succeeded"}'
                result = self._slim(content)
        assert result == "[bash_tool: OK] ls succeeded"

    def test_json_skill_ok_false(self, hm: HistoryManager):
        with patch("core.context.history.manager.estimate_tokens_fast", return_value=10):
            with patch("core.context.history.manager.build_digest", return_value="[bash_tool: FAIL] ls failed"):
                content = '{"skill_name": "bash_tool", "ok": false, "summary": "ls failed"}'
                result = self._slim(content)
        assert result == "[bash_tool: FAIL] ls failed"

    def test_json_results_list(self, hm: HistoryManager):
        with patch("core.context.history.manager.estimate_tokens_fast", return_value=10):
            content = '{"results": [{"tool": "read", "result": "file contents here"}]}'
            result = self._slim(content)
        assert "[batch results]" in result
        assert "read" in result
        assert "file contents here" in result

    def test_json_results_list_truncates_to_10(self, hm: HistoryManager):
        with patch("core.context.history.manager.estimate_tokens_fast", return_value=10):
            results = [{"tool": f"tool_{i}", "result": f"result_{i}"} for i in range(20)]
            content = '{"results": ' + __import__("json").dumps(results) + "}"
            result = self._slim(content)
        assert "[batch results]" in result
        assert "tool_0" in result
        assert "tool_9" in result
        assert "tool_10" not in result

    def test_json_results_error_field(self, hm: HistoryManager):
        with patch("core.context.history.manager.estimate_tokens_fast", return_value=10):
            content = '{"results": [{"tool": "bash", "error": "permission denied"}]}'
            result = self._slim(content)
        assert "bash: FAIL" in result

    def test_non_dict_json(self, hm: HistoryManager):
        with patch("core.context.history.manager.estimate_tokens_fast", return_value=10):
            result = self._slim('[1, 2, 3]')
        assert result == '[1, 2, 3]'

    def test_non_dict_json_long(self, hm: HistoryManager):
        with patch("core.context.history.manager.estimate_tokens_fast", return_value=500):
            with patch("core.context.history.manager.extract_key_content", return_value="truncated"):
                result = self._slim('{"items": ["a"] * 100}')
        assert result == "truncated"

    def test_empty_string(self, hm: HistoryManager):
        result = self._slim("")
        assert result == ""

    def test_mark_historical_prefix_plain(self, hm: HistoryManager):
        with patch("core.context.history.manager.estimate_tokens_fast", return_value=10):
            result = self._slim("short", mark_historical=True)
        assert result.startswith("[historical] ")

    def test_mark_historical_prefix_json(self, hm: HistoryManager):
        with patch("core.context.history.manager.estimate_tokens_fast", return_value=10):
            with patch("core.context.history.manager.build_digest", return_value="[historical] s"):
                result = self._slim('{"skill_name": "s", "summary": "x", "output": {}}', mark_historical=True)
        assert result.startswith("[historical] ")


# ── HistoryManager._slim_tool_results ────────────────────────────────────────

class TestSlimToolResults:
    def _msg(self, role: str, content: str = "", **kwargs) -> dict:
        d = {"role": role, "content": content}
        d.update(kwargs)
        return d

    def test_non_tool_pass_through(self, hm: HistoryManager):
        with patch("core.context.history.manager.estimate_tokens_fast", return_value=10):
            msgs = [self._msg("user", "hello"), self._msg("assistant", "hi")]
            result = hm._slim_tool_results(msgs)
        assert result == msgs

    def test_tool_result_slimmed(self, hm: HistoryManager):
        with patch("core.context.history.manager.estimate_tokens_fast", return_value=500):
            with patch("core.context.history.manager.build_digest", return_value="slimmed"):
                msgs = [self._msg("tool", '{"skill_name": "s", "summary": "x", "output": {}}')]
                result = hm._slim_tool_results(msgs)
        assert result[0]["content"] == "slimmed"
        assert result[0]["role"] == "tool"

    def test_tool_result_mark_historical(self, hm: HistoryManager):
        with patch("core.context.history.manager.estimate_tokens_fast", return_value=10):
            msgs = [self._msg("tool", "hello")]
            result = hm._slim_tool_results(msgs, mark_historical=True)
        assert result[0]["content"] == "[historical] hello"

    def test_tool_empty_content_pass(self, hm: HistoryManager):
        msgs = [self._msg("tool", "")]
        result = hm._slim_tool_results(msgs)
        assert result[0]["content"] == ""

    def test_tool_non_string_content_pass(self, hm: HistoryManager):
        msgs = [self._msg("tool", 12345)]  # type: ignore
        result = hm._slim_tool_results(msgs)
        assert result[0]["content"] == 12345

    def test_does_not_mutate_original(self, hm: HistoryManager):
        original = self._msg("tool", '{"skill_name": "s", "summary": "x", "output": {}}')
        with patch("core.context.history.manager.estimate_tokens_fast", return_value=10):
            with patch("core.context.history.manager.build_digest", return_value="dig"):
                result = hm._slim_tool_results([original])
        assert original["content"] == '{"skill_name": "s", "summary": "x", "output": {}}'
        assert result[0]["content"] == "dig"


# ── HistoryManager.build_history_summary ─────────────────────────────────────

class TestBuildHistorySummary:
    def _msg(self, role: str, content: str = "", tool_calls: list | None = None) -> dict:
        d = {"role": role, "content": content}
        if tool_calls:
            d["tool_calls"] = tool_calls
        return d

    def test_empty_history_no_memory(self, hm: HistoryManager):
        result = hm.build_history_summary(None)
        assert result == "(no prior context)"

    def test_empty_history_empty_list(self, hm: HistoryManager):
        result = hm.build_history_summary([])
        assert result == "(no prior context)"

    def test_empty_meaningful_messages(self, hm: HistoryManager):
        result = hm.build_history_summary([{"role": "system", "content": ""}])
        assert result == "(no prior context)"

    def test_uses_session_memory_when_history_empty(self, hm: HistoryManager):
        with patch("core.context.history.manager.count_tokens", return_value=50):
            with patch("core.context.history.manager.extract_key_content", return_value="extracted") as mock_extract:
                mock_mem = MagicMock()
                mock_mem.is_empty.return_value = False
                mock_mem.get_content.return_value = "memory content here"
                hm._session_memory = mock_mem
                with patch.object(hm, "_model", return_value="gpt-4o"):
                    result = hm.build_history_summary(None)
        assert result == "extracted"
        mock_extract.assert_called_once()

    def test_session_memory_prefers_history(self, hm: HistoryManager):
        with patch("core.context.history.manager.count_tokens", return_value=10):
            mock_mem = MagicMock()
            mock_mem.is_empty.return_value = False
            mock_mem.get_content.return_value = "memory content"
            hm._session_memory = mock_mem
            history = [self._msg("user", "hello"), self._msg("assistant", "hi")]
            with patch.object(hm, "_model", return_value="gpt-4o"):
                result = hm.build_history_summary(history)
        assert result == "user: hello\nassistant: hi"
        mock_mem.get_content.assert_not_called()

    def test_rounds_limit(self, hm: HistoryManager):
        with patch("core.context.history.manager.count_tokens", return_value=10):
            history = [
                self._msg("user", "msg1"), self._msg("assistant", "msg2"),
                self._msg("user", "msg3"), self._msg("assistant", "msg4"),
                self._msg("user", "msg5"), self._msg("assistant", "msg6"),
            ]
            with patch.object(hm, "_model", return_value=""):
                result = hm.build_history_summary(history, max_rounds=2)
        assert "msg5" in result
        assert "msg6" in result
        assert "msg1" not in result

    def test_token_budget_respected(self, hm: HistoryManager):
        with patch("core.context.history.manager.count_tokens", return_value=10):
            history = [self._msg("user", "hello world")]
            with patch.object(hm, "_model", return_value="gpt-4o"):
                result = hm.build_history_summary(history, max_tokens=50)
        assert "hello world" in result

    def test_token_budget_exceeded_triggers_extract(self, hm: HistoryManager):
        with patch("core.context.history.manager.count_tokens", side_effect=[200, 10]):
            with patch("core.context.history.manager.extract_key_content", return_value="shortened"):
                history = [self._msg("user", "x" * 1000)]
                with patch.object(hm, "_model", return_value="gpt-4o"):
                    result = hm.build_history_summary(history, max_tokens=50)
        assert result == "shortened"

    def test_tool_calls_appended(self, hm: HistoryManager):
        with patch("core.context.history.manager.count_tokens", return_value=10):
            history = [
                self._msg("user", "list files",
                          tool_calls=[{"function": {"name": "bash", "arguments": "{}"}}])
            ]
            with patch.object(hm, "_model", return_value=""):
                result = hm.build_history_summary(history)
        assert "called tools: bash" in result

    def test_tool_calls_unknown_function(self, hm: HistoryManager):
        with patch("core.context.history.manager.count_tokens", return_value=10):
            history = [{"role": "user", "content": "call", "tool_calls": [{"function": {}}]}]
            with patch.object(hm, "_model", return_value=""):
                result = hm.build_history_summary(history)
        assert "called tools: unknown" in result

    def test_tool_calls_non_dict_item(self, hm: HistoryManager):
        with patch("core.context.history.manager.count_tokens", return_value=10):
            history = [{"role": "user", "content": "call", "tool_calls": ["not_a_dict"]}]
            with patch.object(hm, "_model", return_value=""):
                result = hm.build_history_summary(history)
        assert "called tools: unknown" in result

    def test_empty_selected_returns_fallback(self, hm: HistoryManager):
        with patch("core.context.history.manager.count_tokens", return_value=99999):
            with patch("core.context.history.manager.extract_key_content", return_value=""):
                history = [self._msg("user", "x" * 1000)]
                with patch.object(hm, "_model", return_value=""):
                    result = hm.build_history_summary(history, max_tokens=50)
        assert result == "(no prior context)"


# ── HistoryManager.load_history ───────────────────────────────────────────────

class TestLoadHistory:
    def _msg(self, role: str, content: str = "") -> dict:
        return {"role": role, "content": content}

    @pytest.mark.asyncio
    async def test_no_loader_returns_empty(self, hm: HistoryManager):
        hm._loader = None
        result = await hm.load_history()
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_raw_returns_empty(self, hm: HistoryManager):
        mock_g = _mock_g_config(-1)
        with patch("core.context.history.manager.g_config", mock_g):
            loader = _make_loader([])
            hm._loader = loader
            result = await hm.load_history()
        assert result == []

    @pytest.mark.asyncio
    async def test_load_limit_applied(self, hm: HistoryManager, mock_cfg: MagicMock):
        mock_g = _mock_g_config(-1)  # <= 0 → skip token logic
        with patch("core.context.history.manager.g_config", mock_g):
            mock_cfg.history_load_limit = 2
            msgs = [self._msg("user", f"msg{i}") for i in range(5)]
            loader = _make_loader(msgs)
            hm._loader = loader
            result = await hm.load_history()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_within_budget_no_split(self, hm: HistoryManager, mock_cfg: MagicMock):
        mock_g = _mock_g_config(100000)
        mock_cfg.history_budget_ratio = 0.5
        mock_cfg.recent_rounds_keep = 3
        with patch("core.context.history.manager.g_config", mock_g):
            with patch("core.context.history.manager.count_tokens", return_value=10):
                loader = _make_loader([self._msg("user", "hi"), self._msg("assistant", "hi")])
                hm._loader = loader
                result = await hm.load_history()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_fewer_than_keep_rounds(self, hm: HistoryManager, mock_cfg: MagicMock):
        mock_g = _mock_g_config(100000)
        mock_cfg.history_budget_ratio = 0.5
        mock_cfg.recent_rounds_keep = 5
        with patch("core.context.history.manager.g_config", mock_g):
            with patch("core.context.history.manager.count_tokens", return_value=10):
                loader = _make_loader([self._msg("user", "a"), self._msg("assistant", "b")])
                hm._loader = loader
                result = await hm.load_history()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_earlier_exceeds_budget_triggers_compact(self, hm: HistoryManager, mock_cfg: MagicMock):
        mock_g = _mock_g_config(100000)
        mock_cfg.history_budget_ratio = 0.5
        mock_cfg.recent_rounds_keep = 1
        mock_cfg.history_load_limit = 20
        compact_result = [self._msg("user", "compacted summary")]
        compact_mock = AsyncMock(return_value=compact_result)

        # count_tokens: first 4 = 10 each (earlier), last = 10 (recent)
        # budget = 50000, recent_tokens = 10, remaining = 49990 > 200 ✓
        # earlier_tokens = 40 < 49990 → no compact  ✗
        # Make earlier tokens large via side_effect: first 4 return 20000, last returns 10
        large_token_count = [20000, 20000, 20000, 20000, 10]

        with patch("core.context.history.manager.g_config", mock_g):
            with patch("core.context.history.manager.count_tokens", side_effect=large_token_count):
                with patch("core.context.history.manager._emergency_compact", compact_mock):
                    # 5 rounds: keep_rounds=1 → recent=[4], earlier=[0,1,2,3]
                    msgs = [
                        self._msg("user", "old1"),
                        self._msg("assistant", "old2"),
                        self._msg("user", "old3"),
                        self._msg("assistant", "old4"),
                        self._msg("user", "recent"),
                    ]
                    loader = _make_loader(msgs)
                    hm._loader = loader
                    result = await hm.load_history()

        compact_mock.assert_called_once()
        assert result[0]["content"] == "compacted summary"
        assert any(m["content"] == "recent" for m in result)

    @pytest.mark.asyncio
    async def test_direct_mode_returns_last_4(self, hm: HistoryManager, mock_cfg: MagicMock):
        mock_g = _mock_g_config(100000)
        mock_cfg.history_budget_ratio = 0.5
        mock_cfg.recent_rounds_keep = 3
        with patch("core.context.history.manager.g_config", mock_g):
            with patch("core.context.history.manager.count_tokens", return_value=10):
                loader = _make_loader([self._msg("user", f"msg{i}") for i in range(10)])
                hm._loader = loader
                result = await hm.load_history(mode="direct")
        assert len(result) == 4

    @pytest.mark.asyncio
    async def test_interrupt_mode_returns_last_4(self, hm: HistoryManager, mock_cfg: MagicMock):
        mock_g = _mock_g_config(100000)
        mock_cfg.history_budget_ratio = 0.5
        mock_cfg.recent_rounds_keep = 3
        with patch("core.context.history.manager.g_config", mock_g):
            with patch("core.context.history.manager.count_tokens", return_value=10):
                msgs = [
                    self._msg("user", "msg0"),
                    self._msg("system", "sys"),
                    self._msg("tool", "tool result"),
                    self._msg("user", "msg3"),
                ]
                loader = _make_loader(msgs)
                hm._loader = loader
                result = await hm.load_history(mode="interrupt")
        assert len(result) == 4

    @pytest.mark.asyncio
    async def test_intent_shifted_filters_non_turn(self, hm: HistoryManager, mock_cfg: MagicMock):
        mock_g = _mock_g_config(100000)
        mock_cfg.history_budget_ratio = 0.5
        mock_cfg.recent_rounds_keep = 3
        with patch("core.context.history.manager.g_config", mock_g):
            with patch("core.context.history.manager.count_tokens", return_value=10):
                msgs = [
                    self._msg("user", "msg0"),
                    self._msg("system", "sys"),
                    self._msg("tool", "tool result"),
                    self._msg("user", "msg3"),
                ]
                loader = _make_loader(msgs)
                hm._loader = loader
                result = await hm.load_history(intent_shifted=True)
        roles = [m["role"] for m in result]
        assert all(r in ("user", "assistant") for r in roles)

    @pytest.mark.asyncio
    async def test_token_budget_respected_in_recent(self, hm: HistoryManager, mock_cfg: MagicMock):
        mock_g = _mock_g_config(100)
        mock_cfg.history_budget_ratio = 0.5
        mock_cfg.recent_rounds_keep = 3
        # Reversed iteration: first msg too big → stops immediately
        with patch("core.context.history.manager.g_config", mock_g):
            with patch("core.context.history.manager.count_tokens", side_effect=[51, 10]):
                msgs = [self._msg("user", "big msg")]
                loader = _make_loader(msgs)
                hm._loader = loader
                result = await hm.load_history()
        assert result == []

    @pytest.mark.asyncio
    async def test_tool_results_slimmed_before_token_count(self, hm: HistoryManager, mock_cfg: MagicMock):
        mock_g = _mock_g_config(100000)
        mock_cfg.history_budget_ratio = 0.5
        mock_cfg.recent_rounds_keep = 3

        with patch("core.context.history.manager.g_config", mock_g):
            with patch("core.context.history.manager.count_tokens", return_value=10):
                msgs = [self._msg("user", "hi"), self._msg("tool", "big output")]
                loader = _make_loader(msgs)
                hm._loader = loader

                with patch.object(hm, "_slim_tool_results", wraps=hm._slim_tool_results) as mock_slim:
                    await hm.load_history()
                    mock_slim.assert_called_once()
                    # _slim_tool_results is called on raw messages BEFORE token counting
                    called_msgs = mock_slim.call_args[0][0]
                    tool_msg = called_msgs[1]
                    assert tool_msg["role"] == "tool"
                    # Content depends on slimming: short enough → original content
                    assert "big output" in tool_msg["content"]

    @pytest.mark.asyncio
    async def test_remaining_budget_too_small_skips_compact(self, hm: HistoryManager, mock_cfg: MagicMock):
        mock_g = _mock_g_config(100)  # budget=100, ratio=1.0 → recent takes all
        mock_cfg.history_budget_ratio = 1.0
        mock_cfg.recent_rounds_keep = 1
        mock_cfg.history_load_limit = 20

        with patch("core.context.history.manager.g_config", mock_g):
            with patch("core.context.history.manager.count_tokens", return_value=10):
                with patch("core.context.history.manager._emergency_compact", new_callable=AsyncMock) as mock_compact:
                    msgs = [
                        self._msg("user", "old1"), self._msg("assistant", "old2"),
                        self._msg("user", "new"),
                    ]
                    loader = _make_loader(msgs)
                    hm._loader = loader
                    result = await hm.load_history()
                    mock_compact.assert_not_called()
                    assert any(m["content"] == "new" for m in result)

    @pytest.mark.asyncio
    async def test_compact_error_falls_back_to_earlier(self, hm: HistoryManager, mock_cfg: MagicMock):
        mock_g = _mock_g_config(100000)
        mock_cfg.history_budget_ratio = 0.5
        mock_cfg.recent_rounds_keep = 1
        mock_cfg.history_load_limit = 20
        compact_mock = AsyncMock(side_effect=RuntimeError("LLM error"))

        with patch("core.context.history.manager.g_config", mock_g):
            with patch("core.context.history.manager.count_tokens", return_value=10):
                with patch("core.context.history.manager._emergency_compact", compact_mock):
                    msgs = [
                        self._msg("user", "old1"), self._msg("assistant", "old2"),
                        self._msg("user", "new"),
                    ]
                    loader = _make_loader(msgs)
                    hm._loader = loader
                    result = await hm.load_history()
        assert any(m["content"] == "old1" for m in result)
        assert any(m["content"] == "new" for m in result)
