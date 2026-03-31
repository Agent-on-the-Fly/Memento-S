#!/usr/bin/env python3
"""
Test core.context module: ContextManager.

Validates:
1. Token counting utilities
2. Scratchpad init/write + persist_tool_result (JSON/non-JSON)
3. ContextManager.append — compress + compact
4. ContextManager._get_context_section — scratchpad ref
5. ContextConfig — defaults

Usage:
    .venv/bin/python tests/test_context_manager.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from middleware.llm.utils import looks_like_tool_call_text
from utils.token_utils import count_tokens_messages
from core.context.schemas import ContextConfig
from core.memento_s.schemas import AgentConfig


# ── Fixtures ────────────────────────────────────────────────────────


def _make_ctx_cfg(**overrides) -> ContextConfig:
    return ContextConfig(**overrides)


def _make_tool_result_json(ok: bool = True, summary: str = "done") -> str:
    payload: dict = {
        "ok": ok,
        "results": [
            {
                "tool": "read_file",
                "args": {"path": "/workspace/test.md"},
                "result": "--- File: /workspace/test.md ---\n" + "x" * 5000,
            }
        ],
    }
    if not ok:
        payload["summary"] = summary
        payload["error_code"] = "RUNTIME_ERROR"
    return json.dumps(payload)


def _create_context_manager(tmp_dir: Path, session_id: str = "test-session"):
    from core.context.manager import ContextManager

    mock_g_config = MagicMock()
    mock_g_config.paths.context_dir = tmp_dir / "context"
    with patch("core.context.manager.g_config", mock_g_config):
        return ContextManager(
            session_id=session_id, config=_make_ctx_cfg(),
        )


def _create_scratchpad(tmp_dir: Path, session_id: str = "test-session"):
    from core.context.scratchpad import Scratchpad

    date_dir = tmp_dir / "context" / "2026-03-17"
    date_dir.mkdir(parents=True, exist_ok=True)
    return Scratchpad(session_id, date_dir)


# ═══════════════════════════════════════════════════════════════════
# Test 1: Token counting + tool call detection
# ═══════════════════════════════════════════════════════════════════


def test_msg_tokens():
    msg = {"role": "user", "content": "hello world"}
    tokens = count_tokens_messages([msg])
    assert tokens > 4

    msg = {"role": "user", "content": [{"type": "text", "text": "hello world"}]}
    tokens = count_tokens_messages([msg])
    assert tokens > 4

    msg = {"role": "user", "content": ""}
    tokens = count_tokens_messages([msg])
    assert tokens > 0


def test_looks_like_tool_call_text():
    assert looks_like_tool_call_text("") is False
    assert looks_like_tool_call_text("normal json {}") is False
    assert looks_like_tool_call_text('<|tool_call_begin|>{"name":"foo"}') is True
    assert looks_like_tool_call_text("<function=search>") is True


# ═══════════════════════════════════════════════════════════════════
# Test 2: Scratchpad — init and write
# ═══════════════════════════════════════════════════════════════════


def test_scratchpad_init_and_write():
    with tempfile.TemporaryDirectory() as tmp:
        sp = _create_scratchpad(Path(tmp))

        assert sp.path.exists()
        initial = sp.path.read_text()
        assert "Session Scratchpad" in initial

        ref = sp.write("Test Section", "content here")
        assert "scratchpad#section-1" in ref

        content = sp.path.read_text()
        assert "Test Section" in content
        assert "content here" in content

        ref2 = sp.write("Second", "more data")
        assert "section-2" in ref2


# ═══════════════════════════════════════════════════════════════════
# Test 3: persist_tool_result — artifact fold for long content
# ═══════════════════════════════════════════════════════════════════


def test_persist_tool_result_long_folded():
    """Long result (>4000 chars) is folded to artifact file, ref+preview returned."""
    with tempfile.TemporaryDirectory() as tmp:
        sp = _create_scratchpad(Path(tmp))
        result_json = _make_tool_result_json(ok=True)
        assert len(result_json) > 4000, "Test fixture must exceed fold threshold"

        msg = sp.persist_tool_result("call-1", "search_file", result_json)

        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call-1"
        # content should be a ref, not the raw JSON
        assert "[artifact_ref:" in msg["content"]
        assert "chars, full content archived]" in msg["content"]

        # artifact file should exist with full content
        artifact_dir = sp.artifacts_dir
        assert artifact_dir.exists()
        artifacts = list(artifact_dir.iterdir())
        assert len(artifacts) == 1
        artifact_content = artifacts[0].read_text(encoding="utf-8")
        assert "read_file" in artifact_content  # original content preserved


def test_persist_tool_result_non_json_long_folded():
    """Long non-JSON result (>120 lines) is folded to artifact."""
    with tempfile.TemporaryDirectory() as tmp:
        sp = _create_scratchpad(Path(tmp))
        raw = "plain text result\n" * 200
        assert raw.count("\n") > 120, "Test fixture must exceed line limit"

        msg = sp.persist_tool_result("call-2", "run_cmd", raw)

        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call-2"
        assert "[artifact_ref:" in msg["content"]
        # preview should contain the first few lines
        assert "plain text result" in msg["content"]


def test_persist_tool_result_short_inline():
    """Short result stays inline (no fold)."""
    with tempfile.TemporaryDirectory() as tmp:
        sp = _create_scratchpad(Path(tmp))
        short_result = '{"ok": true, "summary": "done"}'

        initial_scratchpad = sp.path.read_text()
        msg = sp.persist_tool_result("call-3", "filesystem", short_result)

        assert msg["role"] == "tool"
        assert msg["content"] == short_result
        assert sp.path.read_text() == initial_scratchpad
        # no artifact directory created
        assert not sp.artifacts_dir.exists()


def test_persist_tool_result_small_inline():
    """Small batch result stays inline (no fold)."""
    with tempfile.TemporaryDirectory() as tmp:
        sp = _create_scratchpad(Path(tmp))
        small_result = '{"ok": true, "results": [{"tool": "list_dir", "result": "file1.txt"}]}'

        initial_scratchpad = sp.path.read_text()
        msg = sp.persist_tool_result("call-4", "filesystem", small_result)

        assert msg["role"] == "tool"
        assert msg["content"] == small_result
        assert sp.path.read_text() == initial_scratchpad


def test_persist_tool_result_artifact_preview_limits():
    """Artifact preview respects max_lines and max_chars limits."""
    with tempfile.TemporaryDirectory() as tmp:
        sp = _create_scratchpad(Path(tmp))
        # Generate content with many long lines
        lines = [f"line_{i}: " + "x" * 200 for i in range(200)]
        raw = "\n".join(lines)

        msg = sp.persist_tool_result("call-5", "grep_logs", raw)

        assert "[artifact_ref:" in msg["content"]
        # preview part (between ref line and archive notice)
        content_lines = msg["content"].split("\n")
        # first line is [artifact_ref:...], last line is [N chars, ...]
        # preview should be limited (default 5 lines, 500 chars)
        preview_part = "\n".join(content_lines[1:-1])
        assert len(preview_part) <= 600  # some slack for formatting


def test_persist_tool_result_multiple_artifacts_sequential():
    """Multiple fold calls create sequentially numbered artifacts."""
    with tempfile.TemporaryDirectory() as tmp:
        sp = _create_scratchpad(Path(tmp))
        long_content = "x" * 5000

        sp.persist_tool_result("call-a", "tool_a", long_content)
        sp.persist_tool_result("call-b", "tool_b", long_content)

        artifacts = sorted(sp.artifacts_dir.iterdir())
        assert len(artifacts) == 2
        assert "artifact_0001" in artifacts[0].name
        assert "artifact_0002" in artifacts[1].name


# ═══════════════════════════════════════════════════════════════════
# Test 4: ContextManager — append (compress + compact)
# ═══════════════════════════════════════════════════════════════════


def test_append_short_messages():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _create_context_manager(Path(tmp))
        ctx.init_budget(80000)

        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        ctx.sync_tokens(msgs)

        new_msgs = [{"role": "assistant", "content": "hi there"}]
        result = asyncio.run(ctx.append(msgs, new_msgs))

        assert len(result) == 3
        assert result[-1]["content"] == "hi there"
        assert ctx.total_tokens > 0


def test_append_tracks_tokens():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _create_context_manager(Path(tmp))
        ctx.init_budget(80000)

        msgs = [{"role": "system", "content": "sys"}]
        ctx.sync_tokens(msgs)
        initial_tokens = ctx.total_tokens

        new_msgs = [{"role": "user", "content": "hello world " * 50}]
        asyncio.run(ctx.append(msgs, new_msgs))

        assert ctx.total_tokens > initial_tokens


# ═══════════════════════════════════════════════════════════════════
# Test 5: ContextManager — _get_context_section
# ═══════════════════════════════════════════════════════════════════


def test_get_context_section_empty():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _create_context_manager(Path(tmp))
        section = ctx._get_context_section()
        assert section == ""


def test_get_context_section_with_scratchpad():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _create_context_manager(Path(tmp))
        ctx.write_to_scratchpad("Big Data", "x" * 500)

        section = ctx._get_context_section()
        assert "Scratchpad" in section
        assert "filesystem" in section


# ═══════════════════════════════════════════════════════════════════
# Test 6: ContextConfig — defaults
# ═══════════════════════════════════════════════════════════════════


def test_context_config_defaults():
    cfg = ContextConfig()
    assert cfg.compaction_trigger_ratio == 0.7
    assert cfg.compress_threshold_ratio == 0.5
    assert cfg.summary_ratio == 0.15


def test_agent_config_has_context():
    cfg = AgentConfig()
    assert isinstance(cfg.context, ContextConfig)
    assert cfg.context.compaction_trigger_ratio == 0.7


# ═══════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════


def main():
    print("=" * 60)
    print("Context Manager Tests")
    print("=" * 60)

    tests = [
        test_msg_tokens,
        test_looks_like_tool_call_text,
        test_scratchpad_init_and_write,
        test_persist_tool_result_long_folded,
        test_persist_tool_result_non_json_long_folded,
        test_persist_tool_result_short_inline,
        test_persist_tool_result_small_inline,
        test_persist_tool_result_artifact_preview_limits,
        test_persist_tool_result_multiple_artifacts_sequential,
        test_append_short_messages,
        test_append_tracks_tokens,
        test_get_context_section_empty,
        test_get_context_section_with_scratchpad,
        test_context_config_defaults,
        test_agent_config_has_context,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
            print(f"  PASS: {test_fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL: {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
