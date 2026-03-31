from __future__ import annotations

import json

from core.context.scratchpad import Scratchpad


def test_persist_short_result_inline(scratchpad: Scratchpad):
    """Short results are returned inline without creating artifacts."""
    result = json.dumps({"ok": True, "summary": "done", "skill_name": "filesystem"})

    msg = scratchpad.persist_tool_result("call-1", "filesystem", result)

    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call-1"
    assert msg["content"] == result

    sp_content = scratchpad.path.read_text(encoding="utf-8")
    assert "filesystem" not in sp_content


def test_persist_long_result_folded(scratchpad: Scratchpad):
    """Long result (>4000 chars) is folded to artifact, ref+preview returned."""
    result = json.dumps({
        "ok": True,
        "results": [
            {
                "tool": "read_file",
                "args": {"path": "/workspace/test.md"},
                "result": "file content " * 500,
            }
        ],
    })
    assert len(result) > 4000

    msg = scratchpad.persist_tool_result("call-2", "search_file", result)

    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call-2"
    assert "[artifact_ref:" in msg["content"]
    assert "chars, full content archived]" in msg["content"]

    # artifact file should exist with full original content
    artifacts = list(scratchpad.artifacts_dir.iterdir())
    assert len(artifacts) == 1
    artifact_text = artifacts[0].read_text(encoding="utf-8")
    assert "file content" in artifact_text


def test_persist_long_non_json_folded(scratchpad: Scratchpad):
    """Long non-JSON result (>120 lines) is folded to artifact."""
    raw = "plain text output line\n" * 200

    msg = scratchpad.persist_tool_result("call-3", "run_cmd", raw)

    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call-3"
    assert "[artifact_ref:" in msg["content"]
    assert "plain text output line" in msg["content"]  # preview present
