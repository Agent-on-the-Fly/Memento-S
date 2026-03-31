"""Scratchpad._format_for_scratchpad 格式化测试。"""
from __future__ import annotations

import json

from core.context.scratchpad import Scratchpad


def test_format_skill_payload():
    """Skill payload → markdown 格式。"""
    data = json.dumps({
        "ok": True,
        "skill_name": "filesystem",
        "summary": "Read 3 files",
        "output": "file content here",
    })
    out = Scratchpad._format_for_scratchpad(data)
    assert "**filesystem**" in out
    assert "Read 3 files" in out
    assert "file content here" in out


def test_format_skill_payload_with_diagnostics():
    """Skill payload 带 diagnostics → 包含 diagnostics。"""
    data = json.dumps({
        "ok": False,
        "skill_name": "bash",
        "summary": "Command failed",
        "output": "error output",
        "diagnostics": {"exit_code": 1},
    })
    out = Scratchpad._format_for_scratchpad(data)
    assert "**bash**" in out
    assert "FAIL" in out
    assert "exit_code" in out


def test_format_batch_results():
    """批量 results → markdown sections。"""
    data = json.dumps({
        "results": [
            {"tool": "read_file", "args": {"path": "/tmp/a.py"}, "result": "content here"},
            {"tool": "search_grep", "args": {"query": "foo"}, "result": "line 5: foo"},
        ]
    })
    out = Scratchpad._format_for_scratchpad(data)
    assert "### read_file: /tmp/a.py" in out
    assert "### search_grep: foo" in out


def test_format_batch_with_error():
    """批量 result 中的 error → ERROR 标签。"""
    data = json.dumps({
        "results": [
            {"tool": "run_cmd", "args": {"command": "ls"}, "error": "permission denied"},
        ]
    })
    out = Scratchpad._format_for_scratchpad(data)
    assert "**ERROR**" in out


def test_format_json_no_known_keys():
    """JSON 不含已知 key → 原样返回。"""
    data = json.dumps({"ok": True, "value": 42})
    assert Scratchpad._format_for_scratchpad(data) == data


def test_format_non_json():
    """非 JSON → 原样返回。"""
    raw = "plain text output"
    assert Scratchpad._format_for_scratchpad(raw) == raw


def test_format_empty_results():
    """空 results 数组 → 原样返回。"""
    data = json.dumps({"results": []})
    assert Scratchpad._format_for_scratchpad(data) == data
