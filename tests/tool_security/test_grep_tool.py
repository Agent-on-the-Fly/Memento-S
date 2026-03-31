from __future__ import annotations

from pathlib import Path

from builtin.tools.grep import grep_tool


def test_grep_tool_with_allow_roots(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    allow_roots = [str(workspace)]

    target = workspace / "data.txt"
    target.write_text("hello world", encoding="utf-8")

    result = grep_tool.__wrapped__(
        pattern="hello",
        dir_path=str(workspace),
        allow_roots=allow_roots,
    )

    assert "hello" in result
