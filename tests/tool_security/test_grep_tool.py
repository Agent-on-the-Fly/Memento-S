# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import asyncio

from tools.atomics.grep import grep


def test_grep_tool_basic(tmp_path: Path) -> None:
    """Test grep searches files and finds matching patterns."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    target = workspace / "data.txt"
    target.write_text("hello world", encoding="utf-8")

    result = asyncio.run(grep(
        pattern="hello",
        dir_path=str(workspace),
    ))

    assert "hello" in result
