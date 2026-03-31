"""Smoke test: Skill execution path + uv sandbox + bash tool.

Run:
  python scripts/test_skill_execute_flow.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from core.skill.execution.executor import SkillExecutor
from core.skill.schema import Skill
from bootstrap import bootstrap
from middleware.config import g_config


def _make_skill() -> Skill:
    workspace = Path(__file__).resolve().parents[1]
    skill_dir = workspace / "builtin" / "skills" / "web-search"
    return Skill(
        name="web-search",
        description="Web search skill smoke test",
        content="",
        source_dir=str(skill_dir),
    )


async def _run_fallback(executor: SkillExecutor, skill: Skill) -> None:
    llm_content = """
```python
import os
import sys
from pathlib import Path

print("sys.executable=", sys.executable)
print("cwd=", os.getcwd())
print("VIRTUAL_ENV=", os.environ.get("VIRTUAL_ENV"))
print("UV_PYTHON=", os.environ.get("UV_PYTHON"))
print("PWD=", os.environ.get("PWD"))
print("output_dir_from_env=", os.environ.get("MEMENTO_OUTPUT_DIR"))
print("files_in_cwd=", [p.name for p in Path(".").iterdir()][:5])

Path("fallback_output.txt").write_text("fallback ok")
print("wrote fallback_output.txt")
```
""".strip()

    result, _ = await executor._execute_fallback(skill, llm_content)
    print("[fallback] success:", result.success)
    print("[fallback] result:", result.result)
    if result.error:
        print("[fallback] error:", result.error)


async def _run_tool_calls(executor: SkillExecutor, skill: Skill) -> None:
    tool_calls = [
        {
            "function": {
                "name": "bash",
                "arguments": {
                    "command": 'python scripts/search.sh \'{"query": "Cursor IDE", "max_results": 3}\'',
                },
            }
        },
        {
            "function": {
                "name": "bash",
                "arguments": {
                    "command": './scripts/search.sh \'{"query": "Cursor IDE", "max_results": 1}\'',
                },
            }
        },
        {
            "function": {
                "name": "list_dir",
                "arguments": {"path": "scripts", "max_depth": 1},
            }
        },
    ]

    result, _ = await executor._execute_with_tool_calls(skill, tool_calls)
    print("[tool_calls] success:", result.success)
    print("[tool_calls] result:", result.result)
    if result.error:
        print("[tool_calls] error:", result.error)


async def main() -> None:
    await bootstrap()
    executor = SkillExecutor()
    skill = _make_skill()

    session_id = "default"
    path_info = {
        "workspace_dir": str(g_config.paths.workspace_dir),
        "data_dir": str(g_config.get_data_dir()),
        "venv_dir": str(g_config.paths.venv_dir),
        "session_sandbox_dir": str(
            g_config.get_session_sandbox_dir(skill.name, session_id=session_id)
        ),
    }
    print("== Path Info ==")
    print(json.dumps(path_info, indent=2, ensure_ascii=False))

    print("== Fallback path (sandbox.run_code) ==")
    await _run_fallback(executor, skill)

    print("\n== Tool calls path (bash tool) ==")
    await _run_tool_calls(executor, skill)


if __name__ == "__main__":
    asyncio.run(main())
