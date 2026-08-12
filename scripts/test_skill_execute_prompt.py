# SPDX-License-Identifier: Apache-2.0
"""Prompt-only test: render SkillAgent prompt with runtime paths.

Run:
  python scripts/test_skill_execute_prompt.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from bootstrap import bootstrap
from core.skill.execution import SkillAgent
from core.skill.execution.state import ReActState
from core.skill.schema import Skill
from middleware.config import g_config


def _make_skill() -> Skill:
    workspace = Path(__file__).resolve().parents[1]
    fake_skill_dir = workspace / "tests" / "fixtures" / "skills" / "prompt_only"
    (fake_skill_dir / "scripts").mkdir(parents=True, exist_ok=True)
    (fake_skill_dir / "scripts" / "demo_script.py").write_text(
        "print('ok')\n", encoding="utf-8"
    )
    return Skill(
        name="uv_sandbox_smoke",
        description="Prompt-only test for skill execution",
        content="",
        source_dir=str(fake_skill_dir),
    )


async def main() -> None:
    await bootstrap()
    from shared.schema import SkillConfig
    config = SkillConfig.from_global_config()
    agent = SkillAgent(config=config)
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

    print("\n== Messages (rendered) ==")
    state = ReActState(
        query="test prompt for sandbox and tool paths",
        params={},
        max_turns=30,
    )
    workspace = Path(path_info["workspace_dir"])
    messages = agent._build_messages(skill, state, workspace)
    for msg in messages:
        print(f"\n[{msg['role']}]")
        content = msg.get("content", "")
        if isinstance(content, str) and len(content) > 500:
            print(content[:500] + "...")
        else:
            print(content)


if __name__ == "__main__":
    asyncio.run(main())
