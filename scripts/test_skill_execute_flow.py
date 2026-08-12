# SPDX-License-Identifier: Apache-2.0
"""Smoke test: Skill execution path via SkillAgent.

Run:
  python scripts/test_skill_execute_flow.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from core.skill.execution import SkillAgent
from core.skill.schema import Skill
from bootstrap import bootstrap
from middleware.config import g_config
from shared.schema import SkillConfig


def _make_skill() -> Skill:
    workspace = Path(__file__).resolve().parents[1]
    skill_dir = workspace / "builtin" / "skills" / "web-search"
    return Skill(
        name="web-search",
        description="Web search skill smoke test",
        content="",
        source_dir=str(skill_dir),
    )


async def main() -> None:
    await bootstrap()
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

    print("== SkillAgent smoke test ==")
    workspace = Path(path_info["workspace_dir"])
    print(f"[skill] {skill.name} (source_dir={skill.source_dir})")
    print(f"[workspace] {workspace}")

    # Test that agent can be created
    print("[agent] SkillAgent created successfully")
    print(f"[agent] llm={type(agent._llm).__name__}")
    print(f"[agent] skill_env_cache={agent._skill_env_cache}")

    # Test env vars
    from core.skill.execution.state import ReActState
    state = ReActState(query="test", params={}, max_turns=30)
    env_vars = agent._build_env_vars(workspace, state)
    print(f"[env] WORKSPACE_ROOT={env_vars.get('WORKSPACE_ROOT')}")

    print("\n[ok] SkillAgent smoke test passed")


if __name__ == "__main__":
    asyncio.run(main())
