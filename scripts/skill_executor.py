# SPDX-License-Identifier: Apache-2.0
"""Standalone SkillGateway test runner (SkillAgent-backed).

Usage examples:
  python scripts/skill_executor.py
  python scripts/skill_executor.py --list
  python scripts/skill_executor.py --skill filesystem --request "list files in ."

This script executes the full SkillGateway flow (provider -> SkillAgent) so you can
compare tool_calls / python fallback / text-only responses across skills.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from middleware.config import g_config
from core.skill.gateway import SkillGateway


def _print_payload(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


CASE_LIBRARY: list[dict[str, Any]] = [
    {
        "name": "filesystem_list",
        "skill": "filesystem",
        "request": "List files in the workspace root and include file sizes.",
    },
    {
        "name": "filesystem_read",
        "skill": "filesystem",
        "request": "Read the README.md and summarize its purpose in 3 bullets.",
    },
    {
        "name": "filesystem_missing_request",
        "skill": "filesystem",
        "request": "",
    },
    {
        "name": "web_search",
        "skill": "web-search",
        "request": "最新的 Python 3.13 版本有哪些新特性？",
    },
    {
        "name": "invalid_skill",
        "skill": "nonexistent_skill",
        "request": "This should fail with skill not found.",
    },
]


async def _list_skills(provider: SkillGateway) -> None:
    skills = await provider.discover()
    print(f"Loaded {len(skills)} skill(s) from {g_config.get_skills_path()}")
    for m in skills:
        print(f"- {m.name}: {m.description}")


async def _execute_skill(provider: SkillGateway, skill_name: str, request: str) -> None:
    response = await provider.execute(
        skill_name=skill_name,
        params={"request": request},
    )
    payload = {
        "ok": response.ok,
        "status": response.status.value,
        "error_code": response.error_code.value if response.error_code else None,
        "summary": response.summary,
        "skill_name": response.skill_name,
        "output": response.output,
        "outputs": response.outputs,
        "artifacts": response.artifacts,
        "diagnostics": response.diagnostics,
    }
    _print_payload(payload)


async def _run_cases(provider: SkillGateway) -> None:
    for case in CASE_LIBRARY:
        name = case.get("name", "unnamed")
        print(f"\n=== CASE: {name} ===")
        await _execute_skill(provider, case["skill"], case["request"])

async def main() -> None:
    g_config.load()

    parser = argparse.ArgumentParser(description="Run SkillGateway tests")
    parser.add_argument("--list", action="store_true", help="List local skills")
    parser.add_argument("--skill", type=str, help="Skill name to execute")
    parser.add_argument("--request", type=str, help="Request text for the skill")
    args = parser.parse_args()

    from shared.schema import SkillConfig

    config = SkillConfig.from_global_config()
    provider = await SkillGateway.from_config(config)

    if args.list:
        await _list_skills(provider)
        return

    if args.skill and args.request is not None:
        await _execute_skill(provider, args.skill, args.request)
        return

    await _run_cases(provider)


if __name__ == "__main__":
    asyncio.run(main())
