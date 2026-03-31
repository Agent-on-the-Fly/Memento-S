#!/usr/bin/env python3
"""Run web-search skill end-to-end to verify web tool chain.

Usage:
  python scripts/verify_web_search_chain.py "your query"

This executes the web-search skill through SkillGateway, which should
produce ops and invoke builtin/tools/web.py (tavily_search_tool or fetch_webpage_tool).
Check the application logs to verify the tool calls.
"""

from __future__ import annotations

import asyncio
import sys

from core.skill.gateway import SkillGateway
from core.skill.config import SkillConfig
from middleware.config import g_config
from builtin.tools.bash import bash_tool
from core.skill.execution.sandbox import get_sandbox


def _get_query() -> str:
    if len(sys.argv) >= 2:
        return " ".join(sys.argv[1:]).strip()
    return "OpenClaw latest news and updates"


async def main() -> int:
    query = _get_query()
    g_config.load()
    skill_config = SkillConfig.from_global_config()
    provider = await SkillGateway.from_config(skill_config)
    try:
        result = await provider.execute("web-search", params={"request": query})

        command = "python -c \"print('sandbox_ok')\""
        bash_result = await bash_tool(command)

        sandbox = get_sandbox()
        sandbox.install_python_deps(["httpx"], timeout=120)

        httpx_command = "python -c \"import httpx; print('httpx_ok')\""
        httpx_result = await bash_tool(httpx_command)
    finally:
        pass  # Provider 不需要显式关闭

    print("Skill ok:", result.ok)
    print("Status:", result.status)
    if result.summary:
        print("Summary:", result.summary)
    if result.output:
        print("Output:\n", result.output)
    if result.diagnostics:
        print("Diagnostics:", result.diagnostics)

    print("Sandbox bash result:", bash_result)
    print("Sandbox httpx result:", httpx_result)

    ok = result.ok and ("sandbox_ok" in bash_result) and ("httpx_ok" in httpx_result)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
