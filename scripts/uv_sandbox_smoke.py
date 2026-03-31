"""Minimal smoke test for UvSandbox."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.skill.execution.sandbox.uv import UvSandbox
from core.skill.schema import Skill
from middleware.config import g_config


def _print_header() -> None:
    print("[uv-smoke] config")
    print(f"  sandbox_provider: {g_config.skills.execution.sandbox_provider}")
    print(f"  uv_python_version: {g_config.skills.execution.uv_python_version}")
    print(f"  uv_venv_path: {g_config.skills.execution.uv_venv_path}")
    print(f"  workspace_dir: {g_config.paths.workspace_dir}")


def main() -> int:
    _print_header()
    sandbox = UvSandbox()

    # 1) Simple run via run_code
    skill = Skill(
        name="uv_smoke",
        description="uv sandbox smoke",
        code="print('hello uv')",
    )

    code = (
        "import sys\n"
        "print('hello uv')\n"
        "print('python', sys.version)\n"
        "print('executable', sys.executable)\n"
    )

    result = sandbox.run_code(code, skill, deps=None, session_id="uv_smoke")
    print("[uv-smoke] run_code success:", result.success)
    print("[uv-smoke] run_code result:\n", result.result)
    if result.error:
        print("[uv-smoke] run_code error:\n", result.error)
        return 1

    # 2) Install a tiny dependency and import it
    dep_code = (
        "import tomli\n"
        "print('tomli ok', tomli.__version__)\n"
    )
    dep_result = sandbox.run_code(dep_code, skill, deps=["tomli"], session_id="uv_smoke")
    print("[uv-smoke] deps success:", dep_result.success)
    print("[uv-smoke] deps result:\n", dep_result.result)
    if dep_result.error:
        print("[uv-smoke] deps error:\n", dep_result.error)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
