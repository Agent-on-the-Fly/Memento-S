
from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from core.config import g_settings
from core.config.logging import get_logger

from ..sandbox import collect_local_artifacts, snapshot_files
from ...schema import Skill, SkillExecutionResult
from .base import BaseTrackExecutor

logger = get_logger(__name__)


class PlaybookExecutor(BaseTrackExecutor):

    async def execute(
        self,
        skill: Skill,
        params: dict[str, Any],
    ) -> tuple[SkillExecutionResult, str]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._execute_sync, skill, params)

    def _execute_sync(
        self,
        skill: Skill,
        params: dict[str, Any],
    ) -> tuple[SkillExecutionResult, str]:
        script_name = params.get("script", "")
        cli_args = params.get("args", [])
        if isinstance(cli_args, str):
            cli_args = cli_args.split()
        scripts_dir = Path(skill.source_dir) / "scripts"

        if not script_name:
            if skill.entry_script:
                script_name = skill.entry_script
                logger.info("Using entry_script '%s' for skill '%s'", script_name, skill.name)
            else:
                available = sorted(
                    p for p in scripts_dir.glob("*.py")
                    if p.name != "__init__.py"
                )
                if len(available) == 1:
                    script_name = available[0].stem
                    logger.info(
                        "Single-script auto-fallback: '%s' for skill '%s'",
                        script_name, skill.name,
                    )
                else:
                    names = [p.stem for p in available]
                    return SkillExecutionResult(
                        success=False, result=None,
                        error=f"No script specified. Available scripts: {', '.join(names)}",
                        skill_name=skill.name,
                    ), ""

        script_path = scripts_dir / f"{script_name}.py"
        if not script_path.exists():
            return SkillExecutionResult(
                success=False, result=None,
                error=f"Script not found: {script_name}.py",
                skill_name=skill.name,
            ), ""

        with tempfile.TemporaryDirectory(prefix="delta_playbook_") as tmp:
            work_dir = Path(tmp)
            try:
                shutil.copytree(scripts_dir, work_dir, dirs_exist_ok=True)
                source = Path(skill.source_dir)
                for extra_dir in ("references", "assets"):
                    extra_src = source / extra_dir
                    if extra_src.exists():
                        shutil.copytree(extra_src, work_dir / extra_dir, dirs_exist_ok=True)

                pre_files = snapshot_files(work_dir)
                target_script = work_dir / f"{script_name}.py"
                cmd = [sys.executable, str(target_script)] + [str(a) for a in cli_args]
                logger.info("Playbook executing: %s %s", script_name, cli_args)
                result = self.sandbox.run(
                    cmd,
                    cwd=work_dir,
                    pythonpath=work_dir,
                    timeout=g_settings.execution_timeout_sec,
                    skill_name=skill.name,
                )

                if not result.success:
                    return result, ""
                artifacts = collect_local_artifacts(work_dir, pre_files, skill.name)

                output = result.result
                if not output or (isinstance(output, str) and not output.strip()):
                    output = "Skill executed successfully (no output)."

                return SkillExecutionResult(
                    success=True, result=output,
                    skill_name=skill.name, artifacts=artifacts,
                ), ""
            except Exception as e:
                return SkillExecutionResult(
                    success=False, result=None,
                    error=f"{type(e).__name__}: {e}",
                    skill_name=skill.name,
                ), ""
