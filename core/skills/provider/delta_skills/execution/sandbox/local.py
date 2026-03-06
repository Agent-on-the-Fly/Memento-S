
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from core.config import g_settings
from core.config.logging import get_logger
from ..analyzer import parse_code
from ...schema import Skill, SkillExecutionResult
from .artifacts import collect_local_artifacts, snapshot_files
from .base import BaseSandbox

logger = get_logger(__name__)

_ERROR_PREFIXES = (
    "error:", "error ", "traceback (most recent call last)",
    "exception:", "failed:", "fatal:",
)


class LocalSandbox(BaseSandbox):
    def run(
        self,
        cmd: list[str],
        *,
        cwd: str | Path,
        pythonpath: str | Path | None = None,
        timeout: int | None = None,
        skill_name: str = "",
        check_syntax: str | None = None,
    ) -> SkillExecutionResult:
        if check_syntax is not None and parse_code(check_syntax) is None:
            import ast as _ast
            try:
                _ast.parse(check_syntax)
                syntax_detail = "SyntaxError in generated code (unknown location)"
            except SyntaxError as _se:
                syntax_detail = (
                    f"SyntaxError in generated code at line {_se.lineno}: {_se.msg}\n"
                    f"  {_se.text.rstrip() if _se.text else '(no source)'}"
                )
            return SkillExecutionResult(
                success=False, result=None,
                error=syntax_detail,
                skill_name=skill_name,
            )
        env = os.environ.copy()
        if pythonpath is not None:
            env["PYTHONPATH"] = str(pythonpath) + os.pathsep + env.get("PYTHONPATH", "")

        try:
            proc = subprocess.run(
                cmd, cwd=str(cwd), env=env,
                capture_output=True, text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return SkillExecutionResult(
                success=False, result=None,
                error=f"Execution timed out after {timeout}s",
                skill_name=skill_name,
            )

        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        if proc.returncode != 0:
            return SkillExecutionResult(
                success=False, result=stdout or None,
                error=self._format_error(proc.returncode, stdout, stderr),
                skill_name=skill_name,
            )

        if self._stderr_has_real_errors(stderr):
            return SkillExecutionResult(
                success=False, result=stdout or None,
                error=f"Execution stderr indicates error:\n{stderr[:2000]}",
                skill_name=skill_name,
            )

        if self._stdout_indicates_error(stdout):
            return SkillExecutionResult(
                success=False, result=None,
                error=f"Execution output indicates error:\n{stdout[:2000]}",
                skill_name=skill_name,
            )

        return SkillExecutionResult(
            success=True, result=stdout,
            skill_name=skill_name,
        )

    def run_code(
        self,
        code: str,
        skill: Skill,
        deps: list[str] | None = None,
    ) -> SkillExecutionResult:
        with tempfile.TemporaryDirectory(prefix="delta_sandbox_") as tmp:
            work_dir = Path(tmp)
            try:
                self._prepare_workspace(skill, work_dir)
                pre_files = snapshot_files(work_dir)
                runner_path = work_dir / "__runner__.py"
                runner_path.write_text(code, encoding="utf-8")
                if deps:
                    logger.debug(
                        "Sandbox: skill '%s' declares deps %s; "
                        "expected to be installed by agent before invocation",
                        skill.name, deps,
                    )
                logger.info(
                    "Sandbox executing '%s' in %s (%d files)",
                    skill.name, work_dir,
                    sum(1 for _ in work_dir.rglob("*.py")),
                )
                result = self.run(
                    [sys.executable, str(runner_path)],
                    cwd=work_dir,
                    pythonpath=work_dir,
                    timeout=g_settings.execution_timeout_sec,
                    skill_name=skill.name,
                    check_syntax=code,
                )

                if not result.success:
                    return result
                artifacts = collect_local_artifacts(work_dir, pre_files, skill.name)
                logger.info("Sandbox success for '%s' (%d chars)", skill.name, len(result.result or ""))
                return SkillExecutionResult(
                    success=True, result=result.result,
                    skill_name=skill.name, artifacts=artifacts,
                )
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                logger.error("Sandbox error for '%s': %s", skill.name, e)
                return SkillExecutionResult(
                    success=False, result=None,
                    error=error_msg, skill_name=skill.name,
                )

    def _prepare_workspace(self, skill: Skill, work_dir: Path):
        has_files = False

        if skill.source_dir:
            source = Path(skill.source_dir)
            scripts_dir = source / "scripts"
            if scripts_dir.exists() and any(scripts_dir.glob("*.py")):
                shutil.copytree(scripts_dir, work_dir, dirs_exist_ok=True)
                has_files = True
            for extra_dir in ("references", "assets"):
                extra_src = source / extra_dir
                if extra_src.exists():
                    shutil.copytree(extra_src, work_dir / extra_dir, dirs_exist_ok=True)

        if not has_files and skill.files:
            for filename, content in skill.files.items():
                file_path = work_dir / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content, encoding="utf-8")

        for dirpath in work_dir.rglob("*"):
            if dirpath.is_dir() and not (dirpath / "__init__.py").exists():
                if any(dirpath.glob("*.py")):
                    (dirpath / "__init__.py").touch()

        if not (work_dir / "__init__.py").exists():
            (work_dir / "__init__.py").touch()

    @staticmethod
    def _install_deps(deps: list[str]) -> bool:
        venv_python = Path(sys.executable)
        uv_bin = venv_python.parent / "uv"
        if not uv_bin.exists():
            uv_bin = Path(shutil.which("uv") or "")

        cmds: list[list[str]] = []
        if uv_bin.exists():
            cmds.append([str(uv_bin), "pip", "install", "-q", "--python", str(venv_python), *deps])
        cmds.append([str(venv_python), "-m", "pip", "install", "-q", *deps])

        for cmd in cmds:
            installer = "uv" if "uv" in cmd[0] else "pip"
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120,
                )
                if proc.returncode == 0:
                    logger.info("Installed dependencies via %s: %s", installer, deps)
                    return True
                logger.warning(
                    "%s install failed (exit=%d) for %s: %s",
                    installer, proc.returncode, deps, proc.stderr[:500],
                )
            except subprocess.TimeoutExpired:
                logger.warning("%s install timed out for deps: %s", installer, deps)
            except Exception as e:
                logger.warning("Failed to install deps %s via %s: %s", deps, installer, e)

        logger.error("All install attempts failed for deps: %s", deps)
        return False

    @staticmethod
    def _format_error(returncode: int, stdout: str, stderr: str) -> str:
        parts = [f"Exit code: {returncode}"]
        if stderr:
            parts.append(f"Stderr:\n{stderr[:4000]}")
        if stdout:
            parts.append(f"Stdout:\n{stdout[:2000]}")
        return "\n".join(parts)

    @staticmethod
    def _stderr_has_real_errors(stderr: str) -> bool:
        if not stderr:
            return False
        return not all(
            "warning" in line.lower() or "deprecat" in line.lower() or not line.strip()
            for line in stderr.split("\n")
        )

    @staticmethod
    def _stdout_indicates_error(stdout: str) -> bool:
        if not stdout:
            return False
        lower = stdout.lower().strip()
        if any(lower.startswith(p) for p in _ERROR_PREFIXES):
            return True
        lines = [ln for ln in stdout.strip().split("\n") if ln.strip()]
        if len(lines) == 1 and lines[0].strip().lower().startswith("error"):
            return True
        return False
