
from __future__ import annotations

import shlex
import traceback
from pathlib import Path

from core.config import g_settings
from core.config.logging import get_logger
from ...schema import Skill, SkillExecutionResult
from .artifacts import get_output_dir, should_ignore_artifact
from .base import BaseSandbox

logger = get_logger(__name__)


class E2BSandbox(BaseSandbox):
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
        try:
            from e2b_code_interpreter import Sandbox
            sandbox = Sandbox.create(api_key=g_settings.e2b_api_key)
            try:
                cmd_str = " ".join(shlex.quote(str(c)) for c in cmd)
                if pythonpath is not None:
                    cmd_str = f"PYTHONPATH={shlex.quote(str(pythonpath))}:$PYTHONPATH {cmd_str}"
                if cwd:
                    cmd_str = f"cd {shlex.quote(str(cwd))} && {cmd_str}"

                execution = sandbox.commands.run(cmd_str, timeout=timeout or 300)
                stdout = execution.stdout.strip() if execution.stdout else ""
                stderr = execution.stderr.strip() if execution.stderr else ""
                if execution.exit_code != 0:
                    parts = [f"Exit code: {execution.exit_code}"]
                    if stderr:
                        parts.append(f"Stderr:\n{stderr[:4000]}")
                    if stdout:
                        parts.append(f"Stdout:\n{stdout[:2000]}")
                    return SkillExecutionResult(
                        success=False, result=stdout or None,
                        error="\n".join(parts),
                        skill_name=skill_name,
                    )
                return SkillExecutionResult(
                    success=True, result=stdout,
                    skill_name=skill_name,
                )
            finally:
                sandbox.kill()
        except ImportError:
            from .local import LocalSandbox
            logger.warning("e2b-code-interpreter not installed, falling back to local sandbox")
            return LocalSandbox().run(
                cmd, cwd=cwd, pythonpath=pythonpath,
                timeout=timeout, skill_name=skill_name,
                check_syntax=check_syntax,
            )
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            return SkillExecutionResult(
                success=False, result=None,
                error=error_msg, skill_name=skill_name,
            )

    def run_code(
        self,
        code: str,
        skill: Skill,
        deps: list[str] | None = None,
    ) -> SkillExecutionResult:
        try:
            from e2b_code_interpreter import Sandbox
            sandbox = Sandbox.create(api_key=g_settings.e2b_api_key)
            try:
                if deps:
                    deps_str = " ".join(deps)
                    logger.info("Installing dependencies in E2B: %s", deps_str)
                    sandbox.commands.run(f"pip install -q {deps_str}", timeout=60)
                uploaded_files = self._upload_skill_files(sandbox, skill)
                sandbox.files.write("/home/user/workspace/__runner__.py", code)
                path_setup = "import sys; sys.path.insert(0, '/home/user/workspace')\n"
                execution = sandbox.run_code(
                    path_setup + (
                        "import runpy; "
                        "runpy.run_path('/home/user/workspace/__runner__.py', run_name='__main__')"
                    ),
                    timeout=g_settings.execution_timeout_sec,
                )
                if execution.error:
                    is_success_exit = (
                        execution.error.name == "SystemExit"
                        and str(execution.error.value).strip() in ("0", "None", "")
                    )
                    if is_success_exit:
                        result_text = "".join(execution.logs.stdout).strip()
                        artifacts = self._download_artifacts(sandbox, skill, uploaded_files)
                        return SkillExecutionResult(
                            success=True, result=result_text,
                            skill_name=skill.name, artifacts=artifacts,
                        )
                    stderr_text = "".join(execution.logs.stderr).strip()
                    error_msg = (
                        f"{execution.error.name}: {execution.error.value}\n"
                        f"{execution.error.traceback}"
                    )
                    if stderr_text:
                        error_msg += f"\n\nStderr:\n{stderr_text}"
                    return SkillExecutionResult(
                        success=False, result=None,
                        error=error_msg, skill_name=skill.name,
                    )
                result_text = "".join(execution.logs.stdout).strip()
                artifacts = self._download_artifacts(sandbox, skill, uploaded_files)
                return SkillExecutionResult(
                    success=True, result=result_text,
                    skill_name=skill.name, artifacts=artifacts,
                )
            finally:
                sandbox.kill()
        except ImportError:
            from .local import LocalSandbox
            logger.warning("e2b-code-interpreter not installed, falling back to local sandbox")
            return LocalSandbox().run_code(code, skill, deps)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            return SkillExecutionResult(
                success=False, result=None,
                error=error_msg, skill_name=skill.name,
            )

    @staticmethod
    def _upload_skill_files(sandbox, skill: Skill) -> set[str]:
        base_path = "/home/user/workspace"
        uploaded: set[str] = set()
        if skill.source_dir:
            source = Path(skill.source_dir)
            scripts_dir = source / "scripts"
            if scripts_dir.exists():
                for f in scripts_dir.rglob("*"):
                    if f.is_file():
                        rel = str(f.relative_to(scripts_dir))
                        try:
                            content = f.read_text(encoding="utf-8")
                            sandbox.files.write(f"{base_path}/{rel}", content)
                            uploaded.add(rel)
                        except UnicodeDecodeError:
                            try:
                                sandbox.files.write(f"{base_path}/{rel}", f.read_bytes())
                                uploaded.add(rel)
                            except Exception as e:
                                logger.warning("Failed to upload binary file '%s': %s", rel, e)
                logger.info("Uploaded scripts/ from %s to E2B", scripts_dir)
            for extra_dir in ("references", "assets"):
                extra_src = source / extra_dir
                if extra_src.exists():
                    for f in extra_src.rglob("*"):
                        if f.is_file():
                            rel = str(f.relative_to(source))
                            try:
                                content = f.read_text(encoding="utf-8")
                                sandbox.files.write(f"{base_path}/{rel}", content)
                                uploaded.add(rel)
                            except UnicodeDecodeError:
                                try:
                                    sandbox.files.write(f"{base_path}/{rel}", f.read_bytes())
                                    uploaded.add(rel)
                                except Exception as e:
                                    logger.warning("Failed to upload binary asset '%s': %s", rel, e)
        elif skill.files:
            for filename, content in skill.files.items():
                sandbox.files.write(f"{base_path}/{filename}", content)
                uploaded.add(filename)
            logger.info("Uploaded %d files from skill.files to E2B", len(skill.files))

        return uploaded

    def _download_artifacts(
        self,
        sandbox,
        skill: Skill,
        uploaded_files: set[str],
    ) -> list[str]:
        base_path = "/home/user/workspace"
        local_artifacts: list[str] = []

        try:
            remote_files = self._list_remote_files(sandbox, base_path)
            new_files = [
                f for f in remote_files
                if f not in uploaded_files and not should_ignore_artifact(f)
            ]

            if not new_files:
                return []

            output_dir = get_output_dir(skill.name)
            for remote_rel in new_files:
                try:
                    content = sandbox.files.read(f"{base_path}/{remote_rel}")
                    local_path = output_dir / remote_rel
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    if isinstance(content, bytes):
                        local_path.write_bytes(content)
                    else:
                        local_path.write_text(content, encoding="utf-8")
                    local_artifacts.append(str(local_path))
                    logger.debug("Downloaded artifact: %s → %s", remote_rel, local_path)
                except Exception as e:
                    logger.warning("Failed to download artifact '%s' from E2B: %s", remote_rel, e)

            if local_artifacts:
                logger.info(
                    "Downloaded %d artifacts from E2B for '%s' to %s",
                    len(local_artifacts), skill.name, output_dir,
                )
        except Exception as e:
            logger.warning("Failed to collect artifacts from E2B: %s", e)
        return local_artifacts

    @staticmethod
    def _list_remote_files(sandbox, base_path: str) -> list[str]:
        result: list[str] = []
        E2BSandbox._walk_remote(sandbox, base_path, "", result)
        return result

    @staticmethod
    def _walk_remote(sandbox, current_path: str, rel_prefix: str, out: list[str]):
        try:
            entries = sandbox.files.list(current_path)
        except Exception as e:
            logger.warning("Failed to list remote path '%s': %s", current_path, e)
            return
        for entry in entries:
            entry_name = entry.name if hasattr(entry, "name") else str(entry)
            entry_rel = f"{rel_prefix}/{entry_name}" if rel_prefix else entry_name
            is_dir = getattr(entry, "is_dir", None) or getattr(entry, "type", "") == "directory"
            if is_dir:
                E2BSandbox._walk_remote(sandbox, f"{current_path}/{entry_name}", entry_rel, out)
            else:
                out.append(entry_rel)
