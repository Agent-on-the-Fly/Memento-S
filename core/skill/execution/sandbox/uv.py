"""UV 本地沙箱 - UvLocalSandbox 实现。"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path

from middleware.config import g_config
from utils.logger import get_logger
from ..analyzer import parse_code
from ..analyzer.dependencies import extract_missing_module_from_error
from core.skill.schema import ErrorType, Skill, SkillExecutionOutcome
from .artifacts import ArtifactManager
from .base import BaseSandbox

logger = get_logger(__name__)

_STDERR_TRUNCATE_LEN = 2000
_STDOUT_TRUNCATE_LEN = 2000
_ERROR_MSG_TRUNCATE_LEN = 4000
_INSTALL_STDERR_TRUNCATE_LEN = 500

_ERROR_PREFIXES = (
    "error:",
    "error ",
    "traceback (most recent call last)",
    "exception:",
    "failed:",
    "fatal:",
)

from core.utils.platform import (
    SUBPROCESS_TEXT_KWARGS,
    filter_env_by_whitelist,
    venv_bin_dir as _venv_bin_dir,
    venv_python as _venv_python,
    pip_shim_path,
    pip_shim_content,
    chmod_executable,
    uv_install_hint,
)


def _config_env_vars() -> dict[str, str]:
    """从 config 提取环境变量。

    来源：cfg.env
    用户可在 config.json 的 "env" 中直接配置 PIP_INDEX_URL 等镜像变量。
    """
    env: dict[str, str] = {}
    try:
        cfg = g_config
        extra_env = cfg.env
        if isinstance(extra_env, dict):
            for key, value in extra_env.items():
                if value is None:
                    continue
                if isinstance(value, (str, int, float, bool)):
                    env[str(key).upper()] = str(value)
    except Exception:
        pass
    return env


class UvLocalSandbox(BaseSandbox):
    """使用 UV 管理的隔离虚拟环境沙箱。"""

    def __init__(self):
        self._uv_bin: Path | None = None
        self._venv_path: Path | None = None
        self._python_executable: Path | None = None
        self._ensure_uv_installed()
        self._setup_venv()

    @property
    def python_executable(self) -> Path:
        return self._python_executable

    def _get_uv_env(self) -> dict[str, str]:
        """构建 uv 子进程的环境变量，镜像从 cfg.env 注入。"""
        env = dict(os.environ)
        env.update(_config_env_vars())
        return env

    def _ensure_uv_installed(self) -> None:
        """确保 uv 已安装。"""
        uv_path = shutil.which("uv")
        if not uv_path:
            raise RuntimeError(
                f"uv is not installed. Please install uv first:\n  {uv_install_hint()}"
            )
        self._uv_bin = Path(uv_path)
        logger.info("Using uv: {}", self._uv_bin)

    def _setup_venv(self) -> None:
        """创建或验证虚拟环境。"""
        # 强制使用配置的 venv_dir
        if not g_config.paths.venv_dir:
            raise RuntimeError("venv_dir is not configured")
        self._venv_path = Path(g_config.paths.venv_dir).expanduser()

        python_version = getattr(g_config.skills.execution, "uv_python_version", "3.11")
        version_marker = self._venv_path / ".python-version"

        needs_create = False
        if not self._venv_path.exists():
            logger.info("Virtual environment not found at {}", self._venv_path)
            needs_create = True
        elif not version_marker.exists():
            logger.info("Version marker not found, recreating venv")
            needs_create = True
        elif version_marker.read_text().strip() != python_version:
            current = version_marker.read_text().strip()
            logger.info("Python version changed: {} -> {}", current, python_version)
            needs_create = True

        if needs_create:
            self._create_venv(python_version)
        else:
            logger.debug("Using existing venv at {}", self._venv_path)

        self._python_executable = _venv_python(self._venv_path)

        if not self._python_executable.exists():
            raise RuntimeError(
                f"Python executable not found at {self._python_executable}"
            )

        # 创建 pip shim
        self._create_pip_shim()

        logger.info("Sandbox venv ready: {}", self._venv_path)

    def _create_venv(self, python_version: str) -> None:
        """创建新的 uv venv。"""
        logger.info(
            f"Creating uv venv at {self._venv_path} with Python {python_version}"
        )

        if self._venv_path.exists():
            shutil.rmtree(self._venv_path)

        cmd = [
            str(self._uv_bin),
            "venv",
            str(self._venv_path),
            "--python",
            python_version,
        ]

        try:
            subprocess.run(
                cmd,
                capture_output=True,
                check=True,
                env=self._get_uv_env(),
                **SUBPROCESS_TEXT_KWARGS,
            )
            logger.info("Virtual environment created successfully")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create venv: {e.stderr}") from e

        version_marker = self._venv_path / ".python-version"
        version_marker.write_text(python_version)

    def _create_pip_shim(self) -> None:
        """在 venv 中创建 pip 包装脚本。

        uv venv 默认不包含 pip，创建 shim 调用 'python -m pip'。
        """
        pip_path = pip_shim_path(self._venv_path)
        pip_path.write_text(pip_shim_content(self._python_executable))
        chmod_executable(pip_path)
        logger.debug("Created pip shim at {}", pip_path)

        success, _ = self.install_python_deps(["pip"], timeout=60)
        if success:
            logger.debug("Installed pip into venv")
        else:
            logger.debug("Could not install pip into venv (non-fatal)")

    def get_sandbox_env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """获取沙箱环境变量。"""
        env = filter_env_by_whitelist()
        env.update(_config_env_vars())
        env["VIRTUAL_ENV"] = str(self._venv_path)
        env["UV_PYTHON"] = str(self._python_executable)

        venv_bin = _venv_bin_dir(self._venv_path)
        current_path = env.get("PATH", os.environ.get("PATH", ""))
        env["PATH"] = f"{venv_bin}{os.pathsep}{current_path}"

        if extra:
            env.update(extra)

        return env

    def get_python_executable(self) -> Path | None:
        return self._python_executable

    def run(
        self,
        cmd: list[str],
        *,
        cwd: str | Path,
        pythonpath: str | Path | None = None,
        timeout: int | None = None,
        skill_name: str = "",
        check_syntax: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SkillExecutionOutcome:
        """在沙箱中运行命令。"""
        if check_syntax is not None and parse_code(check_syntax) is None:
            import ast as _ast

            try:
                _ast.parse(check_syntax)
                syntax_detail = "SyntaxError in generated code (unknown location)"
            except SyntaxError as _se:
                syntax_detail = f"SyntaxError at line {_se.lineno}: {_se.msg}"
            return SkillExecutionOutcome(
                success=False,
                result=None,
                error=syntax_detail,
                error_type=ErrorType.INPUT_INVALID,
                skill_name=skill_name,
            )

        resolved_cwd = Path(cwd).resolve()
        data_dir = g_config.get_data_dir().resolve()
        sys_tmp = Path(tempfile.gettempdir()).resolve()

        if g_config.paths.path_validation_enabled:
            if not (
                resolved_cwd.is_relative_to(data_dir)
                or resolved_cwd.is_relative_to(sys_tmp)
            ):
                return SkillExecutionOutcome(
                    success=False,
                    result=None,
                    error=f"Sandbox: cwd '{cwd}' is outside safe boundaries",
                    skill_name=skill_name,
                )

        extra_env: dict[str, str] = {}
        if pythonpath is not None:
            extra_env["PYTHONPATH"] = str(pythonpath)
        sandbox_env = self.get_sandbox_env(extra_env if extra_env else None)

        if env:
            sandbox_env.update(env)

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                env=sandbox_env,
                capture_output=True,
                timeout=timeout,
                **SUBPROCESS_TEXT_KWARGS,
            )
        except subprocess.TimeoutExpired:
            return SkillExecutionOutcome(
                success=False,
                result=None,
                error=f"Execution timed out after {timeout}s",
                skill_name=skill_name,
            )

        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        if proc.returncode != 0:
            return SkillExecutionOutcome(
                success=False,
                result=stdout or None,
                error=self._format_error(proc.returncode, stdout, stderr),
                skill_name=skill_name,
            )

        if self._stderr_has_real_errors(stderr):
            return SkillExecutionOutcome(
                success=False,
                result=stdout or None,
                error=f"Execution stderr indicates error:\n{stderr[:_STDERR_TRUNCATE_LEN]}",
                skill_name=skill_name,
            )

        if self._stdout_indicates_error(stdout):
            return SkillExecutionOutcome(
                success=False,
                result=None,
                error=f"Execution output indicates error:\n{stdout[:_STDOUT_TRUNCATE_LEN]}",
                skill_name=skill_name,
            )

        return SkillExecutionOutcome(success=True, result=stdout, skill_name=skill_name)

    def run_code(
        self,
        code: str,
        skill: Skill,
        deps: list[str] | None = None,
        session_id: str = "",
    ) -> SkillExecutionOutcome:
        """执行代码。"""
        resolved_session_id = session_id or "default"
        work_dir = ArtifactManager.get_sandbox_dir(skill.name, resolved_session_id)
        work_dir.mkdir(parents=True, exist_ok=True)
        return self._run_code_in(code, skill, deps, resolved_session_id, work_dir)

    def _run_code_in(
        self,
        code: str,
        skill: Skill,
        deps: list[str] | None,
        session_id: str,
        work_dir: Path,
    ) -> SkillExecutionOutcome:
        """在指定工作目录执行代码。"""
        if deps:
            logger.info("Installing dependencies for '{}': {}", skill.name, deps)
            pip_timeout = g_config.skills.execution.pip_install_timeout_sec
            success, error_msg = self.install_python_deps(deps, timeout=pip_timeout)
            if not success:
                logger.error(
                    "Failed to install dependencies for '{}': {}", skill.name, error_msg
                )
                return SkillExecutionOutcome(
                    success=False,
                    result=None,
                    error=f"Failed to install dependencies: {error_msg}",
                    error_type=ErrorType.DEPENDENCY_ERROR,
                    error_detail={"deps": deps, "message": error_msg},
                    skill_name=skill.name,
                )
            logger.info("Dependencies installed successfully for '{}'", skill.name)

        try:
            self._prepare_workspace(skill, work_dir)
            pre_files = ArtifactManager.snapshot_files(work_dir)
            runner_path = work_dir / "__runner__.py"
            runner_path.write_text(code, encoding="utf-8")

            logger.info("Sandbox executing '{}' in {}", skill.name, work_dir)

            result = self.run(
                [str(self._python_executable), str(runner_path)],
                cwd=work_dir,
                pythonpath=work_dir,
                timeout=g_config.skills.execution.timeout_sec,
                skill_name=skill.name,
                check_syntax=code,
            )

            if not result.success:
                missing = extract_missing_module_from_error(result.error or "")
                if missing:
                    hint = f"uv pip install {missing}"
                    return SkillExecutionOutcome(
                        success=False,
                        result=result.result,
                        error=f"Missing module '{missing}'. Run: {hint}",
                        error_type=ErrorType.DEPENDENCY_ERROR,
                        error_detail={
                            "deps": [missing],
                            "install_hint": hint,
                            "message": result.error,
                        },
                        skill_name=skill.name,
                    )
                return result

            artifacts = ArtifactManager.collect_local_artifacts(
                work_dir, pre_files, skill.name, session_id=session_id
            )
            logger.info("Sandbox success for '{}'", skill.name)
            return SkillExecutionOutcome(
                success=True,
                result=result.result,
                skill_name=skill.name,
                artifacts=artifacts,
            )
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            logger.error("Sandbox error for '{}': {}", skill.name, e)
            return SkillExecutionOutcome(
                success=False,
                result=None,
                error=error_msg,
                error_type=ErrorType.INTERNAL_ERROR,
                error_detail={"message": error_msg},
                skill_name=skill.name,
            )

    def _prepare_workspace(self, skill: Skill, work_dir: Path):
        """准备工作空间。"""
        has_files = False

        if skill.source_dir:
            source = Path(skill.source_dir)
            scripts_dir = g_config.get_skill_scripts_path(source)
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

    def install_python_deps(
        self, deps: list[str], timeout: int | None = None
    ) -> tuple[bool, str]:
        """在 venv 中安装 Python 依赖，镜像由 env 中的 UV_INDEX_URL 等控制。"""
        if not deps:
            return True, ""

        if timeout is None:
            timeout = g_config.skills.execution.pip_install_timeout_sec

        # 获取环境变量（包含镜像配置）
        install_env = self._get_uv_env()
        install_env["UV_HTTP_TIMEOUT"] = "15"

        # 调试：打印环境变量中的镜像配置
        logger.info(
            "[UV Mirror] UV_PIP_INDEX_URL: {}",
            install_env.get("UV_PIP_INDEX_URL", "Not set"),
        )
        logger.info(
            "[UV Mirror] PIP_INDEX_URL: {}", install_env.get("PIP_INDEX_URL", "Not set")
        )

        # 构建命令，优先使用命令行参数（更可靠）
        cmd = [
            str(self._uv_bin),
            "pip",
            "install",
            "--python",
            str(self._python_executable),
        ]

        # 添加镜像参数（优先使用 UV_PIP_INDEX_URL，否则用 PIP_INDEX_URL）
        index_url = install_env.get("UV_PIP_INDEX_URL") or install_env.get(
            "PIP_INDEX_URL"
        )
        if index_url:
            cmd.extend(["--index-url", index_url])
            logger.info("[UV Mirror] Using index-url: {}", index_url)

        extra_index = install_env.get("UV_PIP_EXTRA_INDEX_URL") or install_env.get(
            "PIP_EXTRA_INDEX_URL"
        )
        if extra_index:
            for url in extra_index.split():
                cmd.extend(["--extra-index-url", url])
            logger.info("[UV Mirror] Using extra-index-url: {}", extra_index)

        cmd.extend(deps)

        logger.info("Installing deps via uv: {}", deps)
        logger.info("cmd: {}", cmd)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=timeout,
                env=install_env,
                **SUBPROCESS_TEXT_KWARGS,
            )

            if result.returncode == 0:
                logger.info("Dependencies installed: {}", deps)
                return True, ""

            last_error = result.stderr[:_INSTALL_STDERR_TRUNCATE_LEN]
            logger.warning("uv pip install failed: {}", last_error)
            return False, f"uv pip install failed: {last_error}"

        except subprocess.TimeoutExpired:
            error = f"Timeout ({timeout}s) installing dependencies: {deps}"
            logger.warning(error)
            return False, error
        except Exception as e:
            error = f"Error installing dependencies: {e}"
            logger.warning(error)
            return False, error

    @staticmethod
    def _format_error(returncode: int, stdout: str, stderr: str) -> str:
        parts = [f"Exit code: {returncode}"]
        if stderr:
            parts.append(f"Stderr:\n{stderr[:_ERROR_MSG_TRUNCATE_LEN]}")
        if stdout:
            parts.append(f"Stdout:\n{stdout[:_STDOUT_TRUNCATE_LEN]}")
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


__all__ = ["UvLocalSandbox"]
