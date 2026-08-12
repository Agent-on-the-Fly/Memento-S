# SPDX-License-Identifier: Apache-2.0
"""UV 本地沙箱 - UvLocalSandbox 实现。"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from middleware.config import g_config
from utils.logger import get_logger
from shared.fs.snapshot import SandboxSnapshot as SandboxArtifactCollector
from .base import BaseSandbox
from .env_builder import build_env
from .schema import ErrorType, SandboxExecutionOutcome
from middleware.utils.platform import (
    SUBPROCESS_TEXT_KWARGS,
    venv_bin_dir as _venv_bin_dir,
    venv_python as _venv_python,
    pip_shim_path,
    pip_shim_content,
    chmod_executable,
    uv_install_hint,
)
from shared.tools.dependency_aliases import normalize_dependency_spec

logger = get_logger(__name__)

# UV 静默环境变量 - 禁用所有进度条和彩色输出
_UV_QUIET_ENV: dict[str, str] = {
    "UV_NO_PROGRESS": "1",
    "UV_NO_COLOR": "1",
    "UV_QUIET": "1",
}

_STDERR_TRUNCATE_LEN = 2000


def _get_hidden_subprocess_startupinfo() -> subprocess.STARTUPINFO | None:
    """获取隐藏子进程窗口的 STARTUPINFO。

    Returns:
        Windows 平台的 STARTUPINFO 对象，非 Windows 返回 None
    """
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        return startupinfo
    return None


def _get_hidden_subprocess_creationflags() -> int:
    """获取隐藏子进程窗口的 creationflags。

    Returns:
        Windows 的 CREATE_NO_WINDOW 标志，非 Windows 返回 0
    """
    if sys.platform == "win32":
        return subprocess.CREATE_NO_WINDOW
    return 0


def _build_uv_env() -> dict[str, str]:
    """构建 uv 命令的环境变量，添加静默设置。

    Returns:
        包含静默环境变量的字典
    """
    env = build_env(full_system_env=True)
    env.update(_UV_QUIET_ENV)
    return env
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


class UvLocalSandbox(BaseSandbox):
    """使用 UV 管理的隔离虚拟环境沙箱。"""

    def __init__(self):
        self._uv_bin: Path | None = None
        self._venv_path: Path | None = None
        self._python_executable: Path | None = None
        self._installed_deps: set[str] = set()
        self._ensure_uv_installed()
        self._setup_venv()

    @property
    def python_executable(self) -> Path:
        return self._python_executable

    @property
    def venv_path(self) -> Path:
        return self._venv_path

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
            self._installed_deps.clear()
            self._create_venv(python_version)
        else:
            logger.debug("Using existing venv at {}", self._venv_path)

        self._python_executable = _venv_python(self._venv_path)

        if not self._python_executable.exists():
            raise RuntimeError(
                f"Python executable not found at {self._python_executable}"
            )

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
            "-q",  # 静默模式，避免终端闪烁
            "--no-progress",  # 禁用进度条
            "venv",
            str(self._venv_path),
            "--python",
            python_version,
        ]

        startupinfo = _get_hidden_subprocess_startupinfo()
        creationflags = _get_hidden_subprocess_creationflags()
        env = _build_uv_env()

        try:
            subprocess.run(
                cmd,
                capture_output=True,
                check=True,
                env=env,
                startupinfo=startupinfo,
                creationflags=creationflags,
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
        注意：不使用 install_python_deps() 安装 pip，避免递归创建新 sandbox 实例。
        """
        pip_path = pip_shim_path(self._venv_path)
        pip_path.write_text(pip_shim_content(self._python_executable))
        chmod_executable(pip_path)
        logger.debug("Created pip shim at {}", pip_path)

        startupinfo = _get_hidden_subprocess_startupinfo()
        creationflags = _get_hidden_subprocess_creationflags()

        # 验证 pip shim 是否可用
        pip_executable = pip_shim_path(self._venv_path)
        result = subprocess.run(
            [str(pip_executable), "--version"],
            capture_output=True,
            shell=False,
            startupinfo=startupinfo,
            creationflags=creationflags,
            **SUBPROCESS_TEXT_KWARGS,
        )
        if result.returncode == 0:
            # logger.debug("pip shim is functional: {}", result.stdout.strip())
            self._installed_deps.add("pip")
        else:
            logger.debug("pip shim not yet functional (non-fatal): {}", result.stderr.strip())

    def install_python_deps(
        self,
        deps: list[str],
        timeout: int = 60,
    ) -> tuple[bool, str]:
        """安装依赖。"""
        if not deps:
            return True, ""

        # Normalize dependency names via centralized alias table.
        normalized_deps: list[str] = []
        for dep in deps:
            normalized = normalize_dependency_spec(dep)
            if not normalized:
                continue
            if normalized != dep:
                logger.debug("Dependency normalized: '{}' -> '{}'", dep, normalized)
            normalized_deps.append(normalized)

        deps = normalized_deps

        # Cross-platform special handling for python-magic
        # Windows requires python-magic-bin (includes libmagic DLL)
        # Linux/Mac can use python-magic (requires system libmagic)
        for i, dep in enumerate(deps):
            if dep.startswith("python-magic") and not dep.startswith(
                "python-magic-bin"
            ):
                if platform.system() == "Windows":
                    deps[i] = "python-magic-bin"
                    logger.debug(
                        "Cross-platform fix: 'python-magic' -> 'python-magic-bin' for Windows"
                    )
        # Skip already-installed deps (cache within this sandbox instance).
        to_install = [d for d in deps if d not in self._installed_deps]
        if not to_install:
            return True, ""

        cmd = [str(self._uv_bin), "-q", "pip", "install", *to_install]

        startupinfo = _get_hidden_subprocess_startupinfo()
        creationflags = _get_hidden_subprocess_creationflags()

        logger.info("Installing dependencies: {}", to_install)
        env = _build_uv_env()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                check=False,
                env=env,
                timeout=timeout,
                startupinfo=startupinfo,
                creationflags=creationflags,
                **SUBPROCESS_TEXT_KWARGS,
            )
        except subprocess.TimeoutExpired:
            return False, f"Dependency install timed out after {timeout}s"

        stderr = proc.stderr.strip()
        if proc.returncode != 0:
            if stderr:
                if len(stderr) > _INSTALL_STDERR_TRUNCATE_LEN:
                    stderr = stderr[:_INSTALL_STDERR_TRUNCATE_LEN] + "..."
                return False, stderr
            return False, f"uv pip install failed (code {proc.returncode})"

        logger.info("Dependencies installed successfully: {} (cached: {})", to_install, len([d for d in to_install if d in self._installed_deps]))
        self._installed_deps.update(to_install)
        return True, ""

    def execute_shell(
        self,
        command: str,
        extra_env: dict[str, str] | None = None,
        work_dir: Path | None = None,
        timeout: int = 300,
        collect_artifacts: bool = False,
        session_id: str = "",
    ) -> SandboxExecutionOutcome:
        """Execute a shell command in the sandbox environment."""
        # Build environment with sandbox variables
        env = build_env(
            venv_path=self._venv_path,
            python_executable=self._python_executable,
            extra=extra_env,
        )

        # Handle artifact collection
        pre_files = None
        effective_work_dir = work_dir

        if collect_artifacts:
            # Unify artifact collection under @ROOT (workspace_dir), not .sandbox
            workspace_root = Path(g_config.paths.workspace_dir).resolve()
            workspace_root.mkdir(parents=True, exist_ok=True)

            # Snapshot files before execution
            pre_files = SandboxArtifactCollector.take(workspace_root)
            effective_work_dir = workspace_root

        startupinfo = _get_hidden_subprocess_startupinfo()
        creationflags = _get_hidden_subprocess_creationflags()
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=effective_work_dir,
                env=env,
                timeout=timeout,
                startupinfo=startupinfo,
                creationflags=creationflags,
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            # Collect artifacts if requested
            artifacts = []
            if collect_artifacts and pre_files and effective_work_dir:
                artifacts = SandboxArtifactCollector.collect_diff(
                    pre_files,
                    effective_work_dir,
                )

            if result.returncode != 0:
                return SandboxExecutionOutcome(
                    success=False,
                    result=stdout or None,
                    error=f"Command failed with code {result.returncode}\nSTDERR:\n{stderr[:_STDERR_TRUNCATE_LEN]}",
                    error_type=ErrorType.EXECUTION_ERROR,
                    skill_name="shell",
                    artifacts=artifacts,
                )

            return SandboxExecutionOutcome(
                success=True,
                result=stdout,
                skill_name="shell",
                artifacts=artifacts,
            )

        except subprocess.TimeoutExpired:
            return SandboxExecutionOutcome(
                success=False,
                result=None,
                error=f"Command timed out after {timeout}s",
                error_type=ErrorType.TIMEOUT,
                skill_name="shell",
            )
        except Exception as e:
            return SandboxExecutionOutcome(
                success=False,
                result=None,
                error=f"{type(e).__name__}: {e}",
                error_type=ErrorType.INTERNAL_ERROR,
                skill_name="shell",
            )

    def run(
        self,
        cmd: list[str],
        cwd: Path,
        pythonpath: Path | None = None,
        timeout: int = 120,
        skill_name: str | None = None,
        check_syntax: str | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> SandboxExecutionOutcome:
        """执行 python 子进程。"""
        env = build_env(
            venv_path=self._venv_path, python_executable=self._python_executable
        )
        if pythonpath:
            env["PYTHONPATH"] = str(pythonpath)
        # ENV VAR JAIL: Inject extra environment variables into subprocess
        if extra_env:
            env.update(extra_env)

        startupinfo = _get_hidden_subprocess_startupinfo()
        creationflags = _get_hidden_subprocess_creationflags()

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                env=env,
                capture_output=True,
                timeout=timeout,
                startupinfo=startupinfo,
                creationflags=creationflags,
                **SUBPROCESS_TEXT_KWARGS,
            )
        except subprocess.TimeoutExpired:
            return SandboxExecutionOutcome(
                success=False,
                result=None,
                error=f"Execution timed out after {timeout}s",
                skill_name=skill_name,
            )

        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        if proc.returncode != 0:
            return SandboxExecutionOutcome(
                success=False,
                result=stdout or None,
                error=self._format_error(proc.returncode, stdout, stderr),
                skill_name=skill_name,
            )

        if self._stderr_has_real_errors(stderr):
            return SandboxExecutionOutcome(
                success=False,
                result=stdout or None,
                error=f"Execution stderr indicates error:\n{stderr[:_STDERR_TRUNCATE_LEN]}",
                skill_name=skill_name,
            )

        if self._stdout_indicates_error(stdout):
            return SandboxExecutionOutcome(
                success=False,
                result=None,
                error=f"Execution output indicates error:\n{stdout[:_STDOUT_TRUNCATE_LEN]}",
                skill_name=skill_name,
            )

        return SandboxExecutionOutcome(
            success=True, result=stdout, skill_name=skill_name
        )

    def run_code(
        self,
        code: str,
        name: str = "python_exec",
        deps: list[str] | None = None,
        session_id: str = "",
        source_dir: str | None = None,
        work_dir: str | Path | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> SandboxExecutionOutcome:
        """执行代码。

        Args:
            code: Python code to execute
            name: Execution name/identifier
            deps: Optional dependencies to install
            session_id: Session identifier for sandbox paths
            source_dir: Optional source directory to copy to workspace
            work_dir: Optional execution working directory (preferred: per-run @ROOT)
        """
        resolved_session_id = session_id or "default"

        if work_dir is None:
            return SandboxExecutionOutcome(
                success=False,
                result=None,
                error="run_code requires explicit work_dir (@ROOT/run_dir).",
                error_type=ErrorType.INPUT_INVALID,
                error_detail={
                    "category": "path",
                    "message": "Missing work_dir for python execution",
                    "hint": "Pass per-run @ROOT as work_dir.",
                    "retryable": False,
                },
                skill_name=name,
            )

        target_work_dir = Path(work_dir).resolve()
        target_work_dir.mkdir(parents=True, exist_ok=True)
        return self._run_code_in(
            code,
            name,
            deps,
            resolved_session_id,
            target_work_dir,
            source_dir,
            extra_env,
        )

    def _run_code_in(
        self,
        code: str,
        name: str,
        deps: list[str] | None,
        session_id: str,
        work_dir: Path,
        source_dir: str | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> SandboxExecutionOutcome:
        """在指定工作目录执行代码。"""
        if deps:
            logger.info("Installing dependencies for '{}': {}", name, deps)
            pip_timeout = g_config.skills.execution.pip_install_timeout_sec
            success, error_msg = self.install_python_deps(deps, timeout=pip_timeout)
            if not success:
                logger.error(
                    "Failed to install dependencies for '{}': {}", name, error_msg
                )
                return SandboxExecutionOutcome(
                    success=False,
                    result=None,
                    error=f"Failed to install dependencies: {error_msg}",
                    error_type=ErrorType.DEPENDENCY_ERROR,
                    error_detail={"deps": deps, "message": error_msg},
                    skill_name=name,
                )
            logger.info("Dependencies installed successfully for '{}'", name)

        try:
            if source_dir:
                self._prepare_workspace(source_dir, work_dir)
            pre_files = SandboxArtifactCollector.take(work_dir)
            runner_path = work_dir / f"__runner__{session_id}.py"
            runner_path.write_text(code, encoding="utf-8")

            logger.info("Sandbox executing '{}' in {}", name, work_dir)

            _timeout = getattr(
                g_config.skills.execution, "bash_timeout_sec", 300
            ) or 300
            result = self.run(
                [str(self._python_executable), str(runner_path)],
                cwd=work_dir,
                pythonpath=work_dir,
                timeout=_timeout,
                skill_name=name,
                check_syntax=code,
                extra_env=extra_env,
            )

            if not result.success:
                return SandboxExecutionOutcome(
                    success=False,
                    result=result.result,
                    error=result.error,
                    error_type=ErrorType.EXECUTION_ERROR,
                    error_detail={"message": result.error},
                    skill_name=name,
                )

            artifacts = SandboxArtifactCollector.collect_diff(
                pre_files,
                work_dir,
            )
            logger.info("Sandbox success for '{}'", name)
            return SandboxExecutionOutcome(
                success=True,
                result=result.result,
                skill_name=name,
                artifacts=artifacts,
            )
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            logger.error("Sandbox error for '{}': {}", name, e)
            return SandboxExecutionOutcome(
                success=False,
                result=None,
                error=error_msg,
                error_type=ErrorType.INTERNAL_ERROR,
                error_detail={"message": error_msg},
                skill_name=name,
            )

    def _prepare_workspace(self, source_dir: str, work_dir: Path) -> None:
        """拷贝技能目录到沙箱执行目录。"""
        if not source_dir:
            return
        src = Path(source_dir)
        if not src.exists():
            return

        dest = work_dir / "skill"
        if dest.exists():
            shutil.rmtree(dest)

        shutil.copytree(
            src,
            dest,
            ignore=shutil.ignore_patterns("*.pyc", "__pycache__"),
        )

    def _stderr_has_real_errors(self, stderr: str) -> bool:
        if not stderr:
            return False
        lower = stderr.lower()
        return any(prefix in lower for prefix in _ERROR_PREFIXES)

    def _stdout_indicates_error(self, stdout: str) -> bool:
        if not stdout:
            return False
        lower = stdout.lower()
        return any(prefix in lower for prefix in _ERROR_PREFIXES)

    def _format_error(self, returncode: int, stdout: str, stderr: str) -> str:
        parts = [f"Process exited with code {returncode}."]
        if stdout:
            stdout_trunc = stdout[:_STDOUT_TRUNCATE_LEN]
            parts.append(f"STDOUT:\n{stdout_trunc}")
        if stderr:
            stderr_trunc = stderr[:_STDERR_TRUNCATE_LEN]
            parts.append(f"STDERR:\n{stderr_trunc}")
        return "\n".join(parts)
