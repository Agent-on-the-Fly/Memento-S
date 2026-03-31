"""Sandbox base classes and factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .schema import SandboxExecutionOutcome


class BaseSandbox(ABC):
    """Base class for all sandbox implementations."""

    @property
    @abstractmethod
    def python_executable(self) -> Path:
        """Return the path to the Python executable in the sandbox."""
        raise NotImplementedError

    @property
    @abstractmethod
    def venv_path(self) -> Path:
        """Return the path to the virtual environment."""
        raise NotImplementedError

    @abstractmethod
    def run_code(
        self,
        code: str,
        name: str = "python_exec",
        deps: list[str] | None = None,
        session_id: str = "",
        source_dir: str | None = None,
    ) -> SandboxExecutionOutcome:
        """Execute Python code in the sandbox.

        Args:
            code: Python code to execute
            name: Execution name/identifier
            deps: Optional dependencies to install
            session_id: Session identifier for sandbox paths
            source_dir: Optional source directory to copy to workspace

        Returns:
            SandboxExecutionOutcome with execution results
        """
        raise NotImplementedError

    @abstractmethod
    def install_python_deps(
        self,
        deps: list[str],
        timeout: int = 60,
    ) -> tuple[bool, str]:
        """Install Python dependencies in the sandbox."""
        raise NotImplementedError

    @abstractmethod
    def execute_shell(
        self,
        command: str,
        extra_env: dict[str, str] | None = None,
        work_dir: Path | None = None,
        timeout: int = 300,
        collect_artifacts: bool = False,
        session_id: str = "",
    ) -> SandboxExecutionOutcome:
        """Execute a shell command in the sandbox environment.

        Args:
            command: Shell command to execute
            extra_env: Additional environment variables
            work_dir: Working directory for the command
            timeout: Timeout in seconds
            collect_artifacts: Whether to collect generated files as artifacts
            session_id: Session identifier for artifact collection

        Returns:
            SandboxExecutionOutcome with execution results and artifacts
        """
        raise NotImplementedError


def get_sandbox() -> BaseSandbox:
    """Get the default sandbox instance."""
    from .uv import UvLocalSandbox

    return UvLocalSandbox()
