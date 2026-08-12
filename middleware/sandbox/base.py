# SPDX-License-Identifier: Apache-2.0
"""Sandbox base classes and factory."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any

from .schema import SandboxExecutionOutcome


class RuntimeType(str, Enum):
    """Runtime type for sandbox selection."""

    PYTHON = "python"
    NODE = "node"
    AUTO = "auto"


# Detection patterns for Node.js commands
_NODE_CMD_PATTERNS: re.Pattern[re.RegexFlag] = re.compile(
    r"^\s*(?:"
    r"npm\s|"
    r"pnpm\s|"
    r"yarn\s|"
    r"bun\s|"
    r"node\s|"
    r"deno\s|"
    r"npx\s|"
    r"tsx\s|"
    r"ts-node\s|"
    r"vite\s|"
    r"esbuild\s|"
    r"webpack\s|"
    r"rollup\s|"
    r"parcel\s|"
    r"eslint\s|"
    r"prettier\s"
    r")",
    re.IGNORECASE,
)

# File extension patterns for JS/TS/TSX detection
_NODE_FILE_PATTERNS: re.Pattern[re.RegexFlag] = re.compile(
    r"\.(?:js|jsx|ts|tsx|mjs|cjs)$",
    re.IGNORECASE,
)


def detect_runtime(command: str) -> RuntimeType:
    """Auto-detect the runtime type from a shell command.

    Returns:
        RuntimeType.NODE if the command appears to be a Node.js ecosystem command,
        RuntimeType.PYTHON otherwise.
    """
    # Check for JS/TS file path references
    if _NODE_FILE_PATTERNS.search(command):
        return RuntimeType.NODE

    # Check for Node.js ecosystem commands
    if _NODE_CMD_PATTERNS.match(command):
        return RuntimeType.NODE

    return RuntimeType.PYTHON


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


def get_sandbox(runtime: RuntimeType = RuntimeType.PYTHON) -> BaseSandbox:
    """Get a sandbox instance for the given runtime type.

    Args:
        runtime: The runtime type (python/node/auto). Default is PYTHON.

    Returns:
        A sandbox instance. For PYTHON returns UvLocalSandbox.
        For NODE returns NodeSandbox.
        For AUTO, callers should use detect_runtime() first and pass the result.
    """
    if runtime == RuntimeType.NODE:
        from .node_sandbox import NodeSandbox

        return NodeSandbox()

    # Default: Python sandbox
    from .uv import UvLocalSandbox

    return UvLocalSandbox()
