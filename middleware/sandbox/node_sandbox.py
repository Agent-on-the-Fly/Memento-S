# SPDX-License-Identifier: Apache-2.0
"""Node.js sandbox — simplified, no .nvenv directory required.

Uses the system Node.js and relies on the Node.js ecosystem's native module
resolution (向上回溯 node_modules) for dependency management. Skills that
need npm packages should include a package.json in their scripts/ directory
and install via npm install.
"""

from __future__ import annotations

import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Literal

from middleware.config import g_config
from utils.logger import get_logger
from shared.fs.snapshot import SandboxSnapshot as SandboxArtifactCollector
from .node_env import build_node_path_from_dir
from .schema import ErrorType, SandboxExecutionOutcome
from middleware.utils.platform import SUBPROCESS_TEXT_KWARGS

logger = get_logger(__name__)

_ERROR_PREFIXES = (
    "error:",
    "error ",
    "fatal:",
    "failed:",
    "npm error",
    "pnpm error",
    "yarn error",
    "bun error",
)

_INSTALL_STDERR_TRUNCATE_LEN = 500


class NodeSandbox:
    """Node.js sandbox using the system Node.js runtime."""

    def __init__(self):
        self._node_bin: Path | None = None
        self._package_manager: Literal["npm", "pnpm", "yarn", "bun"] = "npm"
        self._ensure_node_available()

    @property
    def node_executable(self) -> Path | None:
        """Return the path to the system Node.js executable."""
        return self._node_bin

    @property
    def venv_path(self) -> Path:
        """No venv concept for Node sandbox — return a placeholder."""
        return Path(".")

    def _ensure_node_available(self) -> Path:
        """Find the system Node.js binary.

        Returns the path to node if found, raises RuntimeError otherwise.
        """
        node_path = shutil.which("node")
        if node_path:
            self._node_bin = Path(node_path)
            logger.info("Using system node: {}", self._node_bin)
            return self._node_bin

        raise RuntimeError(
            "Node.js is not installed.\n"
            "Please install Node.js from https://nodejs.org or via your system package manager."
        )

    def install_packages(
        self,
        packages: list[str],
        package_manager: Literal["npm", "pnpm", "yarn", "bun"] = "npm",
        work_dir: Path | None = None,
        timeout: int = 120,
    ) -> tuple[bool, str]:
        """Install JavaScript/TypeScript packages.

        Packages are installed into the work_dir (or current directory) via
        npm/pnpm/yarn/bun. The Node.js module resolver will find them by walking
        up the directory tree from the script's location.
        """
        if not packages:
            return True, ""

        self._package_manager = package_manager
        install_dir = work_dir or Path.cwd()

        if package_manager == "npm":
            cmd = ["npm", "install", "--save"] + list(packages)
        elif package_manager == "pnpm":
            cmd = ["pnpm", "add"] + list(packages)
        elif package_manager == "yarn":
            cmd = ["yarn", "add"] + list(packages)
        elif package_manager == "bun":
            cmd = ["bun", "add"] + list(packages)
        else:
            return False, f"Unknown package manager: {package_manager}"

        logger.info("Installing JS packages with {}: {}", package_manager, packages)

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(install_dir),
                capture_output=True,
                timeout=timeout,
                **SUBPROCESS_TEXT_KWARGS,
            )
        except subprocess.TimeoutExpired:
            return False, f"Package install timed out after {timeout}s"

        stderr = proc.stderr.strip()
        if proc.returncode != 0:
            if stderr and len(stderr) > _INSTALL_STDERR_TRUNCATE_LEN:
                stderr = stderr[:_INSTALL_STDERR_TRUNCATE_LEN] + "..."
            return False, stderr or f"Install failed (code {proc.returncode})"

        logger.info("Packages installed successfully with {}", package_manager)
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
        """Execute a shell command in the Node sandbox environment.

        Runs the command with the system environment plus any extra_env vars.
        """
        import os

        env = dict(os.environ)
        if extra_env:
            env.update(extra_env)

        pre_files = None
        effective_work_dir = work_dir

        if collect_artifacts:
            workspace_root = Path(g_config.paths.workspace_dir).resolve()
            workspace_root.mkdir(parents=True, exist_ok=True)
            pre_files = SandboxArtifactCollector.take(workspace_root)
            effective_work_dir = workspace_root

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(effective_work_dir) if effective_work_dir else None,
                env=env,
                timeout=timeout,
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

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
                    error=f"Command failed with code {result.returncode}\nSTDERR:\n{stderr[:500]}",
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

    def run_code(
        self,
        code: str,
        name: str = "js_exec",
        deps: list[str] | None = None,
        session_id: str = "",
        source_dir: str | None = None,
        work_dir: str | Path | None = None,
        extra_env: dict[str, str] | None = None,
        runtime: Literal["node", "bun", "deno"] = "node",
    ) -> SandboxExecutionOutcome:
        """Execute JavaScript/TypeScript code.

        Supports .js, .ts, .tsx via the appropriate runtime.
        Dependencies are installed into the work_dir (npm install), where
        Node.js will find them via its native module resolution.

        If source_dir is provided (e.g. the skill's scripts/ directory),
        NODE_PATH is injected so that private vendor/ packages can be
        resolved without npm link.

        Args:
            code: The code content to execute
            name: Execution name/identifier (used to determine extension)
            deps: Optional npm packages to install before execution
            session_id: Session identifier for sandbox paths
            source_dir: Optional source directory to copy to workspace and
                use for NODE_PATH resolution (e.g. skill's scripts/ dir)
            work_dir: Execution working directory (required)
            extra_env: Additional environment variables
            runtime: Preferred runtime (node/bun/deno), auto-detected from name
        """
        import os

        resolved_session_id = session_id or "default"

        if work_dir is None:
            return SandboxExecutionOutcome(
                success=False,
                result=None,
                error="run_code requires explicit work_dir (@ROOT/run_dir).",
                error_type=ErrorType.INPUT_INVALID,
                error_detail={
                    "category": "path",
                    "message": "Missing work_dir for JS/TS execution",
                    "hint": "Pass per-run @ROOT as work_dir.",
                    "retryable": False,
                },
                skill_name=name,
            )

        target_work_dir = Path(work_dir).resolve()
        target_work_dir.mkdir(parents=True, exist_ok=True)

        # Determine file extension from name or detect from code
        ext = self._detect_extension(name, code)

        # Resolve the runtime binary
        runtime_bin = self._resolve_runtime(runtime, ext)

        # Install dependencies into work_dir
        if deps:
            logger.info("Installing JS dependencies for '{}': {}", name, deps)
            success, error_msg = self.install_packages(deps, timeout=120, work_dir=target_work_dir)
            if not success:
                logger.error("Failed to install dependencies for '{}': {}", name, error_msg)
                return SandboxExecutionOutcome(
                    success=False,
                    result=None,
                    error=f"Failed to install dependencies: {error_msg}",
                    error_type=ErrorType.DEPENDENCY_ERROR,
                    error_detail={"deps": deps, "message": error_msg},
                    skill_name=name,
                )
            logger.info("Dependencies installed successfully for '{}'", name)

        # Prepare workspace (copy skill dir to work_dir)
        if source_dir:
            self._prepare_workspace(source_dir, target_work_dir)
        pre_files = SandboxArtifactCollector.take(target_work_dir)

        # Write code to file
        runner_path = target_work_dir / f"__runner__{resolved_session_id}{ext}"
        runner_path.write_text(code, encoding="utf-8")

        logger.info("NodeSandbox executing '{}' in {} with {}", name, target_work_dir, runtime_bin)

        env = dict(os.environ)
        if extra_env:
            env.update(extra_env)

        # Inject NODE_PATH: collect vendor/ packages from the source_dir so that
        # private packages (e.g. scripts/vendor/my-pkg/) can be resolved without
        # npm link.  The source_dir is the skill's scripts/ directory, which may
        # contain vendor/ subdirectories alongside the main scripts.
        if source_dir:
            node_path_env = build_node_path_from_dir(Path(source_dir))
            if node_path_env:
                for k, v in node_path_env.items():
                    env.setdefault(k, v)

        _timeout = getattr(
            g_config.skills.execution, "bash_timeout_sec", 300
        ) or 300
        try:
            result = subprocess.run(
                [str(runtime_bin), str(runner_path)],
                cwd=str(target_work_dir),
                capture_output=True,
                text=True,
                env=env,
                timeout=_timeout,
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            if result.returncode != 0:
                return SandboxExecutionOutcome(
                    success=False,
                    result=stdout or None,
                    error=self._format_error(result.returncode, stdout, stderr),
                    error_type=ErrorType.EXECUTION_ERROR,
                    skill_name=name,
                )

            if stderr and self._stderr_has_real_errors(stderr):
                return SandboxExecutionOutcome(
                    success=False,
                    result=stdout or None,
                    error=f"Execution stderr indicates error:\n{stderr[:500]}",
                    error_type=ErrorType.EXECUTION_ERROR,
                    skill_name=name,
                )

            artifacts = SandboxArtifactCollector.collect_diff(
                pre_files,
                target_work_dir,
            )
            return SandboxExecutionOutcome(
                success=True,
                result=stdout,
                skill_name=name,
                artifacts=artifacts,
            )

        except subprocess.TimeoutExpired:
            return SandboxExecutionOutcome(
                success=False,
                result=None,
                error=f"Execution timed out after {_timeout}s",
                error_type=ErrorType.TIMEOUT,
                skill_name=name,
            )
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            logger.error("NodeSandbox error for '{}': {}", name, e)
            return SandboxExecutionOutcome(
                success=False,
                result=None,
                error=error_msg,
                error_type=ErrorType.INTERNAL_ERROR,
                error_detail={"message": error_msg},
                skill_name=name,
            )

    def _resolve_runtime(
        self,
        preferred: Literal["node", "bun", "deno"],
        ext: str,
    ) -> Path:
        """Resolve the appropriate runtime binary for the given extension."""
        # bun handles both .js and .ts/.tsx natively
        if preferred == "bun" or ext in (".ts", ".tsx"):
            bun_path = shutil.which("bun")
            if bun_path:
                return Path(bun_path)

        # deno handles both .js and .ts/.tsx natively
        if preferred == "deno" or ext in (".ts", ".tsx"):
            deno_path = shutil.which("deno")
            if deno_path:
                return Path(deno_path)

        # For TypeScript, try tsx
        if ext in (".ts", ".tsx"):
            tsx_path = shutil.which("tsx")
            if tsx_path:
                return Path(tsx_path)
            # tsx not found — fall back to node (ts files will fail with a clear error)
            logger.warning("tsx not available, falling back to node for {}", ext)

        # Default: node
        if self._node_bin is None:
            self._ensure_node_available()
        return self._node_bin

    def _detect_extension(self, name: str, code: str) -> str:
        """Detect file extension from execution name or code content."""
        name_lower = name.lower()
        if name_lower.endswith(".ts"):
            return ".ts"
        if name_lower.endswith(".tsx"):
            return ".tsx"
        if name_lower.endswith(".jsx"):
            return ".jsx"

        # Heuristic from code content
        if "import " in code and "from " in code and ("<" in code or "interface " in code or "type " in code):
            if "React" in code or "tsx" in code:
                return ".tsx"
            return ".ts"
        if "require(" in code or "module.exports" in code:
            return ".js"
        if "import " in code and "from " in code:
            return ".mjs"
        return ".js"

    def _prepare_workspace(self, source_dir: str, work_dir: Path) -> None:
        """Copy skill directory to sandbox execution directory."""
        import shutil as _shutil

        if not source_dir:
            return
        src = Path(source_dir)
        if not src.exists():
            return

        dest = work_dir / "skill"
        if dest.exists():
            _shutil.rmtree(dest)

        _shutil.copytree(
            src,
            dest,
            ignore=_shutil.ignore_patterns("*.pyc", "__pycache__", "node_modules"),
        )

    def _stderr_has_real_errors(self, stderr: str) -> bool:
        if not stderr:
            return False
        lower = stderr.lower()
        return any(prefix in lower for prefix in _ERROR_PREFIXES)

    def _format_error(self, returncode: int, stdout: str, stderr: str) -> str:
        parts = [f"Process exited with code {returncode}."]
        if stdout:
            parts.append(f"STDOUT:\n{stdout[:500]}")
        if stderr:
            parts.append(f"STDERR:\n{stderr[:500]}")
        return "\n".join(parts)

    def install_python_deps(
        self,
        deps: list[str],
        timeout: int = 60,
    ) -> tuple[bool, str]:
        """Not applicable for NodeSandbox — raises NotImplementedError."""
        raise NotImplementedError(
            "NodeSandbox.install_python_deps is not supported. "
            "Use NodeSandbox.install_packages() for JS/TS packages."
        )
