"""Bash command execution tool with sandbox environment support."""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any

from ._base import _resolve_path
from core.utils.platform import has_bash, has_powershell
from middleware.config import g_config


async def bash_tool(
    command: str,
    env: dict[str, str] | None = None,
    base_dir: str | None = None,
    work_dir: str | None = None,
    stdin: str | None = None,
) -> str:
    """
    Execute a shell command in the workspace.

    This tool supports sandbox isolation - if a sandbox is configured,
    it will automatically use the sandbox's virtual environment.

    IMPORTANT: This is a STATELESS environment. Environment variables or `cd`
    will not persist across calls. Use `&&` to chain commands (e.g., `cd src && ls`).
    Interactive commands (like vim, nano, top) are strictly prohibited.

    Args:
        command: The shell command to run.
        env: Optional custom environment variables to inject.
        base_dir: Optional base directory for path resolution (security boundary).
        work_dir: Optional working directory for command execution.
        stdin: Optional standard input to pass to the command.
    """
    try:
        # Build environment: system env + sandbox env + custom env
        safe_env = _build_sandbox_env(env)

        cwd = None
        if work_dir:
            cwd = _resolve_path(work_dir, Path(base_dir) if base_dir else None)
        elif base_dir:
            cwd = _resolve_path(".", Path(base_dir))
        else:
            cwd = Path(g_config.paths.workspace_dir)

        # Ensure python resolves to uv sandbox python when available
        if "UV_PYTHON" in safe_env:
            print(
                f"bash_tool: UV_PYTHON Running command: {command}! safe_env: {safe_env}"
            )
            command = _rewrite_python_command(command, safe_env["UV_PYTHON"])

        # Rewrite uv pip install to use mirror (more reliable than env vars)
        command = _rewrite_uv_pip_command(command, safe_env)

        shell_args = _select_shell(command)

        print(
            f"bash_tool: Running command: {command}! shell_args: {shell_args}, safe_env: {safe_env}"
        )

        if shell_args is None:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE if stdin is not None else None,
                cwd=cwd,
                env=safe_env,
            )
        else:
            proc = await asyncio.create_subprocess_exec(
                *shell_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE if stdin is not None else None,
                cwd=cwd,
                env=safe_env,
            )

        try:
            input_bytes = stdin.encode("utf-8") if stdin is not None else None
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input_bytes), timeout=300
            )
        except asyncio.TimeoutError:
            proc.kill()
            from core.utils.platform import background_hint

            return f"ERR: Command timed out after 300s. If starting a server, run it in background with '{background_hint()}'"

        out = stdout.decode("utf-8", errors="replace").strip()
        err = stderr.decode("utf-8", errors="replace").strip()

        if len(out) > 50000:
            out = out[:50000] + "\n... [STDOUT TRUNCATED]"
        if len(err) > 50000:
            err = err[:50000] + "\n... [STDERR TRUNCATED]"

        if proc.returncode != 0:
            return f"EXIT CODE: {proc.returncode}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
        return f"STDOUT:\n{out}" if out else "SUCCESS: (No output)"

    except Exception as e:
        return f"ERR: bash execution failed: {e}"


def _build_sandbox_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """构建沙箱环境变量。

    优先级（从低到高）:
    1. 系统环境变量（白名单）
    2. 配置文件中的环境变量（pip 镜像、providers 等）
    3. Sandbox 虚拟环境（如果可用，已包含步骤 2）
    4. 调用者传入的自定义环境变量

    Returns:
        环境变量字典
    """
    from core.utils.platform import filter_env_by_whitelist

    env = filter_env_by_whitelist()

    # 2 & 3. 注入 Sandbox 环境（如果可用，已含 _config_env_vars）
    sandbox_ok = False
    try:
        from core.skill.execution.sandbox import get_sandbox

        sandbox = get_sandbox()
        if hasattr(sandbox, "get_sandbox_env"):
            sandbox_env = sandbox.get_sandbox_env()
            env.update(sandbox_env)
            sandbox_ok = True
    except Exception:
        pass

    # Sandbox 不可用时，仍注入配置中的环境变量（pip 镜像等）
    if not sandbox_ok:
        try:
            from core.skill.execution.sandbox.uv import _config_env_vars

            env.update(_config_env_vars())
        except Exception:
            pass

    # 3. 注入全局配置 env（优先级高于系统，低于调用者）
    try:
        from middleware.config import g_config

        cfg_env = getattr(g_config, "env", None)
        if isinstance(cfg_env, dict):
            env.update({str(k): str(v) for k, v in cfg_env.items()})
    except Exception:
        pass

    # 4. 注入调用者传入的自定义环境（最高优先级）
    if extra_env:
        env.update(extra_env)

    return env


def _select_shell(command: str) -> list[str] | None:
    if os.name == "nt":
        return None

    if has_bash():
        return ["bash", "-lc", command]

    return ["sh", "-c", command]


def _rewrite_python_command(command: str, python_path: str) -> str:
    if not command:
        return command

    import re

    def _replace(match: re.Match) -> str:
        prefix = match.group("prefix") or ""
        cmd = match.group("cmd")
        rest = match.group("rest") or ""
        if cmd == "pip":
            return f'{prefix}"{python_path}" -m pip{rest}'
        return f'{prefix}"{python_path}"{rest}'

    pattern = re.compile(
        r"(?P<prefix>^|[;&|]\s*)(?P<cmd>python3|python|py|pip)(?P<rest>\s+|-m\s+|$)"
    )

    return pattern.sub(_replace, command)


def _rewrite_uv_pip_command(command: str, env: dict[str, str]) -> str:
    """Rewrite uv pip install command to use mirror index URLs."""
    if not command or "uv pip install" not in command:
        return command

    # Already has --index-url, skip
    if "--index-url" in command:
        return command

    # Get mirror URLs from environment
    index_url = env.get("UV_PIP_INDEX_URL") or env.get("PIP_INDEX_URL")
    extra_index = env.get("UV_PIP_EXTRA_INDEX_URL") or env.get("PIP_EXTRA_INDEX_URL")

    if not index_url:
        return command

    # Build mirror arguments
    mirror_args = f'--index-url "{index_url}"'
    if extra_index:
        for url in extra_index.split():
            mirror_args += f' --extra-index-url "{url}"'

    # Insert after "uv pip install"
    import re

    # Match: uv pip install [options] package...
    pattern = r"(uv\s+pip\s+install)(\s+)(.*)"

    def _replace(match: re.Match) -> str:
        cmd = match.group(1)
        space = match.group(2)
        rest = match.group(3)
        return f"{cmd}{space}{mirror_args} {rest}"

    return re.sub(pattern, _replace, command)
