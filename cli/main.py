"""CLI for the Memento-S agent."""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root bootstrap – ensure ``core`` package is importable regardless
# of the working directory from which this script is invoked (e.g. running
# ``python cli/main.py`` from within the ``cli/`` directory or from the
# project root).  The project root is the parent of ``cli/``.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import typer
from rich.console import Console

from cli.commands import agent_command, doctor_command, feishu_bridge_command, verify_command

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("memento-s")
except Exception:
    __version__ = "0.1.0"

app = typer.Typer(name="MementoS", help="Memento-S Agent CLI", no_args_is_help=True)
console = Console()

_bootstrapped = False


def _ensure_bootstrap() -> None:
    global _bootstrapped
    if _bootstrapped:
        return
    _bootstrapped = True
    from bootstrap import bootstrap_sync

    bootstrap_sync()


@app.callback()
def _bootstrap_config() -> None:
    """CLI 启动时执行配置自检并加载配置。"""
    _ensure_bootstrap()


def memento_entry() -> None:
    """Console entrypoint: default to `agent` when no subcommand is provided."""
    _ensure_bootstrap()
    if len(sys.argv) == 1:
        sys.argv.append("agent")
    app()


@app.command()
def agent(
    message: str = typer.Option(
        None, "--message", "-m", help="Single message (non-interactive)"
    ),
    session_id: str | None = typer.Option(None, "--session", "-s", help="Session ID"),
    markdown: bool = typer.Option(
        True, "--markdown/--no-markdown", help="Render output as Markdown"
    ),
) -> None:
    """Chat with the Memento-S agent."""
    agent_command(
        message=message,
        session_id=session_id,
        markdown=markdown,
        version=__version__,
    )


@app.command()
def doctor() -> None:
    """Print configuration and environment info with formatted display."""
    doctor_command()


@app.command()
def feishu() -> None:
    """Start Feishu WebSocket bridge: receive messages and reply via Agent."""
    feishu_bridge_command()


@app.command()
def verify(
    audit_only: bool = typer.Option(False, "--audit-only", help="下载 + 仅静态审查"),
    exec_only: bool = typer.Option(False, "--exec-only", help="下载 + 仅执行验证"),
    download_only: bool = typer.Option(False, "--download-only", help="仅下载 skill"),
    sandbox: str = typer.Option("e2b", "--sandbox", help="沙箱类型: e2b / local"),
    concurrency: int = typer.Option(3, "--concurrency", "-c", help="E2B 并发数"),
    timeout: int = typer.Option(120, "--timeout", "-t", help="单个 skill 超时(秒)"),
    output: str = typer.Option(None, "--output", "-o", help="报告 JSON 输出路径"),
    test_set: str = typer.Option("test_set.jsonl", "--test-set", help="测试集路径"),
    cache_dir: str = typer.Option(
        ".verify_cache/skills", "--cache-dir", help="下载缓存目录"
    ),
    limit: int = typer.Option(None, "--limit", "-n", help="只处理前 N 个 (调试用)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细输出"),
) -> None:
    """批量验证 skill: 从 test_set.jsonl 下载 + 安全审查 + 合规审查 + E2B 沙箱执行"""
    verify_command(
        audit_only=audit_only,
        exec_only=exec_only,
        download_only=download_only,
        sandbox=sandbox,
        concurrency=concurrency,
        timeout=timeout,
        output=output,
        test_set=test_set,
        cache_dir=cache_dir,
        limit=limit,
        verbose=verbose,
    )


if __name__ == "__main__":
    app()
