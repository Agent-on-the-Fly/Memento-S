"""CLI for the Memento-S agent."""

# Suppress litellm logging before any imports
import os

os.environ["LITELLM_LOG"] = "WARNING"

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

from cli.commands import (
    agent_command,
    doctor_command,
    feishu_bridge_command,
    dingtalk_bridge_command,
    wecom_bridge_command,
    im_status_command,
    gateway_worker_command,
    wechat_app,
)

# 版本号管理：开发模式从 version.py 读取，打包模式从包元数据读取
try:
    # 开发模式：优先从 version.py 读取
    from version import __version__
except ImportError:
    try:
        # 打包模式：从包元数据读取
        from importlib.metadata import version as _pkg_version

        __version__ = _pkg_version("memento-s")
    except Exception as e:
        print(f"[Warning] Failed to get version, defaulting to 0.2.0: {e}")
        __version__ = "0.2.0"

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
def dingtalk() -> None:
    """Start DingTalk Stream bridge: receive messages and reply via Agent."""
    dingtalk_bridge_command()


@app.command()
def wecom() -> None:
    """Start WeCom (企业微信) WebSocket bridge: receive messages and reply via Agent."""
    wecom_bridge_command()


# Add wechat subcommand app
app.add_typer(wechat_app, name="wechat", help="WeChat management commands")


@app.command()
def im_status() -> None:
    """Check IM platform (Gateway/Bridge) status."""
    im_status_command()


@app.command("gateway-worker")
def gateway_worker(
    gateway_url: str = typer.Option(
        "ws://127.0.0.1:8765", "--url", "-u", help="Gateway WebSocket URL"
    ),
    agent_id: str = typer.Option(
        "agent_main", "--agent-id", "-a", help="Agent ID for registration"
    ),
) -> None:
    """Start Gateway Agent Worker: connect to Gateway and process messages."""
    gateway_worker_command(gateway_url=gateway_url, agent_id=agent_id)


if __name__ == "__main__":
    app()
