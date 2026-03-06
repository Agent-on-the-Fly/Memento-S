
import asyncio
import atexit
import functools
import json
import logging
import os
import re
import select
import signal
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from core.agent import MementoSAgent
from core.agent.session_manager import generate_session_id
from core.config import g_settings
from core.config.logging import setup_logging
from cli.config import config_app

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("memento-s")
except Exception:
    __version__ = "0.1.0"

app = typer.Typer(name="MementoS", help="Memento-S Agent CLI", no_args_is_help=True)
app.add_typer(config_app, name="config", help="Manage configuration and .env file.")
console = Console()


def memento_entry() -> None:
    if len(sys.argv) == 1:
        sys.argv.append("agent")
    app()


class _InteractiveInput:

    def __init__(self) -> None:
        self._readline = None
        self._history_file: Path | None = None
        self._saved_termios = None
        self._using_libedit = False
        self._atexit_registered = False

    def setup(self) -> None:
        try:
            import termios
            self._saved_termios = termios.tcgetattr(sys.stdin.fileno())
        except Exception:
            pass
        history_file = Path.home() / ".memento-s" / "history" / "cli_history"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        self._history_file = history_file
        try:
            import readline
        except ImportError:
            return
        self._readline = readline
        self._using_libedit = "libedit" in (readline.__doc__ or "").lower()
        try:
            readline.parse_and_bind("tab: complete" if not self._using_libedit else "bind ^I rl_complete")
            readline.parse_and_bind("set editing-mode emacs")
        except Exception:
            pass
        try:
            readline.read_history_file(str(history_file))
        except Exception:
            pass
        if not self._atexit_registered:
            atexit.register(self.teardown, False)
            self._atexit_registered = True

    def teardown(self, say_goodbye: bool = True) -> None:
        if self._readline is not None and self._history_file is not None:
            try:
                self._readline.write_history_file(str(self._history_file))
            except Exception:
                pass
        if self._saved_termios is not None:
            try:
                import termios
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._saved_termios)
            except Exception:
                pass
        if say_goodbye:
            console.print("\n[dim]Bye![/dim]")

    def flush(self) -> None:
        try:
            fd = sys.stdin.fileno()
            if not os.isatty(fd):
                return
        except Exception:
            return
        try:
            import termios
            termios.tcflush(fd, termios.TCIFLUSH)
        except Exception:
            try:
                while select.select([fd], [], [], 0)[0] and os.read(fd, 4096):
                    pass
            except Exception:
                pass

    def prompt_text(self) -> str:
        prompt = "You › "
        if self._readline is None:
            return prompt
        cyan_bold, reset = "\033[1;36m", "\033[0m"
        if self._using_libedit:
            return f"{cyan_bold}{prompt}{reset}"
        return f"\001{cyan_bold}\002{prompt}\001{reset}\002"


def _print_banner(workspace: Path, session_id: str) -> None:
    banner = Text()
    banner.append("Memento-S", style="bold cyan")
    banner.append(f"  v{__version__}", style="dim")
    console.print(Panel(banner, border_style="cyan", padding=(0, 2)))
    console.print(f"  [dim]Workspace[/dim]  {workspace}")
    console.print(f"  [dim]Session[/dim]    {session_id}")
    console.print(f"  [dim]Model[/dim]      {g_settings.llm_model or 'default'}")
    console.print()


def _print_agent_response(response: str, render_markdown: bool) -> None:
    content = response or ""
    body = Markdown(content) if render_markdown else Text(content)
    console.print()
    console.print(
        Panel(body, title="Memento-S Agent", title_align="left", border_style="cyan", padding=(0, 1))
    )
    console.print()



class _StreamRenderer:

    def __init__(self, render_markdown: bool) -> None:
        self._accumulated = ""
        self._render_markdown = render_markdown
        self._dispatch = {
            "status": self._on_status,
            "text_delta": self._on_text_delta,
            "skill_call_start": self._on_skill_call_start,
            "skill_call_result": self._on_skill_call_result,
            "final": self._on_final,
            "error": self._on_error,
        }

    def handle(self, event: dict) -> None:
        handler = self._dispatch.get(event.get("type"))
        if handler:
            handler(event)

    def flush(self) -> None:
        if self._accumulated.strip():
            clean = re.sub(r"</?thought>", "", self._accumulated).strip()
            if clean:
                console.print(f"  [dim]{clean}[/dim]")
        self._accumulated = ""

    def _on_status(self, event: dict) -> None:
        self.flush()
        console.print(Rule(event["message"], style="cyan"))

    def _on_text_delta(self, event: dict) -> None:
        self._accumulated += event["content"]

    def _on_skill_call_start(self, event: dict) -> None:
        self.flush()
        name = event["skill_name"]
        args = json.dumps(event.get("arguments", {}), ensure_ascii=False)
        console.print(f"  [bold yellow]{name}[/bold yellow]")
        console.print(f"    [dim]IN:[/dim]  {args[:300]}")

    def _on_skill_call_result(self, event: dict) -> None:
        result = str(event.get("result", ""))
        preview = result[:500] + "..." if len(result) > 500 else result
        console.print(f"    [dim]OUT:[/dim] {preview}")

    def _on_final(self, event: dict) -> None:
        self.flush()
        _print_agent_response(event["content"], self._render_markdown)

    def _on_error(self, event: dict) -> None:
        console.print(Panel(event.get("message", "Unknown error"), title="Error", border_style="red"))


async def _run_stream(
    agent_instance: "MementoSAgent",
    session_id: str,
    message: str,
    render_markdown: bool,
) -> None:
    renderer = _StreamRenderer(render_markdown)
    async for event in agent_instance.reply_stream(session_id=session_id, user_content=message):
        renderer.handle(event)
    renderer.flush()


async def _run_interactive(
    agent_instance: "MementoSAgent",
    session_id: str,
    inp: _InteractiveInput,
    render_markdown: bool,
) -> None:
    _EXIT_COMMANDS = frozenset({"/q", ":q", "exit", "quit", "/exit", "/quit"})
    while True:
        try:
            inp.flush()
            user_input = await asyncio.to_thread(input, inp.prompt_text())
            command = user_input.strip()
            if not command:
                continue
            if command.lower() in _EXIT_COMMANDS:
                inp.teardown()
                return
            await _run_stream(agent_instance, session_id, command, render_markdown)
        except (KeyboardInterrupt, EOFError):
            inp.teardown()
            return


def _sigint_handler(inp: _InteractiveInput, _signum: int, _frame) -> None:
    inp.teardown()
    os._exit(0)


@app.command()
def agent(
    message: str = typer.Option(None, "--message", "-m", help="Single message (non-interactive)"),
    session_id: str | None = typer.Option(None, "--session", "-s", help="Session ID"),
    markdown: bool = typer.Option(True, "--markdown/--no-markdown", help="Render output as Markdown"),
    logs: bool = typer.Option(False, "--logs/--no-logs", help="Show verbose logs"),
) -> None:
    session_id = session_id or generate_session_id()
    if logs:
        setup_logging(level=g_settings.log_level, console_output=True)
    else:
        setup_logging(level=g_settings.log_level, log_file="memento_cli.log", console_output=False)

    workspace = g_settings.workspace_path
    agent_instance = MementoSAgent(workspace=workspace)
    _print_banner(workspace, session_id)

    if message:
        asyncio.run(_run_stream(agent_instance, session_id, message, render_markdown=markdown))
        return

    inp = _InteractiveInput()
    inp.setup()
    console.print("[dim]Interactive mode. Type [bold]exit[/bold] or [bold]Ctrl+C[/bold] to quit.[/dim]\n")

    signal.signal(signal.SIGINT, functools.partial(_sigint_handler, inp))
    asyncio.run(_run_interactive(agent_instance, session_id, inp, render_markdown=markdown))


def _secret_display(key_lower: str, value: object) -> str:
    if "max_tokens" in key_lower:
        return str(value)
    if any(k in key_lower for k in ("key", "token", "password", "secret")) and value:
        s = str(value)
        return f"{s[:4]}...{s[-4:]} (len={len(s)})" if len(s) > 10 else "***"
    if value is None:
        return "[dim]None[/dim]"
    return str(value)


@app.command()
def doctor() -> None:
    from dotenv import find_dotenv

    console.print(Panel(Text("Memento-S Doctor", style="bold cyan"), border_style="cyan", padding=(0, 2)))
    console.print()

    ok, no = "[green]✓[/green]", "[red]✗[/red]"
    project_root = g_settings.project_root
    workspace = g_settings.workspace_path
    conversations_dir = g_settings.conversations_path
    console.print("[bold]Paths[/bold]")
    console.print(f"  Project root:   {project_root} {ok if project_root.exists() else no}")
    console.print(f"  Workspace:     {workspace} {ok if workspace.exists() else no}")
    console.print(f"  Conversations: {conversations_dir} {ok if conversations_dir.exists() else no}")
    env_path = find_dotenv()
    console.print(f"  .env:          {Path(env_path) if env_path else '[dim]not found[/dim]'} {ok if env_path else '[yellow]![/yellow]'}")
    console.print()

    table = Table(title="Settings", show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green", overflow="fold")
    table.add_column("Env", style="dim")
    for key in sorted(g_settings.model_dump().keys()):
        value = g_settings.model_dump()[key]
        field_info = g_settings.model_fields.get(key)
        alias = (field_info.alias or "") if field_info else ""
        table.add_row(key, _secret_display(key.lower(), value), alias)
    for prop in ("workspace_path", "conversations_path", "data_directory", "skills_directory",
                 "chroma_directory", "qwen3_tokenizer_path_resolved", "qwen3_model_path_resolved"):
        try:
            val = getattr(g_settings, prop)
            table.add_row(prop, str(val) if val is not None else "[dim]None[/dim]", "[property]")
        except Exception as e:
            table.add_row(prop, f"[red]{e}[/red]", "[property]")
    console.print(table)


@app.command()
def verify(
    audit_only: bool = typer.Option(False, "--audit-only", help=" + "),
    exec_only: bool = typer.Option(False, "--exec-only", help=" + "),
    download_only: bool = typer.Option(False, "--download-only", help=" skill"),
    sandbox: str = typer.Option("e2b", "--sandbox", help=": e2b / local"),
    concurrency: int = typer.Option(3, "--concurrency", "-c", help="E2B "),
    timeout: int = typer.Option(120, "--timeout", "-t", help=" skill ()"),
    output: str = typer.Option(None, "--output", "-o", help=" JSON "),
    test_set: str = typer.Option("test_set.jsonl", "--test-set", help=""),
    cache_dir: str = typer.Option(".verify_cache/skills", "--cache-dir", help=""),
    limit: int = typer.Option(None, "--limit", "-n", help=" N  ()"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help=""),
) -> None:
    import subprocess
    cmd = [sys.executable, str(_PROJECT_ROOT / "scripts" / "verify_pipeline.py")]

    if audit_only:
        cmd.append("--audit-only")
    elif exec_only:
        cmd.append("--exec-only")
    elif download_only:
        cmd.append("--download-only")
    else:
        cmd.append("--all")

    cmd.extend(["--sandbox", sandbox])
    cmd.extend(["--concurrency", str(concurrency)])
    cmd.extend(["--timeout", str(timeout)])
    cmd.extend(["--test-set", test_set])
    cmd.extend(["--cache-dir", cache_dir])

    if output:
        cmd.extend(["--output", output])
    if limit:
        cmd.extend(["--limit", str(limit)])
    if verbose:
        cmd.append("--verbose")

    console.print(f"[dim]Running: {' '.join(cmd)}[/dim]\n")
    result = subprocess.run(cmd)
    raise typer.Exit(result.returncode)


if __name__ == "__main__":
    app()
