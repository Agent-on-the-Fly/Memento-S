
import asyncio
import json
import logging
import os
import re
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

# ── Slash commands ──────────────────────────────────────────────────
SLASH_COMMANDS = [
    ("/help",    "Show available commands"),
    ("/status",  "Show current session status"),
    ("/config",  "View/update configuration"),
    ("/history", "List saved sessions or load one"),
    ("/clear",   "Clear context & start new session"),
    ("/exit",    "Exit the CLI"),
]


def _print_help() -> None:
    table = Table(title="Slash Commands", show_header=True, header_style="bold magenta")
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description", style="dim")
    for cmd, desc in SLASH_COMMANDS:
        table.add_row(cmd, desc)
    console.print(table)


def _print_status(session_id: str, agent_instance: "MementoSAgent", turn_count: int) -> None:
    model = g_settings.llm_model or "default"
    provider = g_settings.llm_api or "openai"
    skills = []
    try:
        skills = list(agent_instance.skill_manager.skills.keys())
    except Exception:
        pass

    console.print(f"  [bold]Model[/bold]      {model} [dim]({provider})[/dim]")
    console.print(f"  [bold]Session[/bold]    {session_id}")
    console.print(f"  [bold]Turns[/bold]      {turn_count}")
    console.print(f"  [bold]Workspace[/bold]  {g_settings.workspace_path}")
    if skills:
        console.print(f"  [bold]Skills[/bold]     {', '.join(skills[:15])}" + (" ..." if len(skills) > 15 else ""))
    console.print()


def _handle_clear() -> str:
    new_id = generate_session_id()
    console.print(f"  [green]Session cleared.[/green] New session: [bold]{new_id}[/bold]\n")
    return new_id


def _handle_history(args: str, agent_instance: "MementoSAgent") -> str | None:
    """Handle /history commands. Returns new session_id if loading, else None."""
    parts = args.strip().split()

    # /history load <id>
    if len(parts) >= 2 and parts[0] == "load":
        target = parts[1]
        existing = agent_instance.session_manager.list_sessions()
        if target not in existing:
            console.print(f"  [red]Session '{target}' not found.[/red]")
            return None
        console.print(f"  [green]Switched to session:[/green] [bold]{target}[/bold]\n")
        return target

    # /history [N]
    sessions = agent_instance.session_manager.list_sessions()
    if not sessions:
        console.print("  [dim]No saved sessions.[/dim]\n")
        return None

    limit = 20
    if parts and parts[0].isdigit():
        limit = int(parts[0])

    table = Table(title="Sessions", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", no_wrap=True)
    table.add_column("Session ID", style="cyan")
    table.add_column("Title", style="green")

    for idx, sid in enumerate(sessions[:limit], 1):
        title = ""
        try:
            data = agent_instance.session_manager.get_session(sid)
            title = (data or {}).get("title", "")
        except Exception:
            pass
        table.add_row(str(idx), sid, title or "[dim]-[/dim]")

    console.print(table)
    if len(sessions) > limit:
        console.print(f"  [dim]Showing {limit}/{len(sessions)}. Use /history {limit + 20} to see more.[/dim]")
    console.print(f"  [dim]Load a session: /history load <session_id>[/dim]\n")
    return None


def _handle_config_inline(args: str) -> None:
    """Handle /config sub-commands inline."""
    from cli.config import list_config, get_config, set_config, unset_config

    parts = args.strip().split(maxsplit=2)
    sub = parts[0] if parts else ""

    if sub in ("", "show", "list"):
        list_config()
    elif sub == "get" and len(parts) >= 2:
        get_config(parts[1])
    elif sub == "set" and len(parts) >= 3:
        set_config(parts[1], parts[2])
    elif sub == "unset" and len(parts) >= 2:
        unset_config(parts[1])
    else:
        console.print("  [dim]Usage: /config [show|get <key>|set <key> <val>|unset <key>][/dim]")


def _suggest_command(cmd: str) -> None:
    """Suggest a similar slash command for unknown input."""
    import difflib
    known = [c for c, _ in SLASH_COMMANDS]
    matches = difflib.get_close_matches(cmd, known, n=1, cutoff=0.4)
    console.print(f"  [yellow]Unknown command: {cmd}[/yellow]")
    if matches:
        console.print(f"  [dim]Did you mean [bold]{matches[0]}[/bold]?[/dim]")
    else:
        console.print("  [dim]Type /help for available commands.[/dim]")
    console.print()


def memento_entry() -> None:
    if len(sys.argv) == 1:
        sys.argv.append("agent")
    app()


from prompt_toolkit.completion import Completer, Completion


class _SlashCompleter(Completer):
    """prompt_toolkit Completer that shows slash commands in a floating menu."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        for cmd, desc in SLASH_COMMANDS:
            if cmd.startswith(text):
                yield Completion(cmd, start_position=-len(text), display_meta=desc)


class _InteractiveInput:

    def __init__(self) -> None:
        self._session = None

    def setup(self) -> None:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.styles import Style

        history_file = Path.home() / ".memento-s" / "history" / "cli_history"
        history_file.parent.mkdir(parents=True, exist_ok=True)

        style = Style.from_dict({
            "prompt": "bold cyan",
            "": "",
            # Transparent completion menu (no gray background)
            "completion-menu":                "noinherit",
            "completion-menu.completion":     "noinherit",
            "completion-menu.completion.current": "bold cyan",
            "completion-menu.meta.completion":          "noinherit #888888",
            "completion-menu.meta.completion.current":  "noinherit #888888",
        })
        self._session = PromptSession(
            message=[("class:prompt", "You › ")],
            history=FileHistory(str(history_file)),
            completer=_SlashCompleter(),
            complete_while_typing=True,
            style=style,
        )

    def teardown(self, say_goodbye: bool = True) -> None:
        if say_goodbye:
            console.print("\n[dim]Bye![/dim]")

    async def prompt_async(self) -> str:
        return await self._session.prompt_async()


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

    def __init__(self, render_markdown: bool, quiet: bool = False) -> None:
        self._accumulated = ""
        self._render_markdown = render_markdown
        self._quiet = quiet
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
        if self._quiet:
            self._accumulated = ""
            return
        if self._accumulated.strip():
            clean = re.sub(r"</?thought>", "", self._accumulated).strip()
            if clean:
                console.print(f"  [dim]{clean}[/dim]")
        self._accumulated = ""

    def _on_status(self, event: dict) -> None:
        self.flush()
        if not self._quiet:
            console.print(Rule(event["message"], style="cyan"))

    def _on_text_delta(self, event: dict) -> None:
        self._accumulated += event["content"]

    def _on_skill_call_start(self, event: dict) -> None:
        self.flush()
        if self._quiet:
            return
        name = event["skill_name"]
        args = json.dumps(event.get("arguments", {}), ensure_ascii=False)
        console.print(f"  [bold yellow]{name}[/bold yellow]")
        console.print(f"    [dim]IN:[/dim]  {args[:300]}")

    def _on_skill_call_result(self, event: dict) -> None:
        if self._quiet:
            return
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
    quiet: bool = False,
) -> None:
    renderer = _StreamRenderer(render_markdown, quiet=quiet)
    async for event in agent_instance.reply_stream(session_id=session_id, user_content=message):
        renderer.handle(event)
    renderer.flush()


async def _run_interactive(
    agent_instance: "MementoSAgent",
    session_id: str,
    inp: _InteractiveInput,
    render_markdown: bool,
    quiet: bool = False,
) -> None:
    _EXIT_COMMANDS = frozenset({"/q", ":q", "exit", "quit", "/exit", "/quit"})
    state = {"session_id": session_id, "turns": 0}

    while True:
        try:
            user_input = await inp.prompt_async()
            command = user_input.strip()
            if not command:
                continue

            low = command.lower()

            # ── exit ────────────────────────────────────────────
            if low in _EXIT_COMMANDS:
                inp.teardown()
                return

            # ── slash commands ──────────────────────────────────
            if low in ("/", "/help", "help"):
                _print_help()
                continue

            if low in ("/status", "status"):
                _print_status(state["session_id"], agent_instance, state["turns"])
                continue

            if low in ("/clear", "clear"):
                state["session_id"] = _handle_clear()
                state["turns"] = 0
                continue

            if low.startswith("/history"):
                args = command[len("/history"):].strip()
                new_sid = _handle_history(args, agent_instance)
                if new_sid is not None:
                    state["session_id"] = new_sid
                    state["turns"] = 0
                continue

            if low.startswith("/config"):
                args = command[len("/config"):].strip()
                _handle_config_inline(args)
                continue

            # ── unknown slash command ───────────────────────────
            if command.startswith("/"):
                _suggest_command(command.split()[0])
                continue

            # ── normal message → agent ──────────────────────────
            await _run_stream(agent_instance, state["session_id"], command, render_markdown, quiet=quiet)
            state["turns"] += 1
        except (KeyboardInterrupt, EOFError):
            inp.teardown()
            return


@app.command()
def agent(
    message: str = typer.Option(None, "--message", "-m", help="Single message (non-interactive)"),
    session_id: str | None = typer.Option(None, "--session", "-s", help="Session ID"),
    markdown: bool = typer.Option(True, "--markdown/--no-markdown", help="Render output as Markdown"),
    quiet: bool = typer.Option(False, "--quiet/--no-quiet", "-q", help="Only show final agent response"),
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
        asyncio.run(_run_stream(agent_instance, session_id, message, render_markdown=markdown, quiet=quiet))
        return

    inp = _InteractiveInput()
    inp.setup()
    console.print("[dim]Interactive mode. Type [bold]/help[/bold] for commands, [bold]/exit[/bold] or [bold]Ctrl+C[/bold] to quit.[/dim]\n")

    asyncio.run(_run_interactive(agent_instance, session_id, inp, render_markdown=markdown, quiet=quiet))


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
