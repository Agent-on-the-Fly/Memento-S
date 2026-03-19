"""Agent command for interactive chat with Memento-S."""

import asyncio
import atexit
import functools
import json
import os
import re
import select
import signal
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

# Handle project root for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.memento_s.agent import MementoSAgent
from core.memento_s.stream_output import (
    AGUIEventPipeline,
    AGUIEventType,
    PersistenceSink,
)
from core.manager import ConversationManager, SessionManager, generate_session_id
from middleware.config import g_config
from utils.token_utils import estimate_tokens
from utils.debug_logger import log_agent_start, log_agent_end, log_agent_phase

console = Console()


class _InteractiveInput:
    """
    交互式输入：readline 历史 + 终端恢复。
    - setup: 保存当前 termios，启用 readline（历史文件、补全、emacs），注册退出时保存历史并恢复终端。
    - teardown: 写回历史、恢复 termios，可选打印再见。
    - flush: 清空 stdin 缓冲，避免「思考」期间误按键被下一次 input() 读到。
    """

    def __init__(self) -> None:
        self._readline = None
        self._saved_termios = None
        self._using_libedit = False
        self._atexit_registered = False

    def setup(self) -> None:
        try:
            import termios

            self._saved_termios = termios.tcgetattr(sys.stdin.fileno())
        except Exception:
            pass
        try:
            import readline
        except ImportError:
            return
        self._readline = readline
        self._using_libedit = "libedit" in (readline.__doc__ or "").lower()
        try:
            readline.parse_and_bind(
                "tab: complete" if not self._using_libedit else "bind ^I rl_complete"
            )
            readline.parse_and_bind("set editing-mode emacs")
        except Exception:
            pass
        if not self._atexit_registered:
            atexit.register(self.teardown, False)
            self._atexit_registered = True

    def teardown(self, say_goodbye: bool = True) -> None:
        if self._saved_termios is not None:
            try:
                import termios

                termios.tcsetattr(
                    sys.stdin.fileno(), termios.TCSADRAIN, self._saved_termios
                )
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
        """Return the input prompt string. Uses cyan styling when readline is available."""
        prompt = "You › "
        if self._readline is None:
            return prompt
        # Bold cyan to match CLI theme; libedit ignores \001\002 so we use raw escapes there
        cyan_bold, reset = "\033[1;36m", "\033[0m"
        if self._using_libedit:
            return f"{cyan_bold}{prompt}{reset}"
        return f"\001{cyan_bold}\002{prompt}\001{reset}\002"


def _print_banner(workspace: Path, session_id: str, version: str) -> None:
    """Print startup banner and basic info."""
    try:
        from importlib.metadata import version as _pkg_version

        __version__ = _pkg_version("memento-s")
    except Exception:
        __version__ = version

    banner = Text()
    banner.append("Memento-S", style="bold cyan")
    banner.append(f"  v{__version__}", style="dim")
    console.print(Panel(banner, border_style="cyan", padding=(0, 2)))
    console.print(f"  [dim]Workspace[/dim]  {workspace}")
    console.print(f"  [dim]Session[/dim]    {session_id}")
    model = g_config.llm.current.model
    console.print(f"  [dim]Model[/dim]      {model}")
    console.print()


def _print_agent_response(response: str, render_markdown: bool) -> None:
    content = response or ""
    body = Markdown(content) if render_markdown else Text(content)
    console.print()
    console.print(
        Panel(
            body,
            title="Memento-S Agent",
            title_align="left",
            border_style="cyan",
            padding=(0, 1),
        )
    )
    console.print()


class _StreamRenderer:
    """Dispatch-based renderer for AG-UI reply_stream events."""

    def __init__(self, render_markdown: bool) -> None:
        self._accumulated = ""
        self._render_markdown = render_markdown
        self._dispatch = {
            AGUIEventType.RUN_STARTED: self._on_run_started,
            AGUIEventType.STEP_STARTED: self._on_step_started,
            AGUIEventType.STEP_FINISHED: self._on_step_finished,
            AGUIEventType.TEXT_MESSAGE_START: self._on_text_message_start,
            AGUIEventType.TEXT_MESSAGE_CONTENT: self._on_text_message_content,
            AGUIEventType.TEXT_MESSAGE_END: self._on_text_message_end,
            AGUIEventType.TOOL_CALL_START: self._on_tool_call_start,
            AGUIEventType.TOOL_CALL_RESULT: self._on_tool_call_result,
            AGUIEventType.RUN_FINISHED: self._on_run_finished,
            AGUIEventType.RUN_ERROR: self._on_run_error,
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

    def _on_run_started(self, event: dict) -> None:
        console.print(Rule("Agent run started", style="cyan"))

    def _on_step_started(self, event: dict) -> None:
        step = event.get("step")
        console.print(Rule(f"Thinking (step {step})", style="cyan"))

    def _on_step_finished(self, event: dict) -> None:
        return

    def _on_text_message_start(self, event: dict) -> None:
        self._accumulated = ""

    def _on_text_message_content(self, event: dict) -> None:
        self._accumulated += event.get("delta", "")

    def _on_text_message_end(self, event: dict) -> None:
        return

    def _on_tool_call_start(self, event: dict) -> None:
        self.flush()
        name = event.get("toolName", "unknown_tool")
        args = json.dumps(event.get("arguments", {}), ensure_ascii=False)
        console.print(f"  [bold yellow]{name}[/bold yellow]")
        console.print(f"    [dim]IN:[/dim]  {args[:300]}")

    def _on_tool_call_result(self, event: dict) -> None:
        result = str(event.get("result", ""))
        preview = result[:500] + "..." if len(result) > 500 else result
        console.print(f"    [dim]OUT:[/dim] {preview}")

    def _on_run_finished(self, event: dict) -> None:
        content = event.get("outputText", "") or ""
        if not content and self._accumulated:
            content = self._accumulated
        self._accumulated = ""
        _print_agent_response(content, self._render_markdown)

    def _on_run_error(self, event: dict) -> None:
        console.print(
            Panel(
                event.get("message", "Unknown error"), title="Error", border_style="red"
            )
        )


async def _run_stream(
    agent_instance: MementoSAgent,
    session_id: str,
    message: str,
    render_markdown: bool,
    conversation_manager: ConversationManager,
) -> None:
    """Consume reply_stream events, render, and persist in unified pipeline."""
    renderer = _StreamRenderer(render_markdown)

    user_title = message[:50] + "..." if len(message) > 50 else message
    user_conv = await conversation_manager.create_conversation(
        session_id=session_id,
        role="user",
        title=user_title,
        content=message,
        meta_info={},
    )

    async def _persist_assistant_output(content: str):
        assistant_title = content[:50] + "..." if len(content) > 50 else content
        await conversation_manager.create_conversation(
            session_id=session_id,
            role="assistant",
            title=assistant_title,
            content=content,
            meta_info={"reply_to": user_conv.id},
            tokens=estimate_tokens(content),
        )

    pipeline = AGUIEventPipeline()
    pipeline.add_sink(PersistenceSink(callback=_persist_assistant_output))

    # DEBUG: 记录进入 REPLY 阶段
    log_agent_phase("REPLY_STREAM", session_id, f"Processing: {message[:100]}...")

    async for event in agent_instance.reply_stream(
        session_id=session_id, user_content=message
    ):
        await pipeline.emit(event)
        renderer.handle(event)
    renderer.flush()


async def _run_interactive(
    agent_instance: MementoSAgent,
    session_id: str,
    inp: _InteractiveInput,
    render_markdown: bool,
    conversation_manager: ConversationManager,
) -> None:
    """Interactive REPL loop: read input, stream response, repeat."""
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
            await _run_stream(
                agent_instance,
                session_id,
                command,
                render_markdown,
                conversation_manager,
            )
        except (KeyboardInterrupt, EOFError):
            inp.teardown()
            return


def _sigint_handler(inp: _InteractiveInput, _signum: int, _frame) -> None:
    inp.teardown()
    os._exit(0)


def agent_command(
    message: str = typer.Option(
        None, "--message", "-m", help="Single message (non-interactive)"
    ),
    session_id: str | None = typer.Option(None, "--session", "-s", help="Session ID"),
    markdown: bool = typer.Option(
        True, "--markdown/--no-markdown", help="Render output as Markdown"
    ),
    version: str = "0.1.0",
) -> None:
    """Chat with the Memento-S agent."""
    import time

    start_time = time.monotonic()

    cfg = g_config

    workspace = cfg.paths.workspace_dir
    if workspace is None:
        raise ValueError("paths.workspace_dir 未设置")

    # DEBUG: 记录 Agent 启动
    active_session_id = session_id or generate_session_id()
    model_name = (
        getattr(cfg.llm.active_profile, "model", "unknown")
        if hasattr(cfg, "llm")
        else "unknown"
    )
    log_agent_start(active_session_id, model_name, message or "[interactive mode]")

    # 创建 Skill Gateway
    # bootstrap 阶段已完成 skill 系统同步，此处直接创建 Gateway
    from core.skill.gateway import create_gateway

    try:
        skill_gateway = asyncio.run(create_gateway())
        console.print(
            f"[dim]Skill system ready ({len(skill_gateway.discover())} skills)[/dim]\n"
        )
        agent_instance = MementoSAgent(
            skill_gateway=skill_gateway,
            session_manager=SessionManager(),
        )
    except Exception as e:
        # 创建 Gateway 失败（异常情况）
        console.print(f"[dim]Initializing skill system... ({e})[/dim]")
        agent_instance = MementoSAgent()
    session_manager = SessionManager()
    conversation_manager = ConversationManager()

    if session_id:
        existing = asyncio.run(session_manager.get_session(session_id))
        if existing is None:
            raise ValueError(f"Session not found: {session_id}")
    else:
        created = asyncio.run(
            session_manager.create_session(title="CLI Session", metadata={})
        )
        session_id = created["id"]

    _print_banner(workspace, session_id, version)

    if message:
        asyncio.run(
            _run_stream(
                agent_instance,
                session_id,
                message,
                render_markdown=markdown,
                conversation_manager=conversation_manager,
            )
        )
        # DEBUG: 记录 Agent 结束（仅非交互模式）
        duration = time.monotonic() - start_time
        log_agent_end(session_id, duration, success=True)
        return

    inp = _InteractiveInput()
    inp.setup()
    console.print(
        "[dim]Interactive mode. Type [bold]exit[/bold] or [bold]Ctrl+C[/bold] to quit.[/dim]\n"
    )

    signal.signal(signal.SIGINT, functools.partial(_sigint_handler, inp))
    asyncio.run(
        _run_interactive(
            agent_instance,
            session_id,
            inp,
            render_markdown=markdown,
            conversation_manager=conversation_manager,
        )
    )
