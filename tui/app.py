
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll, Container, Horizontal
from textual.widgets import Footer, TextArea, Static, RichLog, OptionList, LoadingIndicator

from core.agent import MementoSAgent, SessionManager
from core.agent.session_manager import _approx_tokens_from_content as _count_approx_tokens
from core.agent.stateful_context_manager import StatefulContextManager
from core.config import g_settings
from core.llm import LLM
from core.skills import SkillManager
from core.skills.provider.delta_skill_provider import DeltaSkillsProvider
from core.skills.provider.delta_skills.bootstrap import create_app_context

from tui.config import settings
from tui.conversation_store import ConversationStore, Conversation
from tui.interaction_logger import get_interaction_logger
from tui.llm_client import summarize_context, generate_title
from tui.skill_runner import SkillStatusAdapter
from tui.widgets import SelectDialog, ConfirmDialog, ConversationList

logger = logging.getLogger(__name__)



def _platform_bindings() -> list[Binding]:
    return [
        Binding("escape", "check_double_esc", show=False, priority=True),
        Binding("ctrl+q", "quit", "Quit", key_display="Ctrl+Q"),
        Binding("ctrl+h", "toggle_history", key_display="Ctrl+H"),
        Binding("ctrl+b", "toggle_history", key_display="Ctrl+B"),
        Binding("ctrl+t", "toggle_log", "Log", key_display="Ctrl+T"),
        Binding("ctrl+l", "clear_screen", "Clear", key_display="Ctrl+L"),
        Binding("ctrl+c", "copy", "Copy", key_display="Ctrl+C"),
        Binding("ctrl+v", "paste", "Paste", key_display="Ctrl+V"),
        Binding("enter", "submit_message", "Send", priority=True, key_display="Enter"),
        Binding("ctrl+r", "show_context_info", "Context", key_display="Ctrl+R"),
        Binding("ctrl+d", "delete_conversation", "Delete", priority=True, key_display="Ctrl+D"),
        Binding("ctrl+enter", "submit_message", show=False),
        Binding("meta+enter", "submit_message", show=False),
        Binding("command+c", "copy", show=False),
        Binding("command+v", "paste", show=False),
        Binding("ctrl+insert", "copy", show=False),
        Binding("shift+insert", "paste", show=False),
        Binding("ctrl+shift+c", "copy", show=False),
        Binding("ctrl+shift+v", "paste", show=False),
    ]



class MementoSApp(App):

    CSS_PATH = ["styles/main.tcss"]
    BINDINGS = _platform_bindings()

    COMMANDS = {
        "/clear": "Clear chat display",
        "/context": "Show context token usage",
        "/compress": "Force compress context",
        "/reset": "Reset conversation",
        "/skills": "List available skills",
        "/reload": "Reload all skills",
        "/history": "Show conversation history list",
        "/save": "Force save current conversation",
        "/new": "Start a new conversation",
        "/load": "Load a saved conversation",
        "/rename": "Rename current conversation",
        "/delete": "Delete conversation(s), supports batch (space-separated IDs)",
        "/exit": "Quit the application (same as Ctrl+Q, blocked while processing)",
        "/help": "Show available commands and keyboard shortcuts",
    }

    TITLE = settings.app_name

    _WELCOME_IDS = frozenset({"welcome-title", "welcome-info"})

    _DOUBLE_ESC_INTERVAL: float = 0.4

    def __init__(
        self,
        *,
        create_on_miss: bool = False,
        reload_every_turn: bool = True,
        optimize_on_error: bool = True,
        optimize_attempts: int = 1,
    ):
        super().__init__()
        self._interaction = get_interaction_logger()
        self.is_processing = False
        self.create_on_miss = create_on_miss
        self.reload_every_turn = reload_every_turn
        self.optimize_on_error = bool(optimize_on_error)
        self.optimize_attempts = max(0, int(optimize_attempts))

        self.messages: list[dict] = []
        self.total_tokens = 0

        self.compress_threshold = settings.context_compress_threshold
        self.max_tokens = settings.context_max_tokens

        self._last_esc_time: float = 0.0
        self._current_chat_task: Optional[asyncio.Task] = None
        self._background_tasks: set[asyncio.Task] = set()

        self._llm = LLM()
        try:
            _app_context = create_app_context(llm=self._llm, init_logging=False)
            self._skill_provider = DeltaSkillsProvider(app_context=_app_context)
            self._skill_manager = SkillManager(provider=self._skill_provider)
        except Exception as exc:
            logger.warning("Skills initialization failed: %s — running without skills", exc)
            self._skill_provider = None  # type: ignore[assignment]
            self._skill_manager = None  # type: ignore[assignment]

        self._session_manager = SessionManager()
        self._agent = MementoSAgent(
            workspace=g_settings.workspace_path,
            llm=self._llm,
            skill_manager=self._skill_manager,
            session_manager=self._session_manager,
  
        )

        self.conversation_store = ConversationStore(self._session_manager)
        self.skill_status = SkillStatusAdapter(self._skill_manager) if self._skill_manager else None
        self.current_conversation: Optional[Conversation] = None

        self._interaction.event(
            "tui_app_init",
            create_on_miss=self.create_on_miss,
            reload_every_turn=self.reload_every_turn,
            optimize_on_error=self.optimize_on_error,
            optimize_attempts=self.optimize_attempts,
        )


    def compose(self) -> ComposeResult:
        with Horizontal(id="app-header"):
            yield Static(f"[bold #58a6ff] {self.TITLE} [/]", id="header-title")
            yield Static("", id="header-clock")

        with Container(id="history-sidebar"):
            yield ConversationList(
                self.conversation_store,
                id="conversation-list",
            )

        with Container(id="main-layout"):
            with VerticalScroll(id="chat-area"):
                skill_count = self.skill_status.get_skill_count() if self.skill_status else 0
                skill_text = f"{skill_count} loaded" if skill_count else "none"
                cwd = str(Path.cwd())
                yield Static(f"[bold]{self.TITLE}[/]", id="welcome-title")
                welcome_info = (
                    f"[#8b949e]Model[/]    [bold #c9d1d9]{settings.llm_model}[/]\n"
                    f"[#8b949e]Skills[/]   [#c9d1d9]{skill_text}[/]\n"
                    f"[#8b949e]Path[/]     [dim]{cwd}[/]\n\n"
                    f"[#30363d]{'─' * 42}[/]\n\n"
                    f"[#8b949e]Type a message to start, or [bold #c9d1d9]/[/bold #c9d1d9] for commands\n"
                    f"Press [bold #c9d1d9]ESC x2[/bold #c9d1d9] to force-interrupt | "
                    f"[bold #c9d1d9]/help[/bold #c9d1d9] for all shortcuts[/]"
                )
                yield Static(welcome_info, id="welcome-info")

            yield OptionList(id="command-hints")
            yield RichLog(id="log-area", highlight=True, markup=True)
            with Horizontal(id="status-row"):
                yield Static("Tokens: 0 / " + str(self.max_tokens), id="status-bar")
                yield LoadingIndicator(id="loading-indicator")
            yield TextArea(placeholder="Type message or / for commands...", id="user-input")

        yield Footer()


    def on_mount(self) -> None:
        self.query_one("#command-hints").display = False
        self.query_one("#log-area").display = False
        self.query_one("#loading-indicator").display = False
        self._update_clock()
        self.set_interval(1, self._update_clock)
        self.call_after_refresh(self._on_ready)
        self._interaction.event("tui_mount")

    def _update_clock(self) -> None:
        try:
            clock = self.query_one("#header-clock", Static)
            clock.update(f"[#8b949e]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [/]")
        except Exception:
            pass

    def _on_ready(self) -> None:
        try:
            self.query_one("#user-input").focus()
            self.log_info("System ready")
            if self.skill_status and self.skill_status.has_skills():
                names = self.skill_status.get_skill_names()
                self.log_info(f"Loaded {len(names)} skills: {', '.join(names)}")
        except Exception:
            pass

        self._restore_last_session()


    def _clipboard_text_area(self) -> Optional[TextArea]:
        focused = self.focused
        if isinstance(focused, TextArea):
            return focused
        try:
            return self.query_one("#user-input", TextArea)
        except Exception:
            return None

    def _selectable_message(self, text: str, classes: str = "") -> TextArea:
        widget = TextArea()
        widget.text = "" if text is None else str(text)
        try:
            widget.read_only = True
        except Exception:
            pass
        try:
            widget.show_line_numbers = False
        except Exception:
            pass
        try:
            widget.soft_wrap = True
        except Exception:
            pass
        widget.add_class("selectable-msg")
        if classes:
            widget.add_class(*classes.split())
        return widget


    def action_check_double_esc(self) -> None:
        now = time.monotonic()
        if now - self._last_esc_time < self._DOUBLE_ESC_INTERVAL:
            self._last_esc_time = 0.0
            self._force_interrupt()
        else:
            self._last_esc_time = now

    def _force_interrupt(self) -> None:
        was_processing = self.is_processing

        if self._current_chat_task is not None and not self._current_chat_task.done():
            self._current_chat_task.cancel()

        for task in list(self._background_tasks):
            if not task.done():
                task.cancel()

        self.is_processing = False
        self._show_loading(False)

        try:
            input_widget = self.query_one("#user-input", TextArea)
            input_widget.text = ""
            input_widget.focus()
        except Exception:
            pass

        try:
            self.query_one("#command-hints").display = False
        except Exception:
            pass

        if was_processing:
            self.log_info("Force-interrupted by double-ESC")
        self._interaction.event(
            "force_interrupt",
            was_processing=was_processing,
        )


    def _create_background_task(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task


    def _restore_last_session(self) -> None:
        latest_id = self.conversation_store.get_latest_id()
        if latest_id is not None:
            if self._load_conversation(latest_id):
                history = self.query_one("#chat-area", VerticalScroll)
                self._render_loaded_conversation(
                    history,
                    notice=(
                        f"[System] Session restored: {self.current_conversation.title}"
                        f"  ({len(self.messages)} messages)"
                    ),
                )
                self.log_info(
                    f"Restored last session: {self.current_conversation.title} "
                    f"(id: {self.current_conversation.id})"
                )
                self._interaction.event(
                    "session_restored",
                    conversation_id=self.current_conversation.id,
                    title=self.current_conversation.title,
                    message_count=len(self.messages),
                )
            else:
                self.log_info("Failed to restore last session.")
        else:
            self.log_info(
                "No saved conversations found. "
                "Use /new or the sidebar + New Chat to start a new conversation."
            )
            self._interaction.event("no_saved_conversations")


    def _start_new_conversation(self) -> None:
        self.current_conversation = self.conversation_store.create(
            model=settings.llm_model,
        )
        self.messages = []
        self.total_tokens = 0
        self.update_status_bar()
        self._refresh_conversation_list()
        self._interaction.event(
            "conversation_new",
            conversation_id=self.current_conversation.id,
        )

    def _ensure_active_conversation(self) -> None:
        if self.current_conversation is not None:
            return
        self._start_new_conversation()
        self.log_info("Auto-created new conversation for incoming message.")
        self._interaction.event(
            "conversation_auto_created",
            conversation_id=self.current_conversation.id,
        )

    def _auto_save_conversation(self) -> None:
        if self.current_conversation is None:
            return
        self.current_conversation.messages = list(self.messages)
        self.current_conversation.total_tokens = self.total_tokens
        self.current_conversation.model = settings.llm_model

        if self.current_conversation.title == "New Conversation" and self.messages:
            title = ConversationStore.auto_title(self.messages, max_length=20)
            if title != "New Conversation":
                self.current_conversation.title = title

        try:
            self.conversation_store.save(self.current_conversation)
            self._refresh_conversation_list()
        except Exception as e:
            self.log_info(f"Auto-save failed: {e}")
            self._interaction.exception("auto_save_failed", error=str(e))
            return

        if (
            len(self.messages) >= 2
            and not self.current_conversation.metadata.get("title_manual")
        ):
            self._create_background_task(self._update_conversation_title())

    async def _update_conversation_title(self) -> None:
        if self.current_conversation is None:
            return
        conv_id = self.current_conversation.id
        msgs = list(self.messages)

        try:
            loop = asyncio.get_running_loop()
            title = await loop.run_in_executor(None, generate_title, msgs, 20)
            if not title or title == "New Conversation":
                return

            if self.current_conversation and self.current_conversation.id == conv_id:
                self.current_conversation.title = title
                self.conversation_store.save(self.current_conversation)
                self._refresh_conversation_list()
                self._interaction.event(
                    "title_auto_generated",
                    conversation_id=conv_id,
                    title=title,
                )
        except Exception as exc:
            logger.debug("LLM title generation failed for %s: %s", conv_id, exc)

    def _load_conversation(self, conversation_id: str) -> bool:
        conv = self.conversation_store.load(conversation_id)
        if conv is None:
            return False
        self._auto_save_conversation()
        self.current_conversation = conv
        self.messages = list(conv.messages)
        self.total_tokens = conv.total_tokens
        if self.total_tokens == 0 and self.messages:
            self.total_tokens = sum(
                _count_approx_tokens(m.get("content", ""))
                for m in self.messages
            )
            conv.total_tokens = self.total_tokens
        self.update_status_bar()
        self._refresh_conversation_list()
        self._interaction.event(
            "conversation_loaded",
            conversation_id=conversation_id,
            message_count=len(self.messages),
            total_tokens=self.total_tokens,
        )
        return True

    def _render_loaded_conversation(
        self,
        history: VerticalScroll,
        *,
        notice: str = "",
    ) -> None:
        for child in list(history.children):
            if child.id not in self._WELCOME_IDS:
                child.remove()
        if not self.current_conversation:
            return

        if not notice:
            notice = (
                f"[System] Loaded conversation: {self.current_conversation.title} "
                f"(id: {self.current_conversation.id}) "
                f"({len(self.messages)} messages)"
            )
        history.mount(self._selectable_message(notice, classes="system-msg"))

        for msg in self.messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                history.mount(self._selectable_message(f"You: {content}", classes="user-msg"))
            elif role == "assistant":
                history.mount(self._selectable_message("Memento-S:", classes="ai-label"))
                history.mount(self._selectable_message(content, classes="ai-msg"))
        history.scroll_end()

    def _refresh_conversation_list(self) -> None:
        try:
            conv_list = self.query_one("#conversation-list", ConversationList)
            current_id = self.current_conversation.id if self.current_conversation else ""
            conv_list.refresh_list(current_id=current_id)
        except Exception:
            pass

    def _fallback_after_delete(self, history: VerticalScroll) -> None:
        self.action_clear_screen()
        next_id = self.conversation_store.get_latest_id()
        if next_id is not None:
            if self._load_conversation(next_id):
                self._render_loaded_conversation(
                    history,
                    notice=(
                        f"[System] Switched to: {self.current_conversation.title}"
                        f"  ({len(self.messages)} messages)"
                    ),
                )
                self.log_info(f"Switched to conversation: {self.current_conversation.title}")
                return
        self.current_conversation = None
        self.messages = []
        self.total_tokens = 0
        self.update_status_bar()
        history.mount(self._selectable_message(
            "[System] No conversations remaining. Use /new or + New Chat to start.",
            classes="system-msg",
        ))
        history.scroll_end()


    def on_conversation_list_conversation_selected(
        self, event: ConversationList.ConversationSelected,
    ) -> None:
        if self.is_processing:
            return
        if self.current_conversation and event.conversation_id == self.current_conversation.id:
            return
        self._create_background_task(self._switch_to_conversation(event.conversation_id))

    def on_conversation_list_new_conversation_requested(
        self, event: ConversationList.NewConversationRequested,
    ) -> None:
        if self.is_processing:
            return
        self._create_background_task(self._handle_new_conversation())

    async def _switch_to_conversation(self, conversation_id: str) -> None:
        history = self.query_one("#chat-area", VerticalScroll)
        if self._load_conversation(conversation_id):
            self._render_loaded_conversation(history)
            self.log_info(f"Switched to conversation: {self.current_conversation.title}")
        else:
            history.mount(self._selectable_message(
                "[System] Failed to load conversation", classes="system-msg",
            ))
            history.scroll_end()

    async def _handle_new_conversation(self) -> None:
        self._auto_save_conversation()
        self._start_new_conversation()
        self.action_clear_screen()
        history = self.query_one("#chat-area", VerticalScroll)
        history.mount(self._selectable_message(
            "[System] New conversation started", classes="system-msg",
        ))
        history.scroll_end()
        self.log_info("New conversation started")


    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if not self.is_mounted:
            return
        try:
            hints = self.query_one("#command-hints", OptionList)
            input_widget = self.query_one("#user-input", TextArea)
            val = input_widget.text
            if val.startswith("/"):
                matches = [cmd for cmd in self.COMMANDS if cmd.startswith(val)]
                if matches:
                    hints.clear_options()
                    for cmd in matches:
                        hints.add_option(f"{cmd} - {self.COMMANDS[cmd]}")
                    hints.display = True
                else:
                    hints.display = False
            else:
                hints.display = False
        except Exception:
            pass

    async def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        try:
            cmd_text = str(event.option.prompt).split(" - ")[0]
            self._interaction.event("command_hint_selected", option=str(event.option.prompt), cmd=cmd_text)
            input_widget = self.query_one("#user-input", TextArea)
            input_widget.text = cmd_text
            self.query_one("#command-hints").display = False
            input_widget.focus()
        except Exception:
            pass

    async def action_submit_message(self) -> None:
        input_widget = self.query_one("#user-input", TextArea)
        user_text = input_widget.text.strip()
        if not user_text or self.is_processing:
            return
        self._interaction.event(
            "input_submitted",
            text=user_text,
            text_len=len(user_text),
            is_command=user_text.startswith("/"),
        )
        try:
            self.query_one("#command-hints").display = False
        except Exception:
            pass
        input_widget.text = ""

        if user_text.startswith("/"):
            await self.handle_command(user_text)
            return

        self._current_chat_task = asyncio.create_task(
            self.chat_with_skills(user_text)
        )

    def action_copy(self) -> None:
        text_area = self._clipboard_text_area()
        if text_area:
            try:
                text_area.action_copy()
            except Exception:
                pass

    def action_paste(self) -> None:
        text_area = self._clipboard_text_area()
        if text_area:
            try:
                text_area.action_paste()
            except Exception:
                pass


    async def handle_command(self, cmd: str) -> None:
        history = self.query_one("#chat-area", VerticalScroll)
        self._interaction.event("command_invoked", cmd=cmd)
        parts = cmd.split(maxsplit=1)
        base_cmd = parts[0]
        cmd_arg = parts[1].strip() if len(parts) > 1 else ""

        if base_cmd == "/clear":
            self.action_clear_screen()
            history.mount(self._selectable_message("[System] Chat cleared", classes="system-msg"))
        elif base_cmd == "/context":
            await self.show_context_info()
        elif base_cmd == "/compress":
            await self.force_compress_context(history)
        elif base_cmd == "/reset":
            self.messages = []
            self.total_tokens = 0
            if self.current_conversation:
                self.current_conversation.messages = []
                self.current_conversation.total_tokens = 0
                self._auto_save_conversation()
            self.update_status_bar()
            self.action_clear_screen()
            history.mount(self._selectable_message(
                "[System] Conversation reset", classes="system-msg",
            ))
        elif base_cmd == "/skills":
            await self.show_skills_info()
        elif base_cmd == "/reload":
            await self.reload_skills(history)
        elif base_cmd == "/history":
            await self._show_history_list(history)
        elif base_cmd == "/save":
            self._auto_save_conversation()
            title = self.current_conversation.title if self.current_conversation else "?"
            history.mount(self._selectable_message(f"[System] Conversation saved: {title}", classes="system-msg"))
        elif base_cmd == "/new":
            await self._handle_new_conversation()
        elif base_cmd == "/load":
            await self._handle_load_command(history, cmd_arg)
        elif base_cmd == "/rename":
            await self._handle_rename_command(history, cmd_arg)
        elif base_cmd == "/delete":
            await self._handle_delete_command(history, cmd_arg)
        elif base_cmd == "/exit":
            self.action_quit()
            return  # Skip trailing scroll_end — app may already be exiting.
        elif base_cmd == "/help":
            help_text = "\n".join([f"  {k}: {v}" for k, v in self.COMMANDS.items()])
            shortcuts = (
                "\n\n  Keyboard shortcuts:\n"
                "  ESC x2 (double-press): Force-interrupt current session and clear input\n"
                "  Ctrl+H / Ctrl+B: Toggle conversation history sidebar\n"
                "  Ctrl+T: Toggle log panel\n"
                "  Ctrl+L: Clear chat display\n"
                "  Ctrl+R: Show context/token info\n"
                "  Ctrl+D: Delete focused conversation\n"
                "  Ctrl+Q: Quit (same as /exit)"
            )
            history.mount(self._selectable_message(
                f"[System] Available commands:\n{help_text}{shortcuts}",
                classes="system-msg",
            ))
        else:
            history.mount(self._selectable_message(f"[System] Unknown command: {cmd}", classes="system-msg"))
        history.scroll_end()


    async def _show_history_list(self, history: VerticalScroll) -> None:
        summaries = self.conversation_store.list_all()
        if not summaries:
            history.mount(self._selectable_message("[System] No saved conversations.", classes="system-msg"))
            return

        groups = self.conversation_store.group_by_time_period(summaries)
        lines = [f"[System] Saved conversations ({len(summaries)}):"]
        idx = 1
        for group_label, group_summaries in groups:
            lines.append(f"\n  ─── {group_label} ───")
            for s in group_summaries:
                marker = " *" if (self.current_conversation and s.id == self.current_conversation.id) else ""
                lines.append(f"  {idx}. {s.title} ({s.message_count} msgs) id:{s.id}{marker}")
                idx += 1
        lines.append("")
        lines.append("  Use /load <id> to load, /delete <id1> [id2 ...] to delete, Ctrl+H/Ctrl+B to toggle sidebar.")
        history.mount(self._selectable_message("\n".join(lines), classes="system-msg"))
        try:
            sidebar = self.query_one("#history-sidebar")
            if not sidebar.display:
                sidebar.display = True
            self._refresh_conversation_list()
        except Exception:
            pass

    async def _handle_load_command(self, history: VerticalScroll, cmd_arg: str) -> None:
        if not cmd_arg:
            await self._show_history_list(history)
            return
        conversation_id = self._resolve_conversation_ref(cmd_arg)
        if not conversation_id:
            history.mount(self._selectable_message(f"[System] Conversation not found: {cmd_arg}", classes="system-msg"))
            return
        if self._load_conversation(conversation_id):
            self._render_loaded_conversation(history)
            self.log_info(f"Loaded conversation: {self.current_conversation.title}")
        else:
            history.mount(self._selectable_message(f"[System] Failed to load conversation: {cmd_arg}", classes="system-msg"))

    async def _handle_rename_command(self, history: VerticalScroll, cmd_arg: str) -> None:
        if not cmd_arg:
            history.mount(self._selectable_message("[System] Usage: /rename <new title>", classes="system-msg"))
            return
        if not self.current_conversation:
            history.mount(self._selectable_message("[System] No active conversation to rename.", classes="system-msg"))
            return
        old_title = self.current_conversation.title
        self.current_conversation.title = cmd_arg.strip()
        self.current_conversation.metadata["title_manual"] = True
        self._auto_save_conversation()
        history.mount(self._selectable_message(
            f'[System] Renamed: "{old_title}" -> "{self.current_conversation.title}"',
            classes="system-msg",
        ))
        self._interaction.event(
            "conversation_renamed",
            conversation_id=self.current_conversation.id,
            old_title=old_title,
            new_title=self.current_conversation.title,
        )

    _BATCH_DELETE_LIMIT: int = 100

    async def _handle_delete_command(self, history: VerticalScroll, cmd_arg: str) -> None:
        if not cmd_arg:
            history.mount(self._selectable_message(
                "[System] Usage: /delete <id> [id2 id3 ...]\n"
                "  Supports batch deletion with space-separated IDs (max 100).\n"
                "  Example: /delete 8rqjucxx dxj76did\n"
                "  Use /history to see IDs.",
                classes="system-msg",
            ))
            return

        raw_refs = cmd_arg.split()

        if len(raw_refs) > self._BATCH_DELETE_LIMIT:
            history.mount(self._selectable_message(
                f"[System] Batch delete limit is {self._BATCH_DELETE_LIMIT}, "
                f"but {len(raw_refs)} IDs were provided. Please reduce the number of IDs.",
                classes="system-msg",
            ))
            return

        if len(raw_refs) == 1:
            await self._delete_single(history, raw_refs[0])
            return

        await self._delete_batch(history, raw_refs)

    async def _delete_single(self, history: VerticalScroll, ref: str) -> None:
        conversation_id = self._resolve_conversation_ref(ref)
        if not conversation_id:
            history.mount(self._selectable_message(
                f"[System] Conversation not found: {ref}", classes="system-msg",
            ))
            return

        is_current = (
            self.current_conversation is not None
            and conversation_id == self.current_conversation.id
        )

        if self.conversation_store.delete(conversation_id):
            history.mount(self._selectable_message(
                f"[System] Conversation deleted: {conversation_id}", classes="system-msg",
            ))
            self._interaction.event("conversation_deleted", conversation_id=conversation_id)

            if is_current:
                self._fallback_after_delete(history)
            self._refresh_conversation_list()
        else:
            history.mount(self._selectable_message(
                f"[System] Failed to delete: {conversation_id}", classes="system-msg",
            ))

    async def _delete_batch(self, history: VerticalScroll, refs: list[str]) -> None:
        resolved_ids: list[str] = []
        not_found: list[str] = []

        for ref in refs:
            cid = self._resolve_conversation_ref(ref)
            if cid:
                resolved_ids.append(cid)
            else:
                not_found.append(ref)

        seen: set[str] = set()
        unique_ids: list[str] = []
        for cid in resolved_ids:
            if cid not in seen:
                seen.add(cid)
                unique_ids.append(cid)

        if not unique_ids and not_found:
            history.mount(self._selectable_message(
                f"[System] No valid conversations found for: {', '.join(not_found)}",
                classes="system-msg",
            ))
            return

        results = self.conversation_store.batch_delete(unique_ids)

        succeeded = [cid for cid, ok in results.items() if ok]
        failed = [cid for cid, ok in results.items() if not ok]

        lines: list[str] = [f"[System] Batch delete completed ({len(succeeded)}/{len(unique_ids)} succeeded):"]

        if succeeded:
            lines.append(f"  Deleted: {', '.join(succeeded)}")
        if failed:
            lines.append(f"  Failed:  {', '.join(failed)}")
        if not_found:
            lines.append(f"  Not found: {', '.join(not_found)}")

        history.mount(self._selectable_message("\n".join(lines), classes="system-msg"))

        self._interaction.event(
            "conversation_batch_deleted",
            requested=len(refs),
            succeeded=len(succeeded),
            failed=len(failed),
            not_found=len(not_found),
            deleted_ids=succeeded,
        )

        current_deleted = (
            self.current_conversation is not None
            and self.current_conversation.id in succeeded
        )
        if current_deleted:
            self._fallback_after_delete(history)

        self._refresh_conversation_list()

    def _resolve_conversation_ref(self, ref: str) -> Optional[str]:
        ref = ref.strip()
        if self.conversation_store.exists(ref):
            return ref
        try:
            index = int(ref)
            summaries = self.conversation_store.list_all()
            if 1 <= index <= len(summaries):
                return summaries[index - 1].id
        except ValueError:
            pass
        summaries = self.conversation_store.list_all()
        matches = [s for s in summaries if s.id.startswith(ref)]
        if len(matches) == 1:
            return matches[0].id
        return None


    async def show_skills_info(self) -> None:
        history = self.query_one("#chat-area", VerticalScroll)
        self._interaction.event("show_skills_info")
        if self.skill_status and self.skill_status.has_skills():
            names = self.skill_status.get_skill_names()
            skills_list = "\n".join([f"  - {name}" for name in names])
            info = f"[System] Available skills ({len(names)}):\n{skills_list}"
        else:
            info = "[System] No skills available."
        history.mount(self._selectable_message(info, classes="system-msg"))
        history.scroll_end()

    async def reload_skills(self, history: VerticalScroll) -> None:
        old_count = self.skill_status.get_skill_count() if self.skill_status else 0
        old_skills = set(self.skill_status.get_skill_names()) if self.skill_status else set()
        self._interaction.event("reload_skills_start", old_count=old_count, old_skills=sorted(old_skills))

        history.mount(self._selectable_message("[System] Reloading skills...", classes="system-msg"))
        history.scroll_end()

        try:
            _app_context = create_app_context(llm=self._llm)
            self._skill_provider = DeltaSkillsProvider(app_context=_app_context)
            self._skill_manager = SkillManager(provider=self._skill_provider)
            self._agent.skill_manager = self._skill_manager
            self._agent.context_manager = StatefulContextManager(
                workspace=g_settings.workspace_path,
                skill_manager=self._skill_manager,
                session_manager=self._session_manager,
            )
            self.skill_status = SkillStatusAdapter(self._skill_manager)
        except Exception as exc:
            logger.warning("Skills reload failed: %s", exc)
            history.mount(self._selectable_message(f"[System] Skills reload failed: {exc}", classes="error-msg"))
            history.scroll_end()
            return

        new_count = self.skill_status.get_skill_count()
        new_skills = set(self.skill_status.get_skill_names())
        added = new_skills - old_skills
        removed = old_skills - new_skills

        info_lines = [f"[System] Skills reloaded: {old_count} -> {new_count}"]
        if added:
            info_lines.append(f"  + Added: {', '.join(sorted(added))}")
        if removed:
            info_lines.append(f"  - Removed: {', '.join(sorted(removed))}")
        if new_count > 0:
            info_lines.append(f"  Available: {', '.join(sorted(new_skills))}")
        history.mount(self._selectable_message("\n".join(info_lines), classes="system-msg"))
        history.scroll_end()
        self.update_status_bar()
        self.log_info(f"Skills reloaded: {old_count} -> {new_count}")
        self._interaction.event(
            "reload_skills_done",
            old_count=old_count,
            new_count=new_count,
            added=sorted(added),
            removed=sorted(removed),
        )


    def _show_loading(self, show: bool = True) -> None:
        try:
            indicator = self.query_one("#loading-indicator", LoadingIndicator)
            indicator.display = show
        except Exception:
            pass

    async def chat_with_skills(self, user_text: str) -> None:
        self.is_processing = True
        self._show_loading(True)
        history = self.query_one("#chat-area", VerticalScroll)
        user_message_appended = False  # Track whether user msg was added to self.messages

        try:
            self._ensure_active_conversation()

            history.mount(self._selectable_message(f"You: {user_text}", classes="user-msg"))
            history.scroll_end()
            self._interaction.event("chat_user_message_mounted", text=user_text, text_len=len(user_text))

            user_tokens = _count_approx_tokens(user_text)
            if self.total_tokens + user_tokens > self.compress_threshold:
                await self.compress_context(history)

            self.messages.append({"role": "user", "content": user_text})
            self.total_tokens += user_tokens
            user_message_appended = True
            self._interaction.event(
                "chat_user_message_added",
                user_tokens=user_tokens,
                total_tokens=self.total_tokens,
                message_count=len(self.messages),
            )

            session_id = self.current_conversation.id if self.current_conversation else "default"

            history.mount(self._selectable_message("Memento-S:", classes="ai-label"))
            ai_content_widget = self._selectable_message("", classes="ai-msg")
            history.mount(ai_content_widget)
            history.scroll_end()

            accumulated_text = ""
            final_content = ""

            async for event in self._agent.reply_stream(
                session_id=session_id,
                user_content=user_text,
            ):
                event_type = event.get("type")

                if event_type == "text_delta":
                    accumulated_text += event["content"]
                    ai_content_widget.text = accumulated_text
                    history.scroll_end()

                elif event_type == "skill_call_start":
                    skill_name = event.get("skill_name", "")
                    history.mount(self._selectable_message(
                        f"--- Using skill: {skill_name} ---", classes="step-header",
                    ))
                    history.scroll_end()

                elif event_type == "skill_call_result":
                    skill_name = event.get("skill_name", "")
                    result = event.get("result", "")
                    preview = result[:500] + "..." if len(str(result)) > 500 else result
                    history.mount(self._selectable_message(
                        f"[Skill: {skill_name}] {preview}", classes="step-output",
                    ))
                    history.scroll_end()
                    accumulated_text = ""
                    ai_content_widget = self._selectable_message("", classes="ai-msg")
                    history.mount(ai_content_widget)

                elif event_type == "status":
                    self.log_info(event.get("message", ""))

                elif event_type == "final":
                    final_content = event["content"]
                    ai_content_widget.text = final_content
                    history.scroll_end()

                elif event_type == "error":
                    error_msg = event.get("message", "Unknown error")
                    history.mount(self._selectable_message(
                        f"Error: {error_msg}", classes="error-msg",
                    ))
                    history.scroll_end()

            response_text = final_content or accumulated_text
            if response_text:
                self.messages.append({"role": "assistant", "content": response_text})
                response_tokens = _count_approx_tokens(response_text)
                self.total_tokens += response_tokens
                self._interaction.event(
                    "agent_response_done",
                    response_len=len(response_text),
                    response_tokens=response_tokens,
                    total_tokens=self.total_tokens,
                    message_count=len(self.messages),
                )

            self.update_status_bar()
            self._auto_save_conversation()

        except asyncio.CancelledError:
            if user_message_appended and self.messages and self.messages[-1].get("role") == "user":
                removed = self.messages.pop()
                self.total_tokens -= _count_approx_tokens(removed.get("content", ""))

            try:
                history.mount(self._selectable_message(
                    "[System] Session interrupted by user (ESC x2)",
                    classes="system-msg",
                ))
                history.scroll_end()
            except Exception:
                pass

            self.log_info("Chat task force-interrupted by double-ESC")
            self._interaction.event("chat_force_interrupted", user_text=user_text)

        except Exception as e:
            error_msg = f"Error: {e}"
            history.mount(self._selectable_message(error_msg, classes="error-msg"))
            self.log_info(error_msg)
            self._interaction.exception("tui_error", error=str(e))

        finally:
            self.is_processing = False
            self._show_loading(False)
            self._current_chat_task = None


    async def compress_context(self, history: VerticalScroll) -> None:
        if len(self.messages) < 4:
            return
        self.log_info(f"Compressing context ({self.total_tokens} tokens)...")
        history.mount(self._selectable_message("[System] Compressing context...", classes="system-msg"))

        messages_to_summarize = self.messages[:-2]
        recent_messages = self.messages[-2:]

        loop = asyncio.get_running_loop()
        summary = await loop.run_in_executor(
            None,
            summarize_context,
            messages_to_summarize,
            settings.summary_max_tokens,
        )

        self.messages = [
            {"role": "user", "content": f"[Previous conversation summary]\n{summary}"},
            {"role": "assistant", "content": "I understand. Let's continue from where we left off."},
        ] + recent_messages

        new_tokens = sum(_count_approx_tokens(m.get("content", "")) for m in self.messages)
        old_tokens = self.total_tokens
        self.total_tokens = new_tokens
        self.update_status_bar()
        self._auto_save_conversation()
        self.log_info(f"Context compressed: {old_tokens} -> {new_tokens} tokens")
        history.mount(self._selectable_message(f"[System] Context compressed: {old_tokens} -> {new_tokens} tokens", classes="system-msg"))

    async def force_compress_context(self, history: VerticalScroll) -> None:
        if len(self.messages) < 2:
            history.mount(self._selectable_message("[System] Not enough messages to compress", classes="system-msg"))
            return
        await self.compress_context(history)

    async def show_context_info(self) -> None:
        history = self.query_one("#chat-area", VerticalScroll)
        conv_id = self.current_conversation.id if self.current_conversation else "none"
        conv_title = self.current_conversation.title if self.current_conversation else "none"
        skill_count = self.skill_status.get_skill_count() if self.skill_status else 0
        info = f"""[System] Context Info:
  Conversation: {conv_title} (id:{conv_id})
  Messages: {len(self.messages)}
  Tokens: ~{self.total_tokens}
  Threshold: {self.compress_threshold}
  Max: {self.max_tokens}
  Model: {settings.llm_model}
  API: {settings.llm_api}
  Skills: {skill_count}
  Saved conversations: {self.conversation_store.count}"""
        history.mount(self._selectable_message(info, classes="system-msg"))
        history.scroll_end()


    def update_status_bar(self) -> None:
        try:
            status = self.query_one("#status-bar", Static)
            skill_count = self.skill_status.get_skill_count() if self.skill_status else 0
            conv_title = ""
            if self.current_conversation and self.current_conversation.title != "New Conversation":
                title = self.current_conversation.title
                if len(title) > 20:
                    title = title[:17] + "..."
                conv_title = f" | {title}"
            status.update(
                f"Tokens: ~{self.total_tokens} / {self.max_tokens} | "
                f"Messages: {len(self.messages)} | "
                f"Skills: {skill_count}"
                f"{conv_title}"
            )
        except Exception:
            pass


    def action_quit(self) -> None:
        if self.is_processing:
            self.notify(
                "Cannot quit while processing. "
                "Press ESC x2 to interrupt first.",
                title="Quit blocked",
                severity="warning",
            )
            self.log_info("Quit blocked: a task is still processing.")
            self._interaction.event("quit_blocked", reason="is_processing")
            return
        self._auto_save_conversation()
        self._interaction.event("tui_quit")
        self.exit()

    def action_toggle_history(self) -> None:
        try:
            sidebar = self.query_one("#history-sidebar")
            sidebar.display = not sidebar.display
            if sidebar.display:
                self._refresh_conversation_list()
            self._interaction.event("toggle_history", visible=bool(sidebar.display))
        except Exception:
            pass

    def action_delete_conversation(self) -> None:
        if self.is_processing:
            return
        try:
            conv_list = self.query_one("#conversation-list", ConversationList)
            cid = conv_list.get_focused_conversation_id()
            if not cid:
                return

            is_current = self.current_conversation and cid == self.current_conversation.id

            if self.conversation_store.delete(cid):
                if is_current:
                    history = self.query_one("#chat-area", VerticalScroll)
                    self._fallback_after_delete(history)
                self._refresh_conversation_list()
                self.log_info(f"Conversation deleted: {cid}")
                self._interaction.event("conversation_deleted_shortcut", conversation_id=cid)
        except Exception:
            pass

    def action_toggle_log(self) -> None:
        try:
            log = self.query_one("#log-area")
            log.display = not log.display
            self._interaction.event("toggle_log", visible=bool(log.display))
        except Exception:
            pass

    def action_clear_screen(self) -> None:
        try:
            history = self.query_one("#chat-area")
            for child in list(history.children):
                if child.id not in self._WELCOME_IDS:
                    child.remove()
            self._interaction.event("clear_screen")
        except Exception:
            pass

    def action_show_context_info(self) -> None:
        self._interaction.event("show_context_info_action")
        self._create_background_task(self.show_context_info())

    def log_info(self, message: str) -> None:
        try:
            log_area = self.query_one("#log-area", RichLog)
            log_area.write(f"[cyan][INFO][/cyan] {message}")
        except Exception:
            print(f"INFO: {message}")
        try:
            self._interaction.event("tui_info", message=message)
        except Exception:
            pass


    async def prompt_user_select(self, title: str, options: list) -> None:
        def handle_result(result: str) -> None:
            if result:
                try:
                    self.query_one("#chat-area").mount(
                        self._selectable_message(f"Selected: {result}", classes="system-msg")
                    )
                except Exception:
                    pass
        self.push_screen(SelectDialog(title, options), handle_result)

    async def request_approval(self, msg: str) -> None:
        def handle_result(approved: bool) -> None:
            status = "Approved" if approved else "Rejected"
            try:
                self.query_one("#chat-area").mount(
                    self._selectable_message(f"Approval: {status}", classes="system-msg")
                )
            except Exception:
                pass
        self.push_screen(ConfirmDialog(msg), handle_result)
