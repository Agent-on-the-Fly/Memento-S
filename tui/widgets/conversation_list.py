
from __future__ import annotations

import time
from typing import Optional

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.events import Click
from textual.message import Message
from textual.widgets import Static, Button, TextArea

from tui.conversation_store import ConversationStore, ConversationSummary
from tui.interaction_logger import get_interaction_logger

_DOUBLE_CLICK_INTERVAL: float = 0.4



class _ConvEntry(TextArea):

    class EntryActivated(Message):

        def __init__(self, conversation_id: str) -> None:
            super().__init__()
            self.conversation_id = conversation_id

    def __init__(
        self,
        conversation_id: str,
        text: str,
        is_current: bool = False,
    ) -> None:
        super().__init__()
        self.conversation_id = conversation_id
        self._initial_text = text
        self._is_current = is_current
        self._last_click_time: float = 0.0

    def on_mount(self) -> None:
        self.text = self._initial_text
        self.read_only = True
        self.show_line_numbers = False
        self.soft_wrap = True
        self.add_class("conv-entry")
        if self._is_current:
            self.add_class("conv-entry-active")

    def on_click(self, event: Click) -> None:
        now = time.monotonic()
        if now - self._last_click_time < _DOUBLE_CLICK_INTERVAL:
            self.post_message(self.EntryActivated(conversation_id=self.conversation_id))
            self._last_click_time = 0.0
        else:
            self._last_click_time = now

    def on_focus(self) -> None:
        self._update_hint(visible=True)

    def on_blur(self) -> None:
        self._update_hint(visible=False)

    def _update_hint(self, *, visible: bool) -> None:
        try:
            hint = self.screen.query_one("#conv-hint-bar", Static)
            hint.update(
                "[dim]Double-click: Load | Ctrl+D: Delete[/]" if visible else ""
            )
        except Exception:
            pass


class _GroupHeader(Static):

    def __init__(self, label: str) -> None:
        super().__init__(f"─── {label} ───")
        self.add_class("conv-group-header")



class ConversationList(Static):


    class ConversationSelected(Message):

        def __init__(self, conversation_id: str) -> None:
            super().__init__()
            self.conversation_id = conversation_id

    class NewConversationRequested(Message):
        pass

    class ConversationDeleteRequested(Message):

        def __init__(self, conversation_id: str) -> None:
            super().__init__()
            self.conversation_id = conversation_id

    DEFAULT_CSS = """
    ConversationList {
        width: 100%;
        height: 100%;
    }
    """

    def __init__(
        self,
        store: ConversationStore,
        *,
        current_id: str = "",
        id: Optional[str] = None,
    ) -> None:
        super().__init__(id=id)
        self._store = store
        self._current_id = current_id
        self._summaries: list[ConversationSummary] = []


    def compose(self) -> ComposeResult:
        yield Button("+ New Chat", id="new-chat-btn", variant="default")
        yield VerticalScroll(id="conv-scroll-area")
        yield Static("", id="conv-hint-bar")

    def on_mount(self) -> None:
        self.refresh_list()


    def refresh_list(self, current_id: str = "") -> None:
        if current_id:
            self._current_id = current_id
        self._summaries = self._store.list_all()

        try:
            scroll = self.query_one("#conv-scroll-area", VerticalScroll)

            for child in list(scroll.children):
                child.remove()

            if not self._summaries:
                placeholder = TextArea()
                placeholder.text = "(No saved conversations)"
                placeholder.read_only = True
                placeholder.show_line_numbers = False
                placeholder.add_class("conv-entry", "conv-empty")
                scroll.mount(placeholder)
                return

            groups = self._store.group_by_time_period(self._summaries)
            for group_label, group_summaries in groups:
                scroll.mount(_GroupHeader(group_label))
                for summary in group_summaries:
                    is_current = summary.id == self._current_id
                    label = self._format_entry(summary, is_current=is_current)
                    entry = _ConvEntry(
                        conversation_id=summary.id,
                        text=label,
                        is_current=is_current,
                    )
                    scroll.mount(entry)

        except Exception as exc:
            get_interaction_logger().exception("conv_list_refresh_error", error=str(exc))


    @staticmethod
    def _format_entry(
        summary: ConversationSummary,
        *,
        is_current: bool = False,
    ) -> str:
        title = summary.title
        max_len = 26 if is_current else 28
        if len(title) > max_len:
            title = title[: max_len - 1] + "…"
        if is_current:
            title = f"● {title}"
        meta = f"{summary.id} | {summary.message_count} msgs"
        return f"{title}\n{meta}"


    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "new-chat-btn":
            get_interaction_logger().event("conv_list_new_chat")
            self.post_message(self.NewConversationRequested())

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        widget = event.text_area
        if isinstance(widget, _ConvEntry) and widget.text != widget._initial_text:
            widget.text = widget._initial_text

    def on__conv_entry_entry_activated(self, event: _ConvEntry.EntryActivated) -> None:
        cid = event.conversation_id
        if cid:
            get_interaction_logger().event("conv_list_selected", conversation_id=cid)
            self.post_message(self.ConversationSelected(conversation_id=cid))


    def get_focused_conversation_id(self) -> Optional[str]:
        try:
            focused = self.app.focused
            if isinstance(focused, _ConvEntry) and focused.conversation_id:
                return focused.conversation_id
        except Exception:
            pass
        return None

    def set_current(self, conversation_id: str) -> None:
        self._current_id = conversation_id
        self.refresh_list()
