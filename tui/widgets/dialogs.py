
from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical, Grid
from textual.screen import ModalScreen
from textual.widgets import Label, Button, OptionList

from tui.interaction_logger import get_interaction_logger


class ConfirmDialog(ModalScreen[bool]):

    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-container"):
            yield Label(self.message, id="dialog-title")
            with Grid(classes="button-row"):
                yield Button("Confirm", variant="primary", id="confirm-ok")
                yield Button("Cancel", variant="error", id="confirm-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        approved = event.button.id == "confirm-ok"
        get_interaction_logger().event(
            "confirm_dialog_choice",
            message=self.message,
            button_id=event.button.id,
            approved=approved,
        )
        self.dismiss(approved)


class SelectDialog(ModalScreen):

    def __init__(self, title: str, options: list[Any]) -> None:
        super().__init__()
        self.title_str = title
        self.options_list = options

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog-container"):
            yield Label(self.title_str, id="dialog-title")
            yield OptionList(*self.options_list, id="select-list")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        selected = str(event.option.prompt)
        get_interaction_logger().event(
            "select_dialog_choice",
            title=self.title_str,
            option=selected,
        )
        self.dismiss(selected)
