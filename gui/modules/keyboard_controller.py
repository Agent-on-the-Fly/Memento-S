from __future__ import annotations

import asyncio
import time

import flet as ft

from gui.i18n import t


class KeyboardController:
    """Handle keyboard interactions and shortcuts."""

    def __init__(self, app):
        self.app = app

    def on_keyboard(self, e: ft.KeyboardEvent):
        if self.app.command_hints_container.visible:
            if e.key == "Arrow Up":
                if self.app.filtered_commands:
                    self.app.selected_command_index = (
                        len(self.app.filtered_commands) - 1
                        if self.app.selected_command_index <= 0
                        else self.app.selected_command_index - 1
                    )
                    self.app._update_command_hints_highlight()
                return

            if e.key == "Arrow Down":
                if self.app.filtered_commands:
                    self.app.selected_command_index = (
                        0
                        if self.app.selected_command_index
                        >= len(self.app.filtered_commands) - 1
                        else self.app.selected_command_index + 1
                    )
                    self.app._update_command_hints_highlight()
                return

            if e.key == "Enter" and self.app.selected_command_index >= 0:
                selected_cmd = self.app.filtered_commands[
                    self.app.selected_command_index
                ]
                print(
                    f"[DEBUG] Keyboard Enter pressed, selected_cmd: {selected_cmd}, index: {self.app.selected_command_index}"
                )
                print(f"[DEBUG] filtered_commands: {self.app.filtered_commands}")
                asyncio.create_task(self.app._on_command_hint_selected(selected_cmd))
                return

            if e.key == "Escape":
                self.app.command_hints_container.visible = False
                self.app.selected_command_index = -1
                self.app.page.update()
                return

        if e.key == "Escape":
            current_time = time.time()
            if current_time - self.app._last_esc_time < self.app._DOUBLE_ESC_INTERVAL:
                if self.app.is_processing:
                    self.app._on_stop_generation(None)
                    self.app._set_status(t("status.force_stopped"))
                else:
                    self.app.message_input.value = ""
                    self.app.page.update()
            self.app._last_esc_time = current_time
            return

        if (e.ctrl or e.meta) and e.key == "Enter":
            self.app.command_hints_container.visible = False
            self.app.selected_command_index = -1
            asyncio.create_task(self.app._send_current_message())
            return

        if e.ctrl or e.meta:
            if e.key == "n":
                self.app._on_new_chat()
            elif e.key == "l":
                self.app._on_clear_chat()
            elif e.key == "q":
                self.app.exit_app()
