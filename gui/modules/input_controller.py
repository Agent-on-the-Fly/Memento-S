from __future__ import annotations

import asyncio

import flet as ft


class InputController:
    """Handle input box changes and command hint interactions."""

    def __init__(self, app):
        self.app = app

    async def _on_hint_clicked(self, command: str):
        """Handle click on a command hint."""
        print(f"[DEBUG] Hint clicked: {command}")
        await self.app._on_command_hint_selected(command)

    async def on_send_message(self, e=None):
        self.app.command_hints_container.visible = False
        self.app.selected_command_index = -1
        await self.app._send_current_message()

    def on_input_change(self, e):
        if not self.app.page:
            return

        value = self.app.message_input.value or ""
        self.app.selected_command_index = -1

        if value.startswith("/"):
            self.app.filtered_commands = [
                cmd for cmd in self.app.COMMANDS if cmd.startswith(value)
            ]
            if self.app.filtered_commands:
                self.app.command_hints.controls.clear()
                self.app.command_hint_items = []
                for cmd in self.app.filtered_commands:
                    hint_row = ft.Container(
                        content=ft.Row(
                            [
                                ft.Text(
                                    cmd,
                                    size=13,
                                    weight=ft.FontWeight.W_600,
                                    color=ft.Colors.BLUE_400,
                                ),
                                ft.Text(
                                    f" — {self.app.COMMANDS[cmd]}",
                                    size=12,
                                    color=ft.Colors.GREY_400,
                                ),
                            ],
                            spacing=4,
                        ),
                        padding=ft.Padding.symmetric(horizontal=12, vertical=6),
                        on_click=lambda e, c=cmd: asyncio.create_task(
                            self._on_hint_clicked(c)
                        ),
                        ink=True,
                        border_radius=ft.BorderRadius.all(4),
                        bgcolor=ft.Colors.TRANSPARENT,
                    )
                    self.app.command_hint_items.append(hint_row)
                    self.app.command_hints.controls.append(hint_row)
                self.app.command_hints_container.visible = True
            else:
                self.app.command_hints_container.visible = False
                self.app.filtered_commands = []
                self.app.command_hint_items = []
        else:
            self.app.command_hints_container.visible = False
            self.app.filtered_commands = []
            self.app.command_hint_items = []

        self.app.page.update()

    async def on_command_hint_selected(self, command: str):
        if not self.app.page:
            return
        print(f"[DEBUG] Command selected: {command}")
        print(f"[DEBUG] Before setting value: '{self.app.message_input.value}'")

        # Set the value with the command and a trailing space
        self.app.message_input.value = f"{command} "
        print(f"[DEBUG] After setting value: '{self.app.message_input.value}'")

        # Hide command hints
        self.app.command_hints_container.visible = False
        self.app.selected_command_index = -1
        self.app.filtered_commands = []
        self.app.command_hint_items = []

        # Update page to reflect changes
        self.app.page.update()
        print(f"[DEBUG] After update: '{self.app.message_input.value}'")

        # Focus the input after a short delay
        try:
            await self.app.message_input.focus()
        except Exception as e:
            print(f"[DEBUG] Focus error (non-critical): {e}")

    def update_command_hints_highlight(self):
        if not self.app.page:
            return

        for i, item in enumerate(self.app.command_hint_items):
            if i == self.app.selected_command_index:
                item.bgcolor = ft.Colors.GREY_700
                item.border = ft.Border(
                    left=ft.BorderSide(3, ft.Colors.BLUE_400),
                    top=ft.BorderSide(0, ft.Colors.TRANSPARENT),
                    right=ft.BorderSide(0, ft.Colors.TRANSPARENT),
                    bottom=ft.BorderSide(0, ft.Colors.TRANSPARENT),
                )
                if hasattr(self.app.command_hints, "scroll_to"):
                    asyncio.create_task(
                        self.app.command_hints.scroll_to(
                            offset=float(i * 32),
                            duration=150,
                        )
                    )
            else:
                item.bgcolor = ft.Colors.TRANSPARENT
                item.border = None

        self.app.page.update()
