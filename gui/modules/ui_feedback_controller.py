from __future__ import annotations

import flet as ft

from gui.i18n import t
from gui.widgets.chat_message import ChatMessage


class UIFeedbackController:
    """UI helper methods: status, messages, dialogs, token display."""

    def __init__(self, app):
        self.app = app

    def on_clear_chat(self):
        self.app.chat_list.controls.clear()
        self.app.page.update()

    def add_system_message(self, text: str):
        msg = ChatMessage(
            text,
            is_user=False,
            max_width=self.app.page.width - 400 if self.app.page.width else 700,
        )
        self.app.chat_list.controls.append(msg)
        self.app.page.update()

    def update_token_display(self, current: int = None, max_tokens: int | None = None):
        if current is None:
            current = self.app.total_tokens
        if max_tokens is None:
            from middleware.config import g_config
            max_tokens = g_config.llm.current_profile.input_budget
        if hasattr(self.app, "token_text_bottom") and self.app.token_text_bottom:
            self.app.token_text_bottom.value = t(
                "input.token_display", current=current, max=max_tokens
            )
        if (
            hasattr(self.app, "token_progress_bottom")
            and self.app.token_progress_bottom
        ):
            self.app.token_progress_bottom.value = current / max(max_tokens, 1)
            if current > max_tokens * 0.9:
                self.app.token_progress_bottom.color = ft.Colors.RED_400
            elif current > max_tokens * 0.7:
                self.app.token_progress_bottom.color = ft.Colors.ORANGE_400
            else:
                self.app.token_progress_bottom.color = ft.Colors.BLUE_400
        if self.app.page:
            self.app.page.update()

    def update_toolbar_title(self, title: str = None):
        """Update toolbar title. Uses provided title or defaults to translated text."""
        if self.app.toolbar:
            translated_default = t("toolbar.new_conversation")
            self.app.toolbar.update_title(title if title else translated_default)

    def set_status(self, text: str):
        if self.app.status_text:
            self.app.status_text.value = text
            self.app.page.update()

    def show_error(self, message: str):
        def close_dialog(e):
            dialog.open = False
            self.app.page.update()

        dialog = ft.AlertDialog(
            title=ft.Text(t("dialogs.error_title"), color=ft.Colors.RED_400),
            content=ft.Text(message),
            actions=[ft.TextButton(t("dialogs.confirm"), on_click=close_dialog)],
        )
        self.app.page.dialog = dialog
        dialog.open = True
        self.app.page.update()

    def show_snackbar(self, message: str):
        try:
            # 新版 Flet API
            self.app.page.show_snack_bar(ft.SnackBar(content=ft.Text(message)))
        except AttributeError:
            # 旧版 Flet API
            snackbar = ft.SnackBar(content=ft.Text(message))
            self.app.page.snack_bar = snackbar
            snackbar.open = True
            self.app.page.update()
