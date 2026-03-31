"""
Login dialog widget for Memento-S GUI.

Modal dialog that can only be closed via the explicit X button.
Uses AlertDialog(modal=True) to keep keyboard focus inside dialog.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional

import flet as ft

from gui.i18n import t
from gui.modules.auth_service import AuthService

logger = logging.getLogger(__name__)


class LoginDialog:
    """Modal login dialog with verification-code flow."""

    COUNTDOWN_SECONDS = 60

    def __init__(
        self,
        page: ft.Page,
        auth_service: AuthService,
        on_login_success: Optional[Callable] = None,
    ):
        self.page = page
        self.auth = auth_service
        self.on_login_success = on_login_success

        self._overlay_entry: ft.AlertDialog | None = None
        self._countdown_remaining = 0
        self._countdown_timer_id = None

        self._account_field = ft.TextField(
            label=t("auth.account"),
            hint_text=t("auth.account_hint"),
            hint_style=ft.TextStyle(size=12, color="#808080"),
            text_style=ft.TextStyle(size=13, color="#e0e0e0"),
            label_style=ft.TextStyle(size=12, color="#a0a0a0"),
            border_color="#404040",
            focused_border_color="#3b82f6",
            cursor_color=ft.Colors.WHITE,
            bgcolor="#2d2d2d",
            border_radius=6,
            content_padding=ft.Padding(12, 14, 12, 14),
        )

        self._code_field = ft.TextField(
            label=t("auth.verification_code"),
            hint_text=t("auth.code_hint"),
            hint_style=ft.TextStyle(size=12, color="#808080"),
            text_style=ft.TextStyle(size=13, color="#e0e0e0"),
            label_style=ft.TextStyle(size=12, color="#a0a0a0"),
            border_color="#404040",
            focused_border_color="#3b82f6",
            cursor_color=ft.Colors.WHITE,
            bgcolor="#2d2d2d",
            border_radius=6,
            content_padding=ft.Padding(12, 14, 12, 14),
            expand=True,
        )

        self._send_code_label = ft.Text(
            t("auth.send_code"),
            size=13,
            color=ft.Colors.WHITE,
            no_wrap=True,
        )
        self._send_code_loading = ft.ProgressRing(
            width=16,
            height=16,
            stroke_width=2,
            color=ft.Colors.WHITE,
            visible=False,
        )
        self._send_code_content = ft.Stack(
            [
                ft.Container(
                    content=self._send_code_label,
                    alignment=ft.Alignment.CENTER,
                    expand=True,
                ),
                ft.Container(
                    content=self._send_code_loading,
                    alignment=ft.Alignment.CENTER,
                    expand=True,
                ),
            ],
            width=110,
            height=48,
        )
        self._send_code_btn = ft.TextButton(
            content=self._send_code_content,
            on_click=self._on_send_code_sync,
            style=ft.ButtonStyle(
                padding=ft.Padding(0, 0, 0, 0),
                shape=ft.RoundedRectangleBorder(radius=6),
                bgcolor={
                    ft.ControlState.DEFAULT: "#3b82f6",
                    ft.ControlState.HOVERED: ft.Colors.with_opacity(0.8, "#3b82f6"),
                    ft.ControlState.FOCUSED: ft.Colors.with_opacity(0.8, "#3b82f6"),
                    ft.ControlState.PRESSED: ft.Colors.with_opacity(0.7, "#3b82f6"),
                    ft.ControlState.DISABLED: ft.Colors.with_opacity(0.5, "#3b82f6"),
                },
                overlay_color=ft.Colors.with_opacity(0.15, ft.Colors.WHITE),
                mouse_cursor={
                    ft.ControlState.DEFAULT: ft.MouseCursor.CLICK,
                    ft.ControlState.DISABLED: ft.MouseCursor.BASIC,
                },
            ),
        )
        self._send_code_disabled = False

        self._login_label = ft.Text(
            t("auth.login"),
            size=14,
            color=ft.Colors.WHITE,
            weight=ft.FontWeight.W_600,
            no_wrap=True,
        )
        self._login_loading = ft.ProgressRing(
            width=18,
            height=18,
            stroke_width=2,
            color=ft.Colors.WHITE,
            visible=False,
        )
        self._login_content = ft.Stack(
            [
                ft.Container(
                    content=self._login_label,
                    alignment=ft.Alignment.CENTER,
                    expand=True,
                ),
                ft.Container(
                    content=self._login_loading,
                    alignment=ft.Alignment.CENTER,
                    expand=True,
                ),
            ],
            height=44,
        )
        self._login_btn = ft.TextButton(
            content=self._login_content,
            on_click=self._on_login_sync,
            style=ft.ButtonStyle(
                padding=ft.Padding(0, 0, 0, 0),
                shape=ft.RoundedRectangleBorder(radius=6),
                bgcolor={
                    ft.ControlState.DEFAULT: "#3b82f6",
                    ft.ControlState.HOVERED: ft.Colors.with_opacity(0.8, "#3b82f6"),
                    ft.ControlState.FOCUSED: ft.Colors.with_opacity(0.8, "#3b82f6"),
                    ft.ControlState.PRESSED: ft.Colors.with_opacity(0.7, "#3b82f6"),
                    ft.ControlState.DISABLED: ft.Colors.with_opacity(0.5, "#3b82f6"),
                },
                overlay_color=ft.Colors.with_opacity(0.15, ft.Colors.WHITE),
                mouse_cursor={
                    ft.ControlState.DEFAULT: ft.MouseCursor.CLICK,
                    ft.ControlState.DISABLED: ft.MouseCursor.BASIC,
                },
            ),
        )
        self._login_disabled = False

        self._error_text = ft.Text(
            value="",
            size=12,
            color=ft.Colors.RED_400,
            visible=False,
        )

    def show(self):
        """Show the login dialog as a true modal dialog."""
        if isinstance(self._overlay_entry, ft.AlertDialog) and getattr(
            self._overlay_entry, "open", False
        ):
            return

        self._account_field.value = ""
        self._code_field.value = ""
        self._error_text.visible = False
        self._error_text.value = ""
        self._reset_send_code_btn()

        self._login_disabled = False
        self._login_btn.disabled = False
        self._login_label.visible = True
        self._login_loading.visible = False

        close_btn = ft.IconButton(
            icon=ft.Icons.CLOSE,
            icon_color="#a0a0a0",
            icon_size=18,
            on_click=lambda e: self.close(),
            tooltip=t("common.close"),
            style=ft.ButtonStyle(
                padding=ft.Padding(4, 4, 4, 4),
            ),
            mouse_cursor=ft.MouseCursor.CLICK,
        )

        title_bar = ft.Row(
            [
                ft.Text(
                    t("auth.login"),
                    size=16,
                    weight=ft.FontWeight.W_600,
                    color="#e0e0e0",
                ),
                close_btn,
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        )

        code_row = ft.Row(
            [
                self._code_field,
                self._send_code_btn,
            ],
            spacing=8,
            vertical_alignment=ft.CrossAxisAlignment.END,
        )

        form_content = ft.Column(
            [
                title_bar,
                ft.Container(height=16),
                self._account_field,
                ft.Container(height=12),
                code_row,
                ft.Container(height=8),
                self._error_text,
                ft.Container(height=16),
                self._login_btn,
            ],
            spacing=0,
            tight=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )

        dialog_card = ft.Container(
            content=form_content,
            width=380,
            padding=ft.Padding(24, 24, 24, 24),
            bgcolor="#1e1e1e",
            border_radius=8,
            border=ft.border.all(0.5, "#383838"),
            shadow=ft.BoxShadow(
                spread_radius=2,
                blur_radius=20,
                color=ft.Colors.BLACK54,
            ),
        )

        self._overlay_entry = ft.AlertDialog(
            modal=True,
            bgcolor=ft.Colors.with_opacity(0.5, ft.Colors.BLACK),
            content_padding=0,
            inset_padding=20,
            actions_padding=0,
            title_padding=0,
            shape=ft.RoundedRectangleBorder(radius=0),
            content=dialog_card,
        )

        self.page.show_dialog(self._overlay_entry)
        self.page.update()

        try:
            self.page.run_task(self._focus_account_field)
        except Exception as e:
            logger.debug("[LoginDialog] failed to schedule focus: %s", e)

    def close(self):
        """Close the dialog via the X button."""
        self._countdown_timer_id = None
        self._countdown_remaining = 0

        if isinstance(self._overlay_entry, ft.AlertDialog):
            try:
                self.page.pop_dialog()
            except Exception:
                self._overlay_entry.open = False
                self.page.update()

            self._overlay_entry = None

    async def _focus_account_field(self):
        try:
            await asyncio.sleep(0)
            if self._overlay_entry is not None:
                await self._account_field.focus()
        except Exception as e:
            logger.debug("[LoginDialog] account_field.focus() failed: %s", e)

    def _show_error(self, msg: str):
        self._error_text.value = msg
        self._error_text.visible = True
        self.page.update()

    def _hide_error(self):
        self._error_text.visible = False
        self._error_text.value = ""

    def _reset_send_code_btn(self):
        self._send_code_label.value = t("auth.send_code")
        self._send_code_label.color = ft.Colors.WHITE
        self._send_code_label.visible = True
        self._send_code_loading.visible = False
        self._send_code_btn.disabled = False
        self._send_code_disabled = False
        self._countdown_remaining = 0

    def _on_send_code_sync(self, e):
        if self._send_code_disabled:
            return
        try:
            asyncio.create_task(self._on_send_code())
        except RuntimeError:
            loop = asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(self._on_send_code(), loop)
        except Exception as ex:
            logger.error("[LoginDialog] _on_send_code_sync error: %s", ex)

    async def _on_send_code(self):
        account = (self._account_field.value or "").strip()
        if not account:
            self._show_error(t("auth.account_required"))
            return

        if self._countdown_remaining > 0:
            return

        self._hide_error()
        self._send_code_disabled = True
        self._send_code_btn.disabled = True
        self._send_code_label.visible = False
        self._send_code_loading.visible = True
        self.page.update()

        try:
            success, msg = await self.auth.send_verification_code(account)
            print(
                f"[LoginDialog] send_verification_code result: "
                f"success={success}, msg={msg}"
            )
        except Exception as e:
            print(f"[LoginDialog] send_verification_code exception: {e}")
            success, msg = False, str(e)

        if success:
            self._countdown_remaining = self.COUNTDOWN_SECONDS
            self._start_countdown()
        else:
            self._send_code_disabled = False
            self._send_code_btn.disabled = False
            self._send_code_loading.visible = False
            self._send_code_label.visible = True
            self._send_code_label.value = t("auth.send_code")
            self._send_code_label.color = ft.Colors.WHITE
            self._show_error(t("auth.send_code_failed", error=msg))
            self.page.update()

    def _start_countdown(self):
        """Start the countdown timer."""
        self._update_countdown_display()
        if self._countdown_remaining > 0:
            self._countdown_timer_id = object()
            self.page.run_task(self._countdown_tick)

    def _update_countdown_display(self):
        """Update the countdown button display."""
        try:
            if self._overlay_entry is None:
                return

            if self._countdown_remaining > 0:
                self._send_code_loading.visible = False
                self._send_code_label.visible = True
                self._send_code_label.value = t(
                    "auth.resend_after", seconds=self._countdown_remaining
                )
                self._send_code_label.color = ft.Colors.with_opacity(
                    0.5, ft.Colors.WHITE
                )
                self._send_code_btn.disabled = True
                self._send_code_disabled = True
            else:
                self._send_code_loading.visible = False
                self._send_code_label.visible = True
                self._send_code_label.value = t("auth.send_code")
                self._send_code_label.color = ft.Colors.WHITE
                self._send_code_btn.disabled = False
                self._send_code_disabled = False

            self._send_code_btn.update()
        except Exception as e:
            logger.warning("[LoginDialog] Failed to update countdown display: %s", e)

    async def _countdown_tick(self):
        """Countdown tick coroutine - decrements every second."""
        timer_id = self._countdown_timer_id
        try:
            while (
                self._countdown_remaining > 0
                and timer_id == self._countdown_timer_id
                and self._overlay_entry is not None
            ):
                await asyncio.sleep(1)
                if (
                    timer_id != self._countdown_timer_id
                    or self._overlay_entry is None
                ):
                    return
                self._countdown_remaining -= 1
                self._update_countdown_display()
        except asyncio.CancelledError:
            logger.debug("[LoginDialog] Countdown task cancelled")
        except Exception as e:
            logger.error("[LoginDialog] Countdown tick error: %s", e)
        finally:
            if self._countdown_remaining <= 0:
                self._countdown_remaining = 0
                try:
                    self._update_countdown_display()
                except Exception:
                    pass

    def _on_login_sync(self, e):
        if self._login_disabled:
            return
        try:
            asyncio.create_task(self._on_login())
        except RuntimeError:
            loop = asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(self._on_login(), loop)
        except Exception as ex:
            logger.error("[LoginDialog] _on_login_sync error: %s", ex)

    async def _on_login(self):
        account = (self._account_field.value or "").strip()
        code = (self._code_field.value or "").strip()

        if not account:
            self._show_error(t("auth.account_required"))
            return
        if not code:
            self._show_error(t("auth.code_required"))
            return

        self._hide_error()
        self._login_disabled = True
        self._login_btn.disabled = True
        self._login_label.visible = False
        self._login_loading.visible = True
        self.page.update()

        try:
            success, msg = await self.auth.login(account, code)
        except Exception as e:
            logger.error("[LoginDialog] login exception: %s", e)
            success, msg = False, str(e)

        self._login_disabled = False
        self._login_btn.disabled = False
        self._login_label.visible = True
        self._login_loading.visible = False

        if success:
            self.close()
            if self.on_login_success:
                self.on_login_success()
        else:
            self._show_error(t("auth.login_failed", error=msg))
            self.page.update()