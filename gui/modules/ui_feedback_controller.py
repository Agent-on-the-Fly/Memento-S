from __future__ import annotations

import logging

import flet as ft

from gui.i18n import t

logger = logging.getLogger(__name__)
from gui.widgets.chat_message import ChatMessage


class UIFeedbackController:
    """UI helper methods: status, messages, dialogs, token display."""

    def __init__(self, app):
        self.app = app
        self._current_toast = None
        self._timers = []  # 保存timer引用防止被垃圾回收

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

    def _dismiss_toast(self, toast_container):
        """Dismiss toast after timer expires"""

        def do_dismiss():
            try:
                if toast_container in self.app.page.overlay:
                    self.app.page.overlay.remove(toast_container)
                    self.app.page.update()
            except Exception as e:
                logger.error(f"Failed to dismiss toast: {e}")

        # 提交到主线程执行
        self.app.page.run_thread(do_dismiss)

    def show_snackbar(self, message: str, type: str = "info", duration: int = 500):
        """
        显示自定义 Toast 提示（使用页面级 SnackBar）

        Args:
            message: 提示消息
            type: 消息类型 - "info"/"warning"/"error"，默认 "info"
            duration: 显示时长（毫秒），默认 1000ms（1秒）
        """
        # 检查页面是否已初始化
        if not self.app.page:
            logger.debug("Cannot show toast: page not initialized yet")
            return

        # 类型配置
        config = {
            "info": {
                "icon": ft.Icons.INFO_OUTLINED,
                "icon_color": ft.Colors.BLUE_400,
                "border_color": ft.Colors.BLUE_700,
            },
            "warning": {
                "icon": ft.Icons.WARNING_OUTLINED,
                "icon_color": ft.Colors.ORANGE_400,
                "border_color": ft.Colors.ORANGE_700,
            },
            "error": {
                "icon": ft.Icons.ERROR_OUTLINED,
                "icon_color": ft.Colors.RED_400,
                "border_color": ft.Colors.RED_700,
            },
        }

        # 获取样式配置，默认为 info
        style = config.get(type, config["info"])

        # 创建 Toast 内容行（自适应宽度）
        toast_row = ft.Row(
            [
                ft.Icon(style["icon"], color=style["icon_color"], size=22),
                ft.Text(
                    message,
                    color=ft.Colors.WHITE,
                    size=14,
                    weight=ft.FontWeight.W_500,
                    no_wrap=False,
                    max_lines=2,
                    overflow=ft.TextOverflow.ELLIPSIS,
                ),
            ],
            spacing=12,
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            tight=True,  # 关键：让 Row 宽度适应内容
        )

        # 外层容器 - 包装 Row 并设置样式
        toast_container = ft.Container(
            content=toast_row,
            bgcolor=ft.Colors.GREY_900,
            border=ft.border.all(1, style["border_color"]),
            border_radius=30,
            padding=ft.padding.symmetric(horizontal=20, vertical=10),
        )

        # 使用 Positioned 定位到底部居中
        positioned_toast = ft.Container(
            content=toast_container,
            alignment=ft.alignment.Alignment(0, 1),  # 底部居中
            padding=ft.padding.only(bottom=24),
        )

        try:
            # 移除之前的 toast（如果有）
            if self._current_toast and self._current_toast in self.app.page.overlay:
                self.app.page.overlay.remove(self._current_toast)

            # 添加新的 toast 到 overlay
            self._current_toast = positioned_toast
            self.app.page.overlay.append(positioned_toast)
            self.app.page.update()

            # 使用 Timer 自动关闭
            from threading import Timer

            def dismiss_and_cleanup(container=positioned_toast):
                self._dismiss_toast(container)
                # 清理已完成的timer
                self._timers = [t for t in self._timers if t.is_alive()]

            timer = Timer(duration / 1000, dismiss_and_cleanup)
            timer.daemon = True
            self._timers.append(timer)
            timer.start()

        except Exception as e:
            logger.error(f"Failed to show toast: {e}")
