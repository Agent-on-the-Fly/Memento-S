"""
Toolbar widget with conversation title and actions
"""

import flet as ft
from typing import Callable, Optional

from gui.i18n import t, add_observer


class Toolbar(ft.Container):
    """Toolbar with conversation title and actions."""

    def __init__(
        self,
        on_settings: Optional[Callable] = None,
        on_clear: Optional[Callable] = None,
        on_export: Optional[Callable] = None,
    ):
        super().__init__()

        self.on_settings = on_settings
        self.on_clear = on_clear
        self.on_export = on_export

        # Register language change observer
        add_observer(self._on_language_changed)

        # Conversation title
        self.title_text = ft.Text(
            t("toolbar.new_conversation"),
            size=16,
            weight=ft.FontWeight.W_600,
            color=ft.Colors.WHITE,
        )

        # Create action buttons with translated tooltips
        self.clear_btn = ft.IconButton(
            icon=ft.icons.Icons.CLEAR_ALL,
            tooltip=t("toolbar.tooltip.clear"),
            on_click=lambda e: on_clear() if on_clear else None,
            visible=False,
        )

        self.export_btn = ft.IconButton(
            icon=ft.icons.Icons.DOWNLOAD,
            tooltip=t("toolbar.tooltip.export"),
            on_click=lambda e: on_export() if on_export else None,
            visible=False,
        )

        self.settings_btn = ft.IconButton(
            icon=ft.icons.Icons.SETTINGS,
            tooltip=t("toolbar.tooltip.settings"),
            on_click=lambda e: on_settings() if on_settings else None,
        )

        self.content = ft.Row(
            [
                # Title with icon
                ft.Row(
                    [
                        ft.Icon(
                            ft.icons.Icons.CHAT_BUBBLE,
                            size=20,
                            color=ft.Colors.BLUE_400,
                        ),
                        self.title_text,
                    ],
                    spacing=10,
                ),
                ft.Container(expand=True),  # Spacer
                # Actions
                ft.Row(
                    [
                        self.clear_btn,
                        self.export_btn,
                        self.settings_btn,
                    ],
                    spacing=4,
                ),
            ],
            spacing=16,
        )

        self.padding = ft.Padding.symmetric(horizontal=16, vertical=12)
        self.bgcolor = ft.Colors.GREY_900
        self.border = ft.Border.only(bottom=ft.BorderSide(1, ft.Colors.GREY_800))

    def _on_language_changed(self, new_lang: str):
        """Handle language change - refresh UI texts."""
        self.title_text.value = t("toolbar.new_conversation")
        self.clear_btn.tooltip = t("toolbar.tooltip.clear")
        self.export_btn.tooltip = t("toolbar.tooltip.export")
        self.settings_btn.tooltip = t("toolbar.tooltip.settings")
        if self.page:
            self.page.update()

    def update_title(self, title: str):
        """Update conversation title."""
        self.title_text.value = title if title else t("toolbar.new_conversation")
        if self.page:
            self.page.update()
