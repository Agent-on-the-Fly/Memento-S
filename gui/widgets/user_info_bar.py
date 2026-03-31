"""
UserInfoBar - User login entry / logged-in info display at the bottom of sidebar.

Separated from SessionSidebar to follow single responsibility principle.
"""

from typing import Callable, Optional

import flet as ft

from gui.i18n import t, add_observer


class UserInfoBar(ft.Container):
    """Displays user login entry or logged-in user info."""

    def __init__(
        self,
        on_login_click: Optional[Callable] = None,
        on_logout_click: Optional[Callable] = None,
    ):
        super().__init__()
        self.on_login_click = on_login_click
        self.on_logout_click = on_logout_click

        self._logout_btn: Optional[ft.IconButton] = None
        self._login_text: Optional[ft.Text] = None

        self._user_area = self._build_user_area()

        self.content = ft.Column(
            [
                ft.Divider(height=1, color=ft.Colors.GREY_800),
                self._user_area,
            ],
            spacing=0,
        )

        add_observer(self._on_language_changed)

    def _build_user_area(
        self, logged_in: bool = False, display_name: str = ""
    ) -> ft.Container:
        """Build the user area widget."""

        if logged_in and display_name:
            self._login_text = None
            self._logout_btn = ft.IconButton(
                icon=ft.Icons.LOGOUT,
                icon_color=ft.Colors.GREY_600,
                icon_size=16,
                tooltip=t("auth.logout"),
                width=28,
                height=28,
                mouse_cursor=ft.MouseCursor.CLICK,
                on_click=lambda e: self.on_logout_click()
                if self.on_logout_click
                else None,
                style=ft.ButtonStyle(
                    padding=ft.Padding(0, 0, 0, 0),
                    shape=ft.CircleBorder(),
                    overlay_color=ft.Colors.with_opacity(0.15, ft.Colors.RED_400),
                ),
                visible=False,
            )

            user_container = ft.Container(
                content=ft.Row(
                    [
                        ft.Icon(
                            ft.Icons.ACCOUNT_CIRCLE,
                            color=ft.Colors.BLUE_400,
                            size=24,
                        ),
                        ft.Text(
                            display_name,
                            size=14,
                            color=ft.Colors.GREY_300,
                            max_lines=1,
                            overflow=ft.TextOverflow.ELLIPSIS,
                            expand=True,
                        ),
                        self._logout_btn,
                    ],
                    spacing=8,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                padding=ft.Padding(12, 6, 8, 6),
                height=42,
            )

            def _on_enter(e):
                user_container.bgcolor = ft.Colors.GREY_800
                self._logout_btn.icon_color = ft.Colors.RED_400
                user_container.update()

            def _on_exit(e):
                user_container.bgcolor = None
                self._logout_btn.icon_color = ft.Colors.GREY_600
                user_container.update()

            return ft.GestureDetector(
                content=user_container,
                on_enter=_on_enter,
                on_exit=_on_exit,
                mouse_cursor=ft.MouseCursor.BASIC,
            )
        else:
            self._logout_btn = None
            self._login_text = ft.Text(
                t("auth.click_to_login"),
                size=14,
                color=ft.Colors.GREY_500,
            )
            login_container = ft.Container(
                content=ft.Row(
                    [
                        ft.Icon(
                            ft.Icons.ACCOUNT_CIRCLE_OUTLINED,
                            color=ft.Colors.GREY_500,
                            size=24,
                        ),
                        self._login_text,
                    ],
                    spacing=8,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                padding=ft.Padding(16, 10, 16, 10),
                height=42,
            )

            def _on_login_enter(e):
                login_container.bgcolor = ft.Colors.GREY_800
                login_container.update()

            def _on_login_exit(e):
                login_container.bgcolor = None
                login_container.update()

            return ft.GestureDetector(
                content=login_container,
                on_tap=lambda e: self.on_login_click()
                if self.on_login_click
                else None,
                on_enter=_on_login_enter,
                on_exit=_on_login_exit,
                mouse_cursor=ft.MouseCursor.CLICK,
            )

    def update_user_area(self, logged_in: bool = False, display_name: str = ""):
        """Update the user area after login/logout."""
        new_area = self._build_user_area(
            logged_in=logged_in, display_name=display_name
        )
        self._user_area = new_area
        self.content.controls[-1] = new_area
        try:
            if self.page:
                self.update()
        except RuntimeError:
            pass

    def _on_language_changed(self, new_lang: str):
        """Refresh UI text on language change."""
        try:
            if not self.page:
                return
        except RuntimeError:
            return

        if self._logout_btn:
            self._logout_btn.tooltip = t("auth.logout")
        elif self._login_text:
            self._login_text.value = t("auth.click_to_login")

        try:
            if self.page:
                self.update()
        except RuntimeError:
            pass
