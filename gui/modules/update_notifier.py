"""
Update Notifier UI for Memento-S GUI.

Provides notification UI for auto-update feature:
    - Download progress floating window
    - Download complete notification
    - Install confirmation dialog
"""

from __future__ import annotations

import flet as ft
from typing import Callable, TYPE_CHECKING

from gui.modules.auto_update_manager import (
    AutoUpdateManager,
    UpdateStatus,
    UpdateInfo,
    DownloadProgress,
)
from gui.i18n import t
from utils.logger import logger

if TYPE_CHECKING:
    from flet import Page


class UpdateNotifier:
    """
    Manages update notification UI.

    Usage:
        notifier = UpdateNotifier(page, show_error, show_snackbar)
        await notifier.initialize()
    """

    def __init__(
        self,
        page: Page,
        show_error: Callable[[str], None],
        show_snackbar: Callable[[str], None],
    ):
        self.page = page
        self.show_error = show_error
        self.show_snackbar = show_snackbar

        self._manager = AutoUpdateManager()
        self._progress_dialog: ft.AlertDialog | None = None
        self._download_progress: ft.ProgressBar | None = None
        self._download_status: ft.Text | None = None
        self._notification_card: ft.Card | None = None

        self._setup_callbacks()

    def _setup_callbacks(self):
        """Setup manager callbacks."""
        self._manager.set_callbacks(
            on_status_change=self._on_status_change,
            on_progress=self._on_progress,
            on_download_complete=self._on_download_complete,
            on_error=self._on_error,
        )

    async def initialize(self):
        """Initialize and start auto-update check."""
        logger.info("[UpdateNotifier] Initializing auto-update")
        await self._manager.start_auto_check()

    def _on_status_change(self, status: UpdateStatus):
        """Handle status changes."""
        logger.info(f"[UpdateNotifier] Status: {status.name}")

        if status == UpdateStatus.DOWNLOADING:
            self._show_progress_dialog()
        elif status == UpdateStatus.DOWNLOADED:
            self._close_progress_dialog()
            self._show_install_notification()
        elif status == UpdateStatus.ERROR:
            self._close_progress_dialog()
        elif status == UpdateStatus.CANCELLED:
            self._close_progress_dialog()
            self.show_snackbar(t("update.cancelled"))

    def _on_progress(self, progress: DownloadProgress):
        """Handle download progress."""
        if self._download_progress and self.page:
            self._download_progress.value = progress.percentage

            # Update status text
            if self._download_status:
                downloaded_mb = progress.downloaded / (1024 * 1024)
                total_mb = progress.total_size / (1024 * 1024)
                speed_mbps = (progress.speed / (1024 * 1024)) if progress.speed > 0 else 0

                status_text = f"{downloaded_mb:.1f} MB / {total_mb:.1f} MB"
                if speed_mbps > 0:
                    status_text += f" ({speed_mbps:.1f} MB/s)"

                self._download_status.value = status_text

            self.page.update()

    def _on_download_complete(self, update_info: UpdateInfo):
        """Handle download completion."""
        logger.info(f"[UpdateNotifier] Download complete: {update_info.version}")

    def _on_error(self, message: str):
        """Handle errors."""
        logger.error(f"[UpdateNotifier] Error: {message}")
        self.show_error(message)

    def _show_progress_dialog(self):
        """Show download progress dialog."""
        if self._progress_dialog and self._progress_dialog.open:
            return

        self._download_progress = ft.ProgressBar(value=0, width=300)
        self._download_status = ft.Text("Starting download...", size=12)

        self._progress_dialog = ft.AlertDialog(
            title=ft.Text(t("update.downloading"), size=16),
            content=ft.Column(
                [
                    ft.Text(t("update.downloading_desc"), size=13),
                    ft.Divider(height=8, color=ft.Colors.TRANSPARENT),
                    self._download_progress,
                    self._download_status,
                ],
                spacing=8,
                tight=True,
            ),
            actions=[
                ft.TextButton(
                    t("common.cancel"),
                    on_click=lambda e: self._manager.cancel_download(),
                ),
            ],
        )

        self.page.overlay.append(self._progress_dialog)
        self._progress_dialog.open = True
        self.page.update()

    def _close_progress_dialog(self):
        """Close progress dialog."""
        if self._progress_dialog:
            self._progress_dialog.open = False
            self.page.update()
            self._progress_dialog = None

    def _show_install_notification(self):
        """Show install notification card."""
        if not self._manager.current_update:
            return

        update_info = self._manager.current_update

        def on_install_click(e):
            self._hide_notification()
            self._show_install_confirmation(update_info)

        def on_dismiss_click(e):
            self._hide_notification()
            self.show_snackbar(t("update.install_later"))

        self._notification_card = ft.Card(
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.Row(
                            [
                                ft.Icon(
                                    ft.icons.SYSTEM_UPDATE,
                                    color=ft.Colors.BLUE,
                                ),
                                ft.Text(
                                    t("update.ready_title"),
                                    size=14,
                                    weight=ft.FontWeight.BOLD,
                                ),
                                ft.IconButton(
                                    icon=ft.icons.CLOSE,
                                    icon_size=16,
                                    on_click=on_dismiss_click,
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        ),
                        ft.Text(
                            t("update.ready_desc", version=update_info.version),
                            size=12,
                        ),
                        ft.Row(
                            [
                                ft.TextButton(
                                    t("common.later"),
                                    on_click=on_dismiss_click,
                                ),
                                ft.ElevatedButton(
                                    t("update.install_now"),
                                    icon=ft.icons.INSTALL_DESKTOP,
                                    on_click=on_install_click,
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.END,
                        ),
                    ],
                    spacing=8,
                ),
                padding=16,
            ),
            elevation=4,
        )

        # Add to page as a snackbar-like notification at bottom right
        notification_container = ft.Container(
            content=self._notification_card,
            right=20,
            bottom=20,
            animate=ft.animation.Animation(300, ft.AnimationCurve.EASE_OUT),
        )

        self.page.overlay.append(notification_container)
        self.page.update()

        # Store reference to remove later
        self._notification_container = notification_container

    def _hide_notification(self):
        """Hide notification card."""
        if hasattr(self, '_notification_container') and self._notification_container:
            if self._notification_container in self.page.overlay:
                self.page.overlay.remove(self._notification_container)
            self._notification_container = None
            self._notification_card = None
            self.page.update()

    def _show_install_confirmation(self, update_info: UpdateInfo):
        """Show install confirmation dialog."""
        def on_confirm(e):
            dialog.open = False
            self.page.update()
            self._do_install()

        def on_cancel(e):
            dialog.open = False
            self.page.update()

        dialog = ft.AlertDialog(
            title=ft.Text(t("update.confirm_title")),
            content=ft.Text(
                t(
                    "update.confirm_desc",
                    version=update_info.version,
                    current=update_info.current_version,
                )
            ),
            actions=[
                ft.TextButton(t("common.cancel"), on_click=on_cancel),
                ft.ElevatedButton(
                    t("update.restart_and_install"),
                    icon=ft.icons.RESTART_ALT,
                    on_click=on_confirm,
                ),
            ],
        )

        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()

    async def _do_install(self):
        """Execute installation."""
        # Show installing dialog
        installing_dialog = ft.AlertDialog(
            title=ft.Text(t("update.installing")),
            content=ft.Column(
                [
                    ft.Text(t("update.installing_desc")),
                    ft.ProgressRing(),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

        self.page.overlay.append(installing_dialog)
        installing_dialog.open = True
        self.page.update()

        # Perform installation
        success = await self._manager.install_update(
            page=self.page,
            on_complete=lambda: self._on_install_complete(),
        )

        installing_dialog.open = False
        self.page.update()

        if success:
            # Close app for restart
            self.page.window.close()

    def _on_install_complete(self):
        """Called when installation completes."""
        self.show_snackbar(t("update.install_complete"))

    def check_for_updates(self):
        """Manually check for updates."""
        asyncio.create_task(self._manager.check_for_update())

    def get_manager(self) -> AutoUpdateManager:
        """Get the update manager instance."""
        return self._manager


__all__ = ["UpdateNotifier"]
