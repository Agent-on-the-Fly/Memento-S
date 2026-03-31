"""
Update Notifier UI for Memento-S GUI.

Provides notification UI for auto-update feature:
    - Download progress floating window
    - Download complete notification
    - Install confirmation dialog
    - Force update modal dialog
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
        manager = AutoUpdateManager()
        notifier = UpdateNotifier(page, manager, show_error, show_snackbar)
        await notifier.initialize()
    """

    def __init__(
        self,
        page: Page,
        manager: AutoUpdateManager,
        show_error: Callable[[str], None],
        show_snackbar: Callable[[str], None],
    ):
        self.page = page
        self.show_error = show_error
        self.show_snackbar = show_snackbar

        self._manager = manager
        self._progress_dialog: ft.AlertDialog | None = None
        self._download_progress: ft.ProgressBar | None = None
        self._download_status: ft.Text | None = None
        self._notification_card: ft.Card | None = None
        self._notification_container: ft.Container | None = None
        self._force_update_dialog: ft.AlertDialog | None = None

        self._listener = self._manager.add_listener(
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
        is_auto = self._manager.is_auto_check
        is_force = self._manager.is_force_update
        logger.info(f"[UpdateNotifier] Status: {status.name}, auto_check={is_auto}, force={is_force}")

        if status == UpdateStatus.CHECKING:
            pass
        elif status == UpdateStatus.AVAILABLE:
            # Only show the "available" notification when auto_download=False and
            # this is NOT a force_update (force_update always triggers a download
            # in the manager layer regardless of auto_download, so the status will
            # immediately transition to DOWNLOADING — but the AVAILABLE callback
            # still fires first, so we must guard against it here).
            auto_download = self._manager.get_ota_config_value("auto_download", True)
            if is_auto and not auto_download and not is_force and self._manager.current_update:
                self._show_update_available_notification(self._manager.current_update)
        elif status == UpdateStatus.DOWNLOADING:
            if not is_auto:
                self._show_progress_dialog()
            elif is_force:
                # Show a non-cancellable progress indicator for forced downloads.
                self._show_force_download_progress()
        elif status == UpdateStatus.DOWNLOADED:
            self._close_progress_dialog()
            self._close_force_download_progress()
            if is_force:
                # Force update: bypass install_confirmation and show modal dialog.
                if self._manager.current_update:
                    self._show_force_update_dialog(self._manager.current_update)
            elif is_auto:
                self._show_install_notification()
        elif status == UpdateStatus.ERROR:
            self._close_progress_dialog()
            self._close_force_download_progress()
        elif status == UpdateStatus.CANCELLED:
            self._close_progress_dialog()
            self._close_force_download_progress()
            self.show_snackbar(t("update.cancelled"))

    def _on_progress(self, progress: DownloadProgress):
        """Handle download progress."""
        if self._download_progress and self.page:
            self._download_progress.value = progress.percentage

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

    # ------------------------------------------------------------------
    # Public helpers (called from SettingsPanel)
    # ------------------------------------------------------------------

    def show_progress_dialog(self):
        """Public method to show/reopen download progress dialog.

        Can be called externally (e.g. from settings panel) when the user
        re-enters a view while a download is already in progress.
        """
        self._show_progress_dialog()

    # ------------------------------------------------------------------
    # Download progress dialog
    # ------------------------------------------------------------------

    def _show_progress_dialog(self):
        """Show download progress dialog (cancellable, for manual downloads)."""
        if self._progress_dialog and self._progress_dialog.open:
            return

        if self._progress_dialog and not self._progress_dialog.open:
            if self._progress_dialog in self.page.overlay:
                self.page.overlay.remove(self._progress_dialog)
            self._progress_dialog = None
            self._download_progress = None
            self._download_status = None

        initial_progress = self._manager.progress.percentage
        self._download_progress = ft.ProgressBar(value=initial_progress, width=300)
        self._download_status = ft.Text(t("update.starting_download"), size=12)

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

    def _show_force_download_progress(self):
        """Show a non-cancellable progress indicator for forced downloads."""
        if self._progress_dialog and self._progress_dialog.open:
            return

        initial_progress = self._manager.progress.percentage
        self._download_progress = ft.ProgressBar(value=initial_progress, width=300)
        self._download_status = ft.Text(t("update.force_update_downloading"), size=12)

        self._progress_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(t("update.force_update_title"), size=16),
            content=ft.Column(
                [
                    ft.Text(t("update.force_update_desc"), size=13),
                    ft.Divider(height=8, color=ft.Colors.TRANSPARENT),
                    self._download_progress,
                    self._download_status,
                ],
                spacing=8,
                tight=True,
            ),
        )

        self.page.overlay.append(self._progress_dialog)
        self._progress_dialog.open = True
        self.page.update()

    def _close_force_download_progress(self):
        """Close the force-download progress dialog (alias for clarity)."""
        self._close_progress_dialog()

    def _close_progress_dialog(self):
        """Close progress dialog."""
        if self._progress_dialog:
            self._progress_dialog.open = False
            self.page.update()
            self._progress_dialog = None
            self._download_progress = None
            self._download_status = None

    # ------------------------------------------------------------------
    # Notification cards (bottom-right overlay)
    # ------------------------------------------------------------------

    def _show_update_available_notification(self, update_info: UpdateInfo):
        """Show a notification card when a new version is available but auto_download=False.

        Allows the user to choose to download now or dismiss.
        """
        self._hide_notification()

        def on_download_click(e):
            self._hide_notification()
            self.page.run_task(self._do_background_download, update_info)

        def on_dismiss_click(e):
            self._hide_notification()
            self.show_snackbar(t("update.install_later"))

        self._notification_card = ft.Card(
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.Row(
                            [
                                ft.Icon(ft.icons.Icons.SYSTEM_UPDATE, color=ft.Colors.BLUE),
                                ft.Text(
                                    t("update.available"),
                                    size=14,
                                    weight=ft.FontWeight.BOLD,
                                ),
                                ft.IconButton(
                                    icon=ft.icons.Icons.CLOSE,
                                    icon_size=16,
                                    on_click=on_dismiss_click,
                                ),
                            ],
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        ),
                        ft.Text(
                            t("update.new_version", version=update_info.version),
                            size=12,
                        ),
                        ft.Row(
                            [
                                ft.TextButton(t("common.later"), on_click=on_dismiss_click),
                                ft.ElevatedButton(
                                    t("update.download_now"),
                                    icon=ft.icons.Icons.DOWNLOAD,
                                    on_click=on_download_click,
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

        notification_container = ft.Container(
            content=self._notification_card,
            right=20,
            bottom=20,
            animate=ft.Animation(300, ft.AnimationCurve.EASE_OUT),
        )

        self.page.overlay.append(notification_container)
        self.page.update()
        self._notification_container = notification_container

    def _show_install_notification(self):
        """Show install notification card after a silent auto-download completes.

        Respects install_confirmation config: when False, clicking "Install Now"
        skips the confirmation dialog and installs directly. This only applies to
        the auto-update flow; manual checks always show the confirmation dialog.
        """
        if not self._manager.current_update:
            return

        update_info = self._manager.current_update

        def on_install_click(e):
            self._hide_notification()
            install_confirmation = self._manager.get_ota_config_value("install_confirmation", True)
            if install_confirmation:
                self._show_install_confirmation(update_info)
            else:
                self.page.run_task(self._do_install)

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
                                    ft.icons.Icons.SYSTEM_UPDATE,
                                    color=ft.Colors.BLUE,
                                ),
                                ft.Text(
                                    t("update.ready_title"),
                                    size=14,
                                    weight=ft.FontWeight.BOLD,
                                ),
                                ft.IconButton(
                                    icon=ft.icons.Icons.CLOSE,
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
                                    icon=ft.icons.Icons.INSTALL_DESKTOP,
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

        notification_container = ft.Container(
            content=self._notification_card,
            right=20,
            bottom=20,
            animate=ft.Animation(300, ft.AnimationCurve.EASE_OUT),
        )

        self.page.overlay.append(notification_container)
        self.page.update()
        self._notification_container = notification_container

    def _hide_notification(self):
        """Hide notification card."""
        if self._notification_container:
            if self._notification_container in self.page.overlay:
                self.page.overlay.remove(self._notification_container)
            self._notification_container = None
            self._notification_card = None
            self.page.update()

    # ------------------------------------------------------------------
    # Force update modal dialog
    # ------------------------------------------------------------------

    def _show_force_update_dialog(self, update_info: UpdateInfo):
        """Show a modal, non-dismissible force-update dialog.

        Bypasses install_confirmation: installation starts immediately on confirm.
        """
        if self._force_update_dialog and self._force_update_dialog.open:
            return

        def on_install_click(e):
            self._force_update_dialog.open = False
            self.page.update()
            self.page.run_task(self._do_install)

        self._force_update_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Row(
                [
                    ft.Icon(ft.icons.Icons.WARNING_AMBER, color=ft.Colors.ORANGE),
                    ft.Text(
                        t("update.force_update_title"),
                        size=16,
                        weight=ft.FontWeight.BOLD,
                    ),
                ],
                spacing=8,
            ),
            content=ft.Column(
                [
                    ft.Text(t("update.force_update_desc"), size=13),
                    ft.Divider(height=8, color=ft.Colors.TRANSPARENT),
                    ft.Text(
                        t("update.new_version", version=update_info.version),
                        size=12,
                        color=ft.Colors.GREY_500,
                    ),
                ],
                tight=True,
                spacing=6,
            ),
            actions=[
                ft.ElevatedButton(
                    t("update.install_now"),
                    icon=ft.icons.Icons.INSTALL_DESKTOP,
                    on_click=on_install_click,
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self.page.overlay.append(self._force_update_dialog)
        self._force_update_dialog.open = True
        self.page.update()

    # ------------------------------------------------------------------
    # Install confirmation dialog
    # ------------------------------------------------------------------

    def _show_install_confirmation(self, update_info: UpdateInfo):
        """Show install confirmation dialog."""
        def on_confirm(e):
            dialog.open = False
            self.page.update()
            self.page.run_task(self._do_install)

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
                    icon=ft.icons.Icons.RESTART_ALT,
                    on_click=on_confirm,
                ),
            ],
        )

        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()

    # ------------------------------------------------------------------
    # Installation execution
    # ------------------------------------------------------------------

    async def _do_background_download(self, update_info: UpdateInfo):
        """Download update triggered by user from the available-notification card."""
        self._show_progress_dialog()
        success = await self._manager.download_update(update_info)
        if not success:
            self.show_error(t("update.download_failed"))

    async def _do_install(self):
        """Execute installation."""
        installing_dialog = ft.AlertDialog(
            modal=True,
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

        success = await self._manager.install_update(
            page=self.page,
            on_complete=lambda: self._on_install_complete(),
        )

        installing_dialog.open = False
        self.page.update()

        if success:
            await self.page.window.close()

    def _on_install_complete(self):
        """Called when installer script is launched successfully."""
        self.show_snackbar(t("update.install_complete"))


__all__ = ["UpdateNotifier"]
