"""
Main layout manager with file browser drawer support
"""

import flet as ft
from pathlib import Path
from typing import Optional
import time
import asyncio

from gui.widgets.file_browser import FileBrowserDrawer
from gui.i18n import t


class MainLayout:
    """Main application layout with file browser drawer support."""

    def __init__(self, page: ft.Page, sidebar, main_area, workspace_path: Path):
        self.page = page
        self.sidebar = sidebar
        self.main_area = main_area
        self.workspace_path = workspace_path
        self.is_file_drawer_visible = False

        # File drawer reference
        self.file_drawer: Optional[FileBrowserDrawer] = None

        # Build the initial layout
        self._build_layout()

        # Open file browser by default after layout is mounted to page
        self.page.run_task(self._open_file_browser_delayed)

    def _build_layout(self):
        """Build the initial layout."""
        # Divider between sidebar and main area
        self.divider1 = ft.VerticalDivider(width=1, color=ft.Colors.GREY_800)

        # Create toggle button
        self.toggle_btn = self._create_toggle_button()

        # Wrapper to center toggle button vertically using Column + Spacer
        self.toggle_wrapper = ft.Column(
            [
                ft.Container(expand=True),  # Top spacer
                self.toggle_btn,
                ft.Container(expand=True),  # Bottom spacer
            ],
            width=16,
            spacing=0,
        )

        # Base layout: [Sidebar] | [Main Area] (without toggle)
        self.content_row = ft.Row(
            [
                self.sidebar,
                self.divider1,
                self.main_area,
            ],
            expand=True,
            spacing=0,
        )

        # Use Stack: content at bottom, toggle button absolutely positioned
        # Toggle position will be updated dynamically based on file browser state
        self.toggle_container = ft.Container(
            content=self.toggle_wrapper,
            right=0,  # Initial position at right edge
            top=0,
            bottom=0,
            width=16,
        )

        self.row = ft.Stack(
            [
                self.content_row,
                self.toggle_container,
            ],
            expand=True,
        )

    def _create_toggle_button(self):
        """Create the toggle button for file browser."""
        # Icon widget - chevron left (closed state, hint to open)
        icon_widget = ft.Icon(
            ft.Icons.CHEVRON_LEFT,
            size=16,
            color=ft.Colors.GREY_500,
        )

        # Outer container - width 16px to fit icon, height 40px
        btn_container = ft.Container(
            content=icon_widget,
            width=16,
            height=40,
            bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.GREY_400),
            alignment=ft.alignment.Alignment(0, 0),
            border_radius=4,
            tooltip=t("file_browser.tooltip"),
        )

        def on_click(e):
            print("[ToggleButton] Clicked!")
            self.toggle_file_browser()

        def on_hover(e):
            if e.data == "true":
                btn_container.bgcolor = ft.Colors.with_opacity(0.15, ft.Colors.GREY_400)
                icon_widget.color = ft.Colors.GREY_300
            else:
                btn_container.bgcolor = ft.Colors.with_opacity(0.05, ft.Colors.GREY_400)
                icon_widget.color = ft.Colors.GREY_500

        # Use Container's on_click for faster response
        btn_container.on_click = on_click
        btn_container.on_hover = on_hover

        # Store reference to icon widget for updates
        btn_container._icon_widget = icon_widget

        return btn_container

    async def _open_file_browser_delayed(self):
        """Open file browser after layout is mounted to page."""
        # Small delay to ensure layout is mounted
        await asyncio.sleep(0.1)
        self.toggle_file_browser()

    def toggle_file_browser(self):
        """Toggle file browser drawer visibility."""
        if self.is_file_drawer_visible:
            self._hide_file_browser()
        else:
            self._show_file_browser()

    def _show_file_browser(self):
        """Show file browser drawer."""
        start_time = time.time()
        print(f"[FileBrowser] Opening... (time: {start_time:.3f})")

        # Create file drawer if first time
        if self.file_drawer is None:
            t1 = time.time()
            self.file_drawer = FileBrowserDrawer(
                workspace_path=self.workspace_path,
                width=280,
                on_file_select=self._on_file_selected,
                on_close=self._hide_file_browser,
                visible=False,
            )
            t2 = time.time()
            print(f"[FileBrowser] Created drawer: {(t2 - t1) * 1000:.1f}ms")

            self.content_row.controls.append(self.file_drawer)
            t3 = time.time()
            print(f"[FileBrowser] Added to layout: {(t3 - t2) * 1000:.1f}ms")

            # Update page first to add control
            self.page.update()
            t4 = time.time()
            print(f"[FileBrowser] Page updated: {(t4 - t3) * 1000:.1f}ms")

            # Then load files (now control has page)
            self.file_drawer.refresh()
            t5 = time.time()
            print(f"[FileBrowser] Files loaded: {(t5 - t4) * 1000:.1f}ms")

        # Show it
        t6 = time.time()
        if self.file_drawer:
            self.file_drawer.visible = True
        self.is_file_drawer_visible = True

        # Move toggle button to left of file browser (280px + 16px button width from right edge)
        self.toggle_container.right = 296

        # Update icon to chevron right (open state, hint to close)
        if self.toggle_btn and hasattr(self.toggle_btn, "_icon_widget"):
            self.toggle_btn._icon_widget.name = ft.Icons.CHEVRON_RIGHT
            self.toggle_btn._icon_widget.color = ft.Colors.GREY_300
            self.toggle_btn.bgcolor = ft.Colors.with_opacity(0.15, ft.Colors.GREY_400)
            self.toggle_btn._icon_widget.update()

        self.page.update()
        end_time = time.time()
        print(f"[FileBrowser] Opened! Total: {(end_time - start_time) * 1000:.1f}ms")

    def _hide_file_browser(self):
        """Hide file browser drawer."""
        start_time = time.time()
        print(f"[FileBrowser] Closing... (time: {start_time:.3f})")

        if self.file_drawer:
            t1 = time.time()
            self.file_drawer.visible = False
            t2 = time.time()
            print(f"[FileBrowser] Set invisible: {(t2 - t1) * 1000:.1f}ms")

        self.is_file_drawer_visible = False

        # Move toggle button back to right edge
        self.toggle_container.right = 0

        # Update icon to chevron left (closed state, hint to open)
        t3 = time.time()
        if self.toggle_btn and hasattr(self.toggle_btn, "_icon_widget"):
            self.toggle_btn._icon_widget.name = ft.Icons.CHEVRON_LEFT
            self.toggle_btn._icon_widget.color = ft.Colors.GREY_500
            self.toggle_btn.bgcolor = ft.Colors.with_opacity(0.05, ft.Colors.GREY_400)
            self.toggle_btn._icon_widget.update()
        t4 = time.time()
        print(f"[FileBrowser] Icon updated: {(t4 - t3) * 1000:.1f}ms")

        self.page.update()
        end_time = time.time()
        print(f"[FileBrowser] Closed! Total: {(end_time - start_time) * 1000:.1f}ms")

    def _on_file_selected(self, path: Path):
        """Handle file selection from file browser."""
        print(f"[MainLayout] File selected: {path}")

    def get_layout(self) -> ft.Control:
        """Get the layout control for adding to page."""
        return self.row
