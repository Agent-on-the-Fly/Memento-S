"""
Workspace file browser widget with tree view and context menu
Optimized with virtual scrolling - fixed control pool, no memory leaks
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, List, Set, Dict
import os
import threading
import flet as ft
from gui.i18n import t


# Virtual scrolling constants
ITEM_HEIGHT = 22
VIEWPORT_ITEMS = 20
BUFFER_ITEMS = 3
MAX_CONTROLS = VIEWPORT_ITEMS + BUFFER_ITEMS * 2  # 26 controls max


@dataclass
class FlatItem:
    """Flattened tree item for virtual scrolling."""

    path: Path
    level: int
    is_directory: bool
    is_expanded: bool
    has_children: bool
    parent_indices: List[int] = field(default_factory=list)


class FileTreeItem(ft.Container):
    """Reusable file tree item control - gets bound to different data."""

    def __init__(self, file_browser):
        super().__init__()
        self.file_browser = file_browser
        self.item_index = -1  # Current index in flat_items
        self.flat_item: Optional[FlatItem] = None

        # Create UI components once
        self.expand_btn = ft.IconButton(
            icon=ft.Icons.ARROW_RIGHT,
            icon_size=16,
            icon_color=ft.Colors.GREY_400,
            width=16,
            height=22,
            padding=0,
            on_click=self._on_expand_click,
        )
        self.icon_widget = ft.Icon(ft.Icons.FOLDER_OUTLINED, size=16)
        self.name_text = ft.Text(
            size=13,
            color=ft.Colors.WHITE,
            no_wrap=True,
            overflow=ft.TextOverflow.ELLIPSIS,
        )

        # Spacer container for files (no expand button)
        self.spacer = ft.Container(width=16)

        # Item row with indentation
        self.item_row = ft.Row(
            spacing=2,
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        # Container styling
        self.padding = ft.padding.only(left=8, right=8, top=0, bottom=0)
        self.height = ITEM_HEIGHT
        self.bgcolor = None
        self.border = None
        self.border_radius = 4

        # Gesture detector wrapper
        self.content = ft.GestureDetector(
            content=self.item_row,
            on_tap=self._on_row_click,
            on_double_tap=self._on_double_click,
            on_secondary_tap_down=self._on_right_click,
        )

    def bind_data(self, flat_item: FlatItem, index: int):
        """Bind this reusable control to new data."""
        self.flat_item = flat_item
        self.item_index = index
        self.visible = True

        # Update icon based on file type
        if flat_item.is_directory:
            self.icon_widget.name = ft.Icons.FOLDER_OUTLINED
            self.icon_widget.color = ft.Colors.AMBER_400
        else:
            self.icon_widget.name = self._get_file_icon(flat_item.path)
            self.icon_widget.color = ft.Colors.BLUE_400

        # Update expand button
        if flat_item.is_directory:
            self.expand_btn.visible = True
            if flat_item.is_expanded:
                self.expand_btn.icon = ft.Icons.ARROW_DROP_DOWN
            else:
                self.expand_btn.icon = ft.Icons.ARROW_RIGHT
        else:
            self.expand_btn.visible = False

        # Update text
        self.name_text.value = flat_item.path.name
        self.name_text.tooltip = str(flat_item.path)

        # Build row with indentation
        indent = ft.Container(width=flat_item.level * 8)
        expand_widget = self.expand_btn if flat_item.is_directory else self.spacer

        self.item_row.controls = [
            indent,
            expand_widget,
            self.icon_widget,
            self.name_text,
        ]

        # Update selection state
        if self.file_browser.selected_index == index:
            self.bgcolor = ft.Colors.with_opacity(0.25, ft.Colors.BLUE_400)
        else:
            self.bgcolor = None

    def clear(self):
        """Clear this control (hide it)."""
        self.visible = False
        self.flat_item = None
        self.item_index = -1

    def _get_file_icon(self, path: Path):
        """Get appropriate icon for file type."""
        suffix = path.suffix.lower()
        icon_map = {
            ".py": ft.Icons.CODE,
            ".js": ft.Icons.JAVASCRIPT,
            ".ts": ft.Icons.JAVASCRIPT,
            ".json": ft.Icons.DATA_OBJECT,
            ".yaml": ft.Icons.DESCRIPTION,
            ".yml": ft.Icons.DESCRIPTION,
            ".md": ft.Icons.DESCRIPTION,
            ".txt": ft.Icons.DESCRIPTION,
            ".pdf": ft.Icons.PICTURE_AS_PDF,
            ".png": ft.Icons.IMAGE,
            ".jpg": ft.Icons.IMAGE,
            ".jpeg": ft.Icons.IMAGE,
            ".gif": ft.Icons.IMAGE,
        }
        return icon_map.get(suffix, ft.Icons.INSERT_DRIVE_FILE)

    def _on_expand_click(self, e):
        """Handle expand button click."""
        if self.flat_item and self.flat_item.is_directory:
            self.file_browser._toggle_expand(self.item_index)

    def _on_row_click(self, e):
        """Handle row click."""
        if self.flat_item:
            self.file_browser._select_index(self.item_index)
            if self.flat_item.is_directory:
                self.file_browser._toggle_expand(self.item_index)

    def _on_double_click(self, e):
        """Handle double click event."""
        if self.flat_item:
            if self.flat_item.is_directory:
                self.file_browser._toggle_expand(self.item_index)
            else:
                self.file_browser._open_in_explorer(self.flat_item.path)

    def _on_right_click(self, e):
        """Handle right click - show context menu."""
        if self.flat_item:
            self.file_browser._show_context_menu(self.flat_item.path, e)


class FileBrowserDrawer(ft.Container):
    """File browser drawer with virtual scrolling and fixed control pool."""

    def __init__(
        self,
        workspace_path: Path,
        width: int = 280,
        on_file_select: Optional[Callable[[Path], None]] = None,
        on_close: Optional[Callable] = None,
        visible: bool = True,
    ):
        super().__init__()
        self.workspace_path = Path(workspace_path)
        self.on_file_select = on_file_select
        self.on_close = on_close

        # Selection state
        self.selected_index = -1

        # Virtual scrolling data
        self.flat_items: List[FlatItem] = []  # All flattened items
        self.visible_indices: List[int] = []  # Indices currently visible
        self.scroll_offset = 0  # Current scroll position
        self._expanded_paths: Set[str] = set()  # Set of expanded directory paths

        # Search state
        self._is_searching = False
        self._search_query = ""
        self._search_timer = None
        self._search_delay = 0.05
        self._file_index: Dict[str, List[Path]] = {}
        self._index_lock = threading.Lock()
        self._search_cache: Dict[str, List[Path]] = {}
        self._cache_keys: List[str] = []
        self._cache_max_size = 10

        # Fixed control pool - NEVER grows
        self._control_pool: List[FileTreeItem] = []

        # Search icons
        self._search_icon = ft.Icons.SEARCH
        self._loading_icon = ft.Container(
            content=ft.ProgressRing(
                width=8,
                height=8,
                stroke_width=2,
                color=ft.Colors.GREY_400,
            ),
            width=16,
            height=16,
            alignment=ft.alignment.Alignment(0, 0),
        )

        # Build UI
        self._build_ui()

        # Set width and visibility
        self.width = width
        self.visible = visible

    def _build_ui(self):
        """Build the drawer UI."""
        # Search box at top
        self.search_box = ft.TextField(
            hint_text=t("file_browser.search_hint"),
            prefix_icon=ft.Icons.SEARCH,
            border_radius=20,
            height=36,
            text_size=13,
            on_change=self._on_search,
            content_padding=ft.padding.only(left=6, right=12, top=8, bottom=8),
        )

        # Create fixed control pool - only once!
        for i in range(MAX_CONTROLS):
            control = FileTreeItem(self)
            control.visible = False
            self._control_pool.append(control)

        # List view for virtual scrolling
        self.list_view = ft.ListView(
            controls=self._control_pool,
            spacing=0,
            expand=True,
            on_scroll=self._on_scroll,
        )

        # Main content
        content = ft.Column(
            [
                ft.Container(
                    content=self.search_box,
                    padding=ft.padding.symmetric(horizontal=12, vertical=8),
                    border=ft.border.only(bottom=ft.BorderSide(1, ft.Colors.GREY_800)),
                    height=52,
                ),
                ft.Container(
                    content=self.list_view,
                    expand=True,
                    padding=ft.padding.only(left=4, right=4, bottom=8, top=8),
                ),
            ],
            expand=True,
            spacing=0,
        )

        # Set Container properties
        self.bgcolor = ft.Colors.GREY_900
        self.content = content
        self.border = ft.border.only(left=ft.BorderSide(1, ft.Colors.GREY_800))

    def refresh(self):
        """Refresh the file tree."""
        try:
            _ = self.page
        except RuntimeError:
            return

        if not self.workspace_path.exists():
            self.flat_items = []
            self.visible_indices = []
            self._render_viewport()
            return

        # Clear and rebuild flat items
        self.flat_items = []
        self._expanded_paths.clear()
        self.selected_index = -1

        # Load workspace contents
        try:
            items = []
            for item in self.workspace_path.iterdir():
                if item.name.startswith("."):
                    continue
                items.append(item)
                if len(items) >= 100:
                    break

            items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))

            # Build flat items
            for item in items:
                self._add_item_recursive(item, 0)

        except PermissionError:
            pass

        # Render viewport
        self._render_viewport()

    def _add_item_recursive(
        self, path: Path, level: int, parent_indices: List[int] = None
    ):
        """Add item to flat_items, recursively if expanded."""
        if parent_indices is None:
            parent_indices = []

        is_dir = path.is_dir()
        has_children = False

        if is_dir:
            try:
                has_children = any(
                    not child.name.startswith(".") for child in path.iterdir()
                )
            except PermissionError:
                pass

        current_index = len(self.flat_items)
        flat_item = FlatItem(
            path=path,
            level=level,
            is_directory=is_dir,
            is_expanded=False,
            has_children=has_children,
            parent_indices=parent_indices[:],
        )
        self.flat_items.append(flat_item)

        # If directory has children and is "expanded" (during search), add them
        if is_dir and has_children and str(path) in self._expanded_paths:
            flat_item.is_expanded = True
            new_parent_indices = parent_indices + [current_index]
            try:
                children = sorted(
                    [c for c in path.iterdir() if not c.name.startswith(".")],
                    key=lambda x: (not x.is_dir(), x.name.lower()),
                )
                for child in children:
                    self._add_item_recursive(child, level + 1, new_parent_indices)
            except PermissionError:
                pass

    def _render_viewport(self):
        """Render only the visible viewport items."""
        # Calculate visible range
        start_idx = max(0, self.scroll_offset - BUFFER_ITEMS)
        end_idx = min(len(self.flat_items), start_idx + MAX_CONTROLS)

        # Update visible indices
        self.visible_indices = list(range(start_idx, end_idx))

        # Bind controls to data
        for i, control in enumerate(self._control_pool):
            data_idx = start_idx + i
            if data_idx < len(self.flat_items):
                control.bind_data(self.flat_items[data_idx], data_idx)
            else:
                control.clear()

        # Update list view only (not entire container)
        if self.page:
            self.list_view.update()

    def _on_scroll(self, e):
        """Handle scroll events."""
        # Calculate scroll offset from pixel position
        if hasattr(e, "pixels"):
            self.scroll_offset = int(e.pixels / ITEM_HEIGHT)
        self._render_viewport()

    def _toggle_expand(self, index: int):
        """Toggle expand/collapse for a directory."""
        if index < 0 or index >= len(self.flat_items):
            return

        item = self.flat_items[index]
        if not item.is_directory or not item.has_children:
            return

        path_str = str(item.path)

        if item.is_expanded:
            # Collapse: remove all children from flat_items
            item.is_expanded = False
            self._expanded_paths.discard(path_str)

            # Find and remove all descendant items
            indices_to_remove = []
            for i in range(index + 1, len(self.flat_items)):
                if path_str in [
                    str(self.flat_items[j].path)
                    for j in self.flat_items[i].parent_indices
                ]:
                    indices_to_remove.append(i)
                else:
                    break

            # Remove in reverse order to maintain indices
            for i in reversed(indices_to_remove):
                del self.flat_items[i]

            # Adjust selected_index if needed
            if self.selected_index in indices_to_remove:
                self.selected_index = index
            elif self.selected_index > index:
                self.selected_index -= len(indices_to_remove)
        else:
            # Expand: add children to flat_items
            item.is_expanded = True
            self._expanded_paths.add(path_str)

            # Load children
            try:
                children = sorted(
                    [c for c in item.path.iterdir() if not c.name.startswith(".")],
                    key=lambda x: (not x.is_dir(), x.name.lower()),
                )

                # Insert children after parent
                insert_idx = index + 1
                new_parent_indices = item.parent_indices + [index]

                for child in children:
                    child_is_dir = child.is_dir()
                    child_has_children = False

                    if child_is_dir:
                        try:
                            child_has_children = any(
                                not grandchild.name.startswith(".")
                                for grandchild in child.iterdir()
                            )
                        except PermissionError:
                            pass

                    child_item = FlatItem(
                        path=child,
                        level=item.level + 1,
                        is_directory=child_is_dir,
                        is_expanded=False,
                        has_children=child_has_children,
                        parent_indices=new_parent_indices[:],
                    )
                    self.flat_items.insert(insert_idx, child_item)
                    insert_idx += 1

            except PermissionError:
                pass

        self._render_viewport()

    def _select_index(self, index: int):
        """Select an item by index."""
        if index < 0 or index >= len(self.flat_items):
            return

        # Deselect previous
        prev_index = self.selected_index
        self.selected_index = index

        # Update UI for both old and new selection
        if prev_index >= 0:
            self._update_item_selection(prev_index)
        self._update_item_selection(index)

        # Callback for files
        item = self.flat_items[index]
        if not item.is_directory and self.on_file_select:
            self.on_file_select(item.path)

    def _update_item_selection(self, index: int):
        """Update selection visual state for an item."""
        # Find control showing this index
        for control in self._control_pool:
            if control.item_index == index:
                if index == self.selected_index:
                    control.bgcolor = ft.Colors.with_opacity(0.25, ft.Colors.BLUE_400)
                else:
                    control.bgcolor = None
                if control.page:
                    control.update()
                break

    def _on_search(self, e):
        """Handle search input."""
        query = e.control.value.lower().strip()

        # Cancel previous timer
        if self._search_timer:
            self._search_timer.cancel()
            self._search_timer = None

        if not query:
            # Clear search
            self._is_searching = False
            self._search_query = ""
            self.search_box.prefix_icon = self._search_icon
            if self.page:
                self.search_box.update()

            # Restore full tree
            def clear_and_refresh():
                with self._index_lock:
                    self._file_index.clear()
                self.refresh()

            clear_and_refresh()
            return

        # Show loading icon
        self.search_box.prefix_icon = self._loading_icon
        if self.page:
            self.search_box.update()

        # Debounce search
        def do_search():
            try:
                self._filter_items(query)
            except Exception as ex:
                print(f"[do_search] ERROR: {ex}")
            finally:
                self._search_timer = None
                self.search_box.prefix_icon = self._search_icon
                if self.page:
                    self.search_box.update()

        self._search_timer = threading.Timer(self._search_delay, do_search)
        self._search_timer.start()

    def _filter_items(self, query: str):
        """Filter items using inverted index."""
        query = query.lower()
        self._search_query = query

        # Check cache
        if query in self._search_cache:
            matched_files = self._search_cache[query]
            self._cache_keys.remove(query)
            self._cache_keys.append(query)
        else:
            # Build index if needed
            index_ready = False
            with self._index_lock:
                index_ready = bool(self._file_index)

            if not index_ready:
                self._build_file_index()

            # Search using index
            index_data = {}
            with self._index_lock:
                index_data = dict(self._file_index)

            if not index_data:
                return

            matched_items = set()
            for keyword, paths in index_data.items():
                if query in keyword:
                    matched_items.update(paths)
                    if len(matched_items) >= 20:
                        break

            # Keep both files and directories (total 20)
            matched_files = list(matched_items)[:20]

            # Cache result
            self._search_cache[query] = matched_files
            self._cache_keys.append(query)

            # LRU eviction
            if len(self._cache_keys) > self._cache_max_size:
                oldest_key = self._cache_keys.pop(0)
                del self._search_cache[oldest_key]

        if not matched_files:
            self.flat_items = []
            self.visible_indices = []
            self._render_viewport()
            return

        self._is_searching = True

        # Show matched items directly (flat list)
        self.flat_items = []
        self.selected_index = -1

        # Sort: directories first, then by name
        sorted_paths = sorted(
            matched_files,
            key=lambda p: (not p.is_dir(), p.name.lower()),
        )

        for path in sorted_paths:
            is_dir = path.is_dir()
            has_children = False

            if is_dir:
                try:
                    has_children = any(
                        not child.name.startswith(".") for child in path.iterdir()
                    )
                except PermissionError:
                    pass

            flat_item = FlatItem(
                path=path,
                level=0,  # All at same level in search results
                is_directory=is_dir,
                is_expanded=False,  # Directories are collapsed by default
                has_children=has_children,
                parent_indices=[],
            )
            self.flat_items.append(flat_item)

        self._render_viewport()

    def _build_file_index(self):
        """Build inverted index for fast search."""
        index = {}
        try:
            for path in self.workspace_path.rglob("*"):
                if path.name.startswith("."):
                    continue

                # Security check
                try:
                    path.relative_to(self.workspace_path)
                except ValueError:
                    continue

                is_dir = path.is_dir()
                name_lower = path.name.lower()

                # Build keywords
                keywords = set()
                keywords.add(name_lower)

                # For files, also add stem (filename without extension)
                if not is_dir:
                    keywords.add(path.stem.lower())
                    for part in (
                        path.stem.lower().replace("_", " ").replace("-", " ").split()
                    ):
                        if len(part) >= 2:
                            keywords.add(part)

                for keyword in keywords:
                    if keyword not in index:
                        index[keyword] = []
                    index[keyword].append(path)

        except PermissionError:
            pass

        with self._index_lock:
            self._file_index = index

    def _close_context_menu(self):
        """Close the context menu."""
        if not self.page:
            return

        menu_stack = getattr(self, "_context_menu_stack", None)
        if menu_stack and menu_stack in self.page.overlay:
            try:
                self.page.overlay.remove(menu_stack)
                self._context_menu_stack = None
                self.page.update()
            except Exception:
                self._context_menu_stack = None

    def _show_context_menu(self, path: Path, e):
        """Show context menu for file/directory."""
        if not self.page:
            return

        self._close_context_menu()
        self._context_menu_stack = None

        def close_menu():
            self._close_context_menu()

        def on_menu_item_click(handler):
            def wrapper(_):
                handler()
                close_menu()

            return wrapper

        menu_column = ft.Column(
            [
                ft.Container(
                    content=ft.TextButton(
                        content=ft.Row(
                            [
                                ft.Icon(
                                    ft.Icons.CONTENT_COPY,
                                    size=16,
                                    color=ft.Colors.WHITE,
                                ),
                                ft.Text(
                                    t("file_browser.menu.copy_path"),
                                    color=ft.Colors.WHITE,
                                ),
                            ],
                            spacing=8,
                        ),
                        on_click=on_menu_item_click(lambda: self._copy_path(path)),
                        style=ft.ButtonStyle(
                            padding=ft.padding.symmetric(horizontal=12, vertical=8),
                        ),
                    ),
                    expand=True,
                ),
                ft.Container(
                    content=ft.TextButton(
                        content=ft.Row(
                            [
                                ft.Icon(
                                    ft.Icons.FOLDER_OPEN, size=16, color=ft.Colors.WHITE
                                ),
                                ft.Text(
                                    t("file_browser.menu.open_in_folder"),
                                    color=ft.Colors.WHITE,
                                ),
                            ],
                            spacing=8,
                        ),
                        on_click=on_menu_item_click(
                            lambda: self._open_in_explorer(path)
                        ),
                        style=ft.ButtonStyle(
                            padding=ft.padding.symmetric(horizontal=12, vertical=8),
                        ),
                    ),
                    expand=True,
                ),
                ft.Container(
                    content=ft.TextButton(
                        content=ft.Row(
                            [
                                ft.Icon(
                                    ft.Icons.DELETE, size=16, color=ft.Colors.RED_400
                                ),
                                ft.Text(
                                    t("file_browser.menu.delete"),
                                    color=ft.Colors.RED_400,
                                ),
                            ],
                            spacing=8,
                        ),
                        on_click=on_menu_item_click(lambda: self._delete_file(path)),
                        style=ft.ButtonStyle(
                            padding=ft.padding.symmetric(horizontal=12, vertical=8),
                        ),
                    ),
                    expand=True,
                ),
            ],
            spacing=0,
            tight=True,
        )

        x = e.global_position.x if e.global_position else 0
        y = e.global_position.y if e.global_position else 0

        menu_container = ft.Container(
            content=menu_column,
            left=x,
            top=y,
            width=160,
            bgcolor=ft.Colors.GREY_800,
            border_radius=4,
            padding=ft.padding.symmetric(horizontal=4, vertical=4),
            border=ft.border.all(1, ft.Colors.GREY_700),
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=4,
                color=ft.Colors.BLACK54,
            ),
            animate_opacity=300,
        )

        backdrop = ft.GestureDetector(
            content=ft.Container(
                expand=True,
                bgcolor=ft.Colors.TRANSPARENT,
            ),
            on_tap=lambda _: close_menu(),
            on_secondary_tap_down=lambda _: close_menu(),
        )

        self._context_menu_stack = ft.Stack(
            [
                backdrop,
                menu_container,
            ],
            expand=True,
        )

        if self.page:
            self.page.overlay.append(self._context_menu_stack)
            self.page.update()

    def _copy_path(self, path: Path):
        """Copy path to clipboard."""
        if self.page:
            import subprocess
            import platform

            path_str = str(path)
            try:
                if platform.system() == "Darwin":
                    subprocess.run(["pbcopy"], input=path_str.encode(), check=True)
                elif platform.system() == "Windows":
                    subprocess.run(["clip"], input=path_str.encode(), check=True)
                else:
                    subprocess.run(
                        ["xclip", "-selection", "clipboard"],
                        input=path_str.encode(),
                        check=True,
                    )
            except Exception as e:
                print(f"[FileBrowser] Failed to copy: {e}")

    def _open_in_explorer(self, path: Path):
        """Open file/directory's parent folder in system explorer."""
        import platform
        import subprocess

        try:
            if platform.system() == "Windows":
                subprocess.run(["explorer", f"/select,{str(path)}"])
            elif platform.system() == "Darwin":
                subprocess.run(["open", "-R", str(path)])
            else:
                file_managers = [
                    ["nautilus", "--select", str(path)],
                    ["nemo", str(path)],
                    ["dolphin", "--select", str(path)],
                    ["thunar", str(path.parent)],
                    ["xdg-open", str(path.parent)],
                ]

                for fm in file_managers:
                    try:
                        subprocess.run(fm, check=True)
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
        except Exception as e:
            print(f"[FileBrowser] Failed to open: {e}")

    def _delete_file(self, path: Path):
        """Delete file with confirmation."""

        def confirm_delete(e):
            try:
                import shutil

                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)

                dialog.open = False
                if self.page:
                    self.page.update()

                self.refresh()
            except Exception as e:
                print(f"[FileBrowser] Failed to delete: {e}")
                dialog.open = False
                if self.page:
                    self.page.update()

        def cancel_delete(e):
            dialog.open = False
            if self.page:
                self.page.update()

        dialog = ft.AlertDialog(
            title=ft.Text(t("file_browser.dialog.delete_title")),
            content=ft.Text(t("file_browser.dialog.delete_content", name=path.name)),
            actions=[
                ft.TextButton(t("file_browser.dialog.cancel"), on_click=cancel_delete),
                ft.TextButton(
                    t("file_browser.dialog.confirm_delete"),
                    on_click=confirm_delete,
                    style=ft.ButtonStyle(color=ft.Colors.RED_400),
                ),
            ],
        )

        if self.page:
            self.page.overlay.append(dialog)
            dialog.open = True
            self.page.update()

    def _on_close(self, e):
        """Handle close button click."""
        if self.on_close:
            self.on_close()
