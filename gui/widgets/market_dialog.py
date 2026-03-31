"""Markket Dialog - Skill market with install/uninstall functionality."""

from __future__ import annotations

import asyncio
from typing import Optional

import flet as ft

from core.skill.market import SkillMarket
from core.skill.retrieval.schema import RecallCandidate
from core.skill.schema import ExecutionMode, SkillManifest
from core.utils.text import to_kebab_case
from gui.i18n import t
from utils.logger import get_logger

logger = get_logger(__name__)


class MarkketDialog:
    """Skill market dialog with search, list, and detail view."""

    def __init__(self, app):
        self.app = app
        self.dialog: Optional[ft.AlertDialog] = None
        self.skill_market: Optional[SkillMarket] = None
        self.all_skills: list[SkillManifest] = []
        self.installed_skills: set[str] = set()
        self.selected_skill: Optional[SkillManifest] = None
        self.current_filter_text: str = ""
        self.current_displayed_skills: list[
            SkillManifest
        ] = []  # Track currently displayed skills

        # UI Components
        self.search_field: Optional[ft.TextField] = None
        self.skills_list: Optional[ft.ListView] = None
        self.detail_name: Optional[ft.Text] = None
        self.detail_description: Optional[ft.Text] = None
        self.detail_params: Optional[ft.Column] = None
        self.detail_dependencies: Optional[ft.Column] = None
        self.action_button_container: Optional[ft.Container] = None
        self.status_text: Optional[ft.Text] = None
        self.local_import_button: Optional[ft.ElevatedButton] = None
        self.online_import_button: Optional[ft.ElevatedButton] = None

    async def _init_skill_market(self):
        """Initialize skill market instance."""
        try:
            self.skill_market = await SkillMarket.from_config()
            await self._load_skills()
        except Exception as e:
            logger.error(f"Failed to initialize skill market: {e}")
            self._show_error("Failed to initialize skill market")

    async def _load_skills(self):
        """Load all skills from remote catalog and local store."""
        if not self.skill_market:
            return

        try:
            # Load installed skills
            local_skills = await self.skill_market.list_skills()
            self.installed_skills = {s.name for s in local_skills}

            # Search remote catalog (empty query to get all)
            remote_skills = await self.skill_market.search("", k=50)

            # Merge and deduplicate
            skill_map: dict[str, SkillManifest] = {}

            # Add remote skills first
            for skill in remote_skills:
                if isinstance(skill, dict):
                    skill = SkillManifest(**skill)
                skill_map[skill.name] = skill

            # Add local skills (override or add)
            for skill in local_skills:
                skill_map[skill.name] = skill

            self.all_skills = list(skill_map.values())
            self.all_skills.sort(key=lambda s: s.name)

            # Default select first skill
            if self.all_skills:
                self.selected_skill = self.all_skills[0]

            self._update_skills_list()

            # Update detail panel for selected skill
            if self.selected_skill:
                self._update_detail_panel()

            self._update_status(f"Loaded {len(self.all_skills)} skills")

        except Exception as e:
            logger.error(f"Failed to load skills: {e}")
            self._show_error("Failed to load skills")

    def _update_skills_list(self, filter_text: str = ""):
        """Update the skills list with optional filter."""
        if not self.skills_list:
            return

        self.skills_list.controls.clear()

        filtered_skills = [
            s
            for s in self.all_skills
            if filter_text.lower() in s.name.lower()
            or filter_text.lower() in s.description.lower()
        ]

        # Save currently displayed skills
        self.current_displayed_skills = filtered_skills
        self._render_skills_list(filtered_skills)

    def _update_skills_list_with_data(self, skills: list[SkillManifest]):
        """Update the skills list with provided data (search results)."""
        if not self.skills_list:
            return

        self.skills_list.controls.clear()

        # Save currently displayed skills
        self.current_displayed_skills = skills
        self._render_skills_list(skills)

    def _render_skills_list(self, skills: list[SkillManifest]):
        """Render skills list items."""
        for skill in skills:
            is_installed = skill.name in self.installed_skills
            is_selected = self.selected_skill and skill.name == self.selected_skill.name

            # Create list item
            item = ft.Container(
                content=ft.Row(
                    [
                        ft.Icon(
                            ft.icons.Icons.BOOKMARK
                            if is_installed
                            else ft.icons.Icons.BOOKMARK_BORDER,
                            color=ft.Colors.BLUE_400
                            if is_installed
                            else ft.Colors.GREY_500,
                            size=20,
                        ),
                        ft.Column(
                            [
                                ft.Text(
                                    to_kebab_case(skill.name),
                                    size=14,
                                    weight=ft.FontWeight.W_600
                                    if is_selected
                                    else ft.FontWeight.W_500,
                                    color=ft.Colors.WHITE,
                                ),
                                ft.Text(
                                    skill.description[:50] + "..."
                                    if len(skill.description) > 50
                                    else skill.description,
                                    size=11,
                                    color=ft.Colors.GREY_300
                                    if is_selected
                                    else ft.Colors.GREY_500,
                                    max_lines=1,
                                    overflow=ft.TextOverflow.ELLIPSIS,
                                ),
                            ],
                            spacing=2,
                            expand=True,
                        ),
                        ft.Text(
                            "已安装" if is_installed else "",
                            size=10,
                            color=ft.Colors.GREEN_400,
                        )
                        if is_installed
                        else ft.Container(),
                    ],
                    spacing=12,
                ),
                padding=ft.Padding(12, 10, 12, 10),
                border_radius=ft.BorderRadius.all(6),
                bgcolor=ft.Colors.BLUE_900 if is_selected else None,
                data=skill,
                on_click=self._on_skill_select,
            )

            # Add hover effect (only for non-selected items)
            if not is_selected:
                item.on_hover = lambda e, container=item: self._on_item_hover(
                    e, container
                )

            self.skills_list.controls.append(item)

        if self.skills_list.page:
            self.skills_list.update()

    def _on_item_hover(self, e, container: ft.Container):
        """Handle list item hover effect."""
        if e.data == "true":
            container.bgcolor = ft.Colors.GREY_800
        else:
            container.bgcolor = None
        container.update()

    def _on_skill_select(self, e: ft.ControlEvent):
        """Handle skill selection."""
        container = e.control
        skill: SkillManifest = container.data

        self.selected_skill = skill

        # Update list to show selection highlight using current displayed skills
        if self.skills_list:
            self.skills_list.controls.clear()
            self._render_skills_list(self.current_displayed_skills)

        # Update detail panel
        self._update_detail_panel()

    def _update_detail_panel(self):
        """Update the right panel with selected skill details."""
        if not self.selected_skill:
            return

        skill = self.selected_skill
        is_installed = skill.name in self.installed_skills

        # Update name and description
        self.detail_name.value = to_kebab_case(skill.name)
        self.detail_description.value = skill.description or "No description available"

        # Update parameters
        self.detail_params.controls.clear()
        if skill.parameters and skill.parameters.get("properties"):
            for param_name, param_info in skill.parameters["properties"].items():
                param_type = param_info.get("type", "unknown")
                param_desc = param_info.get("description", "")
                self.detail_params.controls.append(
                    ft.Row(
                        [
                            ft.Text(
                                f"• {param_name}:",
                                size=12,
                                color=ft.Colors.WHITE,
                                weight=ft.FontWeight.W_500,
                            ),
                            ft.Text(
                                f" {param_type}", size=12, color=ft.Colors.GREY_400
                            ),
                            ft.Text(
                                f" - {param_desc}" if param_desc else "",
                                size=12,
                                color=ft.Colors.GREY_500,
                            ),
                        ],
                        spacing=4,
                    )
                )
        else:
            self.detail_params.controls.append(
                ft.Text("No parameters required", size=12, color=ft.Colors.GREY_500)
            )

        # Update dependencies
        self.detail_dependencies.controls.clear()
        if skill.dependencies:
            for dep in skill.dependencies:
                self.detail_dependencies.controls.append(
                    ft.Text(f"• {dep}", size=12, color=ft.Colors.GREY_400)
                )
        else:
            self.detail_dependencies.controls.append(
                ft.Text("No dependencies", size=12, color=ft.Colors.GREY_500)
            )

        # Update action button
        if is_installed:
            self.action_button_container.content = ft.ElevatedButton(
                "🗑️ 卸载",
                on_click=self._on_uninstall_click,
                style=ft.ButtonStyle(
                    color=ft.Colors.WHITE,
                    bgcolor=ft.Colors.RED_700,
                ),
            )
        else:
            self.action_button_container.content = ft.ElevatedButton(
                "⬇️ 安装",
                on_click=self._on_install_click,
                style=ft.ButtonStyle(
                    color=ft.Colors.WHITE,
                    bgcolor=ft.Colors.BLUE_700,
                ),
            )

        # Update UI
        if self.detail_name.page:
            self.detail_name.update()
            self.detail_description.update()
            self.detail_params.update()
            self.detail_dependencies.update()
            self.action_button_container.update()

    async def _on_install_click(self, e: ft.ControlEvent):
        """Handle install button click."""
        if not self.selected_skill or not self.skill_market:
            return

        skill_name = self.selected_skill.name
        self._update_status(f"Installing {skill_name}...")

        # Show loading state
        self.action_button_container.content = ft.ElevatedButton(
            "⏳ 安装中...",
            disabled=True,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.GREY_600,
            ),
        )
        self.action_button_container.update()
        await asyncio.sleep(0.1)  # Give UI time to render

        try:
            skill = await self.skill_market.install(skill_name)
            if skill:
                self._update_status(f"✓ {skill_name} installed successfully")

                # Reload all skills from SkillMarket to ensure list is up-to-date
                await self._load_skills()
            else:
                self._update_status(f"✗ Failed to install {skill_name}")
                # Restore button state
                self._update_detail_panel()
        except Exception as e:
            logger.error(f"Failed to install {skill_name}: {e}")
            self._update_status(f"✗ Error: {str(e)}")
            # Restore button state
            self._update_detail_panel()

    async def _on_uninstall_click(self, e: ft.ControlEvent):
        """Handle uninstall button click."""
        if not self.selected_skill or not self.skill_market:
            return

        skill_name = self.selected_skill.name
        self._update_status(f"Uninstalling {skill_name}...")

        # Show loading state
        self.action_button_container.content = ft.ElevatedButton(
            "⏳ 卸载中...",
            disabled=True,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.GREY_600,
            ),
        )
        self.action_button_container.update()
        await asyncio.sleep(0.1)  # Give UI time to render

        try:
            success = await self.skill_market.uninstall(skill_name)
            if success:
                self.installed_skills.discard(skill_name)
                self._update_status(f"✓ {skill_name} uninstalled successfully")

                # Refresh the list
                if self.skills_list:
                    self.skills_list.controls.clear()
                    self._render_skills_list(self.current_displayed_skills)

                self._update_detail_panel()
            else:
                self._update_status(f"✗ Failed to uninstall {skill_name}")
                # Restore button state
                self._update_detail_panel()
        except Exception as e:
            logger.error(f"Failed to uninstall {skill_name}: {e}")
            self._update_status(f"✗ Error: {str(e)}")
            # Restore button state
            self._update_detail_panel()

    def _on_search_change(self, e: ft.ControlEvent):
        """Handle search text change - search from remote catalog."""
        self.current_filter_text = e.control.value or ""

        # If search text is empty, show all skills
        if not self.current_filter_text.strip():
            self._update_skills_list("")
            return

        # Search from remote catalog
        asyncio.create_task(self._search_skills(self.current_filter_text))

    async def _search_skills(self, query: str):
        """Search skills from remote catalog and update UI."""
        if not self.skill_market:
            return

        try:
            self._update_status(f"Searching for '{query}'...")

            # Search from remote catalog
            search_results = await self.skill_market.search(query, k=20)
            # print(f"Raw search results for '{query}': {search_results}")

            # Convert to SkillManifest
            searched_skills = []
            for skill_data in search_results:
                if isinstance(skill_data, dict):
                    skill = SkillManifest(**skill_data)
                elif isinstance(skill_data, SkillManifest):
                    skill = skill_data
                elif isinstance(skill_data, RecallCandidate):
                    # Convert RecallCandidate to SkillManifest
                    skill = SkillManifest(
                        name=skill_data.name,
                        description=skill_data.description,
                        execution_mode=ExecutionMode.KNOWLEDGE,  # Default mode
                        parameters=None,
                        dependencies=[],
                    )
                else:
                    logger.debug(
                        f"[MarkketDialog] Unknown skill data type: {type(skill_data)}"
                    )
                    continue
                searched_skills.append(skill)

            print(f"Search results for '{query}': {[s.name for s in searched_skills]}")

            # Merge with local skills (to show installed status)
            local_skill_map = {s.name: s for s in self.all_skills}
            final_results = []

            for skill in searched_skills:
                if skill.name in local_skill_map:
                    # Use local skill data if available (to show installed status)
                    final_results.append(local_skill_map[skill.name])
                else:
                    final_results.append(skill)

            print(f"Search results for '{query}': {[s.name for s in searched_skills]}")

            # Update the skills list with search results
            self._update_skills_list_with_data(final_results)
            self._update_status(f"Found {len(final_results)} skills")

        except Exception as e:
            logger.error(f"Search failed: {e}")
            self._update_status(f"Search failed: {str(e)}")

    async def _on_import_click(self, e: ft.ControlEvent):
        """Handle import button click - open directory picker using osascript."""

        import subprocess

        try:
            # Use AppleScript to open directory picker on macOS
            script = 'choose folder with prompt "选择要导入的 Skill 目录"'
            logger.info("[Import] Opening directory picker with AppleScript")
            result = subprocess.run(
                ["osascript", "-e", script], capture_output=True, text=True, timeout=30
            )

            logger.info(f"[Import] AppleScript return code: {result.returncode}")
            logger.info(f"[Import] AppleScript stdout: {repr(result.stdout)}")
            logger.info(f"[Import] AppleScript stderr: {repr(result.stderr)}")

            if result.returncode == 0 and result.stdout.strip():
                # Parse result (format: alias Macintosh HD:Users:...)
                path_str = result.stdout.strip()
                logger.info(f"[Import] Raw path string: {path_str}")

                if path_str.startswith("alias "):
                    # Extract path from alias format (alias <volume>:<path>)
                    path_str = path_str[6:]  # Remove 'alias ' prefix
                    # Split by colon and reconstruct path
                    parts = path_str.split(":")
                    # Remove volume name (first part), join rest with /
                    if len(parts) > 1:
                        local_path = "/" + "/".join(parts[1:])
                    else:
                        local_path = "/" + parts[0]
                    # Remove trailing slash if present
                    local_path = local_path.rstrip("/")
                else:
                    local_path = path_str

                logger.info(f"[Import] Parsed local path: {local_path}")
                self._update_status(f"导入中: {local_path}...")

                # Show loading state on import button
                self._update_import_button_loading(True)
                await asyncio.sleep(0.1)  # Give UI time to render

                try:
                    logger.info(
                        f"[Import] Calling install_from_local with path: {local_path}"
                    )
                    skill = await self.skill_market.install_from_local(local_path)
                    if skill:
                        logger.info(
                            f"[Import] Successfully installed skill: {skill.name}"
                        )
                        self.installed_skills.add(skill.name)
                        self._update_status(f"✓ {skill.name} 导入成功")

                        # Set selected skill to the newly imported one
                        self.selected_skill = skill

                        # Refresh skills list (will auto-select the new skill)
                        await self._load_skills()
                    else:
                        logger.error(
                            f"[Import] install_from_local returned None for path: {local_path}"
                        )
                        self._update_status("✗ 导入失败")
                except Exception as ex:
                    logger.error(f"[Import] Import failed with exception: {ex}")
                    self._update_status(f"✗ 错误: {str(ex)}")
                finally:
                    self._update_import_button_loading(False)
            else:
                # User cancelled or error
                logger.info(
                    "[Import] User cancelled directory selection or AppleScript failed"
                )
                pass

        except subprocess.TimeoutExpired:
            logger.error("[Import] Directory picker timeout")
            self._update_status("✗ 选择目录超时")
        except Exception as ex:
            logger.error(f"[Import] Failed to open directory picker: {ex}")
            self._update_status(f"✗ 无法打开目录选择器: {str(ex)}")

    def _update_import_button_loading(self, is_loading: bool):
        """Update import button to show loading state or normal state."""
        if self.local_import_button:
            import_btn = self.local_import_button
            if is_loading:
                import_btn.text = "⏳ 导入中..."
                import_btn.disabled = True
                import_btn.style = ft.ButtonStyle(
                    color=ft.Colors.WHITE,
                    bgcolor=ft.Colors.GREY_600,
                    shape=ft.RoundedRectangleBorder(radius=6),
                )
            else:
                import_btn.text = "本地导入"
                import_btn.disabled = False
                import_btn.style = ft.ButtonStyle(
                    color=ft.Colors.WHITE,
                    bgcolor=ft.Colors.BLUE_700,
                    shape=ft.RoundedRectangleBorder(radius=6),
                )
            import_btn.update()

    def _update_online_import_button_loading(self, is_loading: bool):
        """Update online import button to show loading state or normal state."""
        if self.online_import_button:
            import_btn = self.online_import_button
            if is_loading:
                import_btn.text = "⏳ 导入中..."
                import_btn.disabled = True
                import_btn.style = ft.ButtonStyle(
                    color=ft.Colors.WHITE,
                    bgcolor=ft.Colors.GREY_600,
                    shape=ft.RoundedRectangleBorder(radius=6),
                )
            else:
                import_btn.text = "在线导入"
                import_btn.disabled = False
                import_btn.style = ft.ButtonStyle(
                    color=ft.Colors.WHITE,
                    bgcolor=ft.Colors.BLUE_700,
                    shape=ft.RoundedRectangleBorder(radius=6),
                )
            import_btn.update()

    async def _on_online_import_click(self, e: ft.ControlEvent):
        """Handle online import button click - show URL input dialog."""
        url_input = ft.TextField(
            label="Skill URL",
            hint_text="https://example.com/skill.zip",
            expand=True,
        )

        async def on_confirm(e: ft.ControlEvent):
            url = url_input.value.strip()
            if not url:
                self._update_status("✗ 请输入 URL")
                return
            if not url.startswith("http://") and not url.startswith("https://"):
                self._update_status("✗ URL 必须以 http:// 或 https:// 开头")
                return

            # Close dialog
            url_dialog.open = False
            self.app.page.update()

            # Start import
            await self._on_import_from_url(url)

        def on_cancel(e: ft.ControlEvent):
            url_dialog.open = False
            self.app.page.update()

        url_dialog = ft.AlertDialog(
            title=ft.Text("输入 Skill URL"),
            content=ft.Container(
                content=ft.Column(
                    [
                        url_input,
                    ],
                    tight=True,
                    spacing=16,
                ),
                width=200,
                height=100,
            ),
            actions=[
                ft.TextButton("取消", on_click=on_cancel),
                ft.ElevatedButton(
                    "确认",
                    on_click=on_confirm,
                    style=ft.ButtonStyle(
                        color=ft.Colors.WHITE,
                        bgcolor=ft.Colors.BLUE_700,
                    ),
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
            shape=ft.RoundedRectangleBorder(radius=8),
        )

        self.app.page.overlay.append(url_dialog)
        url_dialog.open = True
        self.app.page.update()

    async def _on_import_from_url(self, url: str):
        """Import skill from URL."""
        self._update_status(f"从 URL 导入中...")

        # Show loading state on online import button
        self._update_online_import_button_loading(True)
        await asyncio.sleep(0.1)  # Give UI time to render

        try:
            logger.info(f"[Online Import] Installing from URL: {url}")
            skill = await self.skill_market.install_from_url(url)
            if skill:
                logger.info(
                    f"[Online Import] Successfully installed skill: {skill.name}"
                )
                self.installed_skills.add(skill.name)
                self._update_status(f"✓ {skill.name} 导入成功")

                # Set selected skill to the newly imported one
                self.selected_skill = skill

                # Refresh skills list (will auto-select the new skill)
                await self._load_skills()
            else:
                logger.error(
                    f"[Online Import] install_from_url returned None for URL: {url}"
                )
                self._update_status("✗ 导入失败")
        except Exception as ex:
            logger.error(f"[Online Import] Import failed with exception: {ex}")
            self._update_status(f"✗ 错误: {str(ex)}")
        finally:
            self._update_online_import_button_loading(False)

    def _update_status(self, message: str):
        """Update status text."""
        if self.status_text:
            self.status_text.value = message
            self.status_text.update()

    def _show_error(self, message: str):
        """Show error message."""
        self._update_status(f"✗ {message}")

    def _close_dialog(self, e=None):
        """Close the dialog."""
        if self.dialog and self.dialog.open:
            self.dialog.open = False
            if self.app.page:
                self.app.page.update()

    def show(self):
        """Show the Markket dialog."""
        # Create left panel components
        self.search_field = ft.TextField(
            prefix_icon=ft.icons.Icons.SEARCH,
            hint_text="Search skills...",
            on_change=self._on_search_change,
            border_radius=ft.BorderRadius.all(8),
        )

        self.skills_list = ft.ListView(
            expand=True,
            spacing=4,
            padding=ft.Padding(0, 8, 0, 8),
        )

        # Create right panel components
        self.detail_name = ft.Text(
            "Select a skill",
            size=18,
            weight=ft.FontWeight.W_600,
            color=ft.Colors.WHITE,
        )

        self.detail_description = ft.Text(
            "Click on a skill from the list to view details",
            size=13,
            color=ft.Colors.GREY_400,
        )

        self.detail_params = ft.Column(
            spacing=8,
            scroll=ft.ScrollMode.AUTO,
        )

        self.detail_dependencies = ft.Column(
            spacing=4,
        )

        self.action_button_container = ft.Container(
            content=ft.ElevatedButton(
                "安装",
                disabled=True,
                style=ft.ButtonStyle(
                    color=ft.Colors.WHITE,
                ),
            )
        )

        self.status_text = ft.Text(
            "",
            size=12,
            color=ft.Colors.GREY_500,
        )

        # Build left panel
        # Create buttons row with two buttons side by side
        self.local_import_button = ft.ElevatedButton(
            "本地导入",
            on_click=self._on_import_click,
            width=136,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.BLUE_700,
                shape=ft.RoundedRectangleBorder(radius=6),
            ),
        )
        self.online_import_button = ft.ElevatedButton(
            "在线导入",
            on_click=self._on_online_import_click,
            width=136,
            style=ft.ButtonStyle(
                color=ft.Colors.WHITE,
                bgcolor=ft.Colors.BLUE_700,
                shape=ft.RoundedRectangleBorder(radius=6),
            ),
        )
        buttons_row = ft.Row(
            [
                self.local_import_button,
                self.online_import_button,
            ],
            spacing=16,
            alignment=ft.MainAxisAlignment.CENTER,
        )

        left_panel = ft.Container(
            content=ft.Column(
                [
                    self.search_field,
                    ft.Divider(height=1, color=ft.Colors.GREY_800),
                    self.skills_list,
                    ft.Divider(height=1, color=ft.Colors.GREY_800),
                    buttons_row,
                ],
                spacing=12,
                expand=True,
            ),
            width=320,
            padding=ft.Padding(16, 16, 16, 16),
            border=ft.Border(right=ft.BorderSide(1, ft.Colors.GREY_800)),
        )

        # Build right panel
        detail_scrollable = ft.Column(
            [
                self.detail_description,
                ft.Divider(height=1, color=ft.Colors.GREY_800),
                ft.Text(
                    "Parameters:",
                    size=14,
                    weight=ft.FontWeight.W_500,
                    color=ft.Colors.WHITE,
                ),
                self.detail_params,
                ft.Divider(height=1, color=ft.Colors.GREY_800),
                ft.Text(
                    "Dependencies:",
                    size=14,
                    weight=ft.FontWeight.W_500,
                    color=ft.Colors.WHITE,
                ),
                self.detail_dependencies,
            ],
            spacing=16,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )

        right_panel = ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Icon(
                                ft.icons.Icons.EXTENSION,
                                color=ft.Colors.BLUE_400,
                                size=24,
                            ),
                            self.detail_name,
                        ],
                        spacing=8,
                    ),
                    detail_scrollable,
                    ft.Row(
                        [
                            self.status_text,
                            ft.Container(expand=True),
                            self.action_button_container,
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                ],
                spacing=16,
                expand=True,
            ),
            padding=ft.Padding(24, 24, 24, 16),
            expand=True,
        )

        # Build main content
        content = ft.Row(
            [
                left_panel,
                right_panel,
            ],
            spacing=0,
            expand=True,
        )

        # Calculate dialog size (same as settings panel)
        dialog_width = (
            min(self.app.page.width * 0.8, 900) if self.app.page.width else 800
        )
        dialog_height = (
            min(self.app.page.height * 0.8, 700) if self.app.page.height else 600
        )

        close_button = ft.IconButton(
            icon=ft.Icons.CLOSE,
            icon_color="#a0a0a0",
            icon_size=18,
            on_click=lambda _: self._close_dialog(),
            tooltip=t("common.close"),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=4),
                padding=ft.Padding(4, 4, 4, 4),
                mouse_cursor=ft.MouseCursor.CLICK,
            )
        )

        # Create dialog container
        dialog_container = ft.Container(
            content=ft.Stack(
                [
                    ft.Container(
                        content=content,
                        width=dialog_width,
                        height=dialog_height,
                    ),
                    ft.Container(
                        content=close_button,
                        right=10,
                        top=10,
                    ),
                ],
            ),
            width=dialog_width,
            height=dialog_height,
            bgcolor=ft.Colors.GREY_900,
            border_radius=ft.BorderRadius.all(8),
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
        )

        # Create dialog
        self.dialog = ft.AlertDialog(
            content=dialog_container,
            content_padding=0,
            bgcolor="#00000000",
            shape=ft.RoundedRectangleBorder(radius=8),
        )

        # Show dialog
        self.app.page.overlay.append(self.dialog)
        self.dialog.open = True
        self.app.page.update()

        # Initialize skill market and load skills
        asyncio.create_task(self._init_skill_market())
