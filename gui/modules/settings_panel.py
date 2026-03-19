from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

import flet as ft

from middleware.config import g_config
from gui.i18n import t
from gui.modules.auto_update_manager import (
    AutoUpdateManager,
    UpdateStatus,
    UpdateInfo,
    DownloadProgress,
)


# Setup logger
logger = logging.getLogger(__name__)

# Title mapping for settings fields
SETTINGS_TITLE_MAP = {
    # LLM Profile fields
    "model": "模型",
    "api_key": "API Key",
    "base_url": "Base URL",
    "context_window": "上下文窗口",
    "max_tokens": "最大输出 Token 数",
    "temperature": "Temperature",
    "timeout": "超时时间(秒)",
    "extra_headers": "额外请求头",
    "extra_body": "额外请求参数",
    # App fields
    "theme": "主题",
    "language": "语言",
    "name": "应用名称",
    # General fields
    "active_profile": "当前配置",
    # Skills fields
    "catalog_path": "技能目录路径",
    "github_token": "GitHub Token",
    "cloud_catalog_url": "云端技能目录",
    "top_k": "检索Top K",
    "min_score": "最小分数",
    "embedding_model": "嵌入模型",
    "embedding_api_key": "嵌入API Key",
    "embedding_base_url": "嵌入Base URL",
    "reranker_enabled": "启用重排序",
    "reranker_min_score": "重排序最小分数",
    "timeout_sec": "执行超时(秒)",
    "max_reflection_retries": "最大反思重试次数",
    "sandbox_provider": "沙箱提供者",
    "e2b_api_key": "E2B API Key",
    "resolve_strategy": "解析策略",
    "download_method": "下载方式",
    # Provider fields
    "search": "搜索",
    "skills": "技能",
    "storage": "存储",
    "advanced": "高级",
}


def _get_settings_title(key_path: str) -> tuple[str, str]:
    """Get display title and description from key path.

    Returns:
        tuple: (title, description)
    """
    parts = key_path.split(".")
    field_name = parts[-1]

    # Map field names to translation keys
    field_to_trans_key = {
        "model": "fields.model",
        "api_key": "fields.api_key",
        "base_url": "fields.base_url",
        "context_window": "fields.context_window",
        "max_tokens": "fields.max_tokens",
        "temperature": "fields.temperature",
        "timeout": "fields.timeout",
        "extra_headers": "fields.extra_headers",
        "extra_body": "fields.extra_body",
        "theme": "fields.theme",
        "language": "fields.language",
        "name": "fields.name",
        "active_profile": "fields.active_profile",
        "catalog_path": "fields.catalog_path",
        "cloud_catalog_url": "fields.cloud_catalog_url",
        "top_k": "fields.top_k",
        "min_score": "fields.min_score",
        "embedding_model": "fields.embedding_model",
        "embedding_api_key": "fields.embedding_api_key",
        "embedding_base_url": "fields.embedding_base_url",
        "reranker_enabled": "fields.reranker_enabled",
        "reranker_min_score": "fields.reranker_min_score",
        "timeout_sec": "fields.timeout_sec",
        "max_reflection_retries": "fields.max_reflection_retries",
        "sandbox_provider": "fields.sandbox_provider",
        "resolve_strategy": "fields.resolve_strategy",
        "download_method": "fields.download_method",
    }

    # Try to get translation, fallback to map or generated name
    trans_key = field_to_trans_key.get(field_name)
    if trans_key:
        title = t(
            f"settings_panel.{trans_key}",
            default=SETTINGS_TITLE_MAP.get(
                field_name, field_name.replace("_", " ").capitalize()
            ),
        )
    else:
        title = SETTINGS_TITLE_MAP.get(
            field_name, field_name.replace("_", " ").capitalize()
        )

    # Generate description based on context
    if len(parts) >= 3:
        if parts[0] == "llm" and parts[1] == "profiles":
            description = f"LLM Profile {parts[-2]}.{field_name}"
        else:
            description = ".".join(parts[:-1])
    else:
        description = key_path

    return title, description


class SettingsPanel:
    """Settings panel with category-based navigation like VS Code/IDE settings."""

    def __init__(
        self,
        page: ft.Page,
        show_error: Callable[[str], None],
        show_snackbar: Callable[[str], None],
        on_save_callback: Callable[[], None] | None = None,
    ):
        self.page = page
        self.show_error = show_error
        self.show_snackbar = show_snackbar
        self.on_save_callback = on_save_callback
        self.dialog = None
        # 使用内部键（非翻译文本）来标识当前分类
        self.current_category = "general"
        self.settings_data = {}

        # Initialize auto update manager
        self._update_manager: AutoUpdateManager | None = None
        self._init_update_manager()

        # 注册语言切换观察者
        from gui.i18n import add_observer

        add_observer(self._on_language_changed)

    def _on_language_changed(self, new_lang: str):
        """语言切换时的回调 - 刷新设置面板"""
        # 如果设置面板当前打开，刷新内容
        if self.dialog and self.dialog.open:
            self._refresh_content()

    def show(self, default_category: str | None = None):
        """Show settings dialog.

        Args:
            default_category: 默认选中的分类名称，如 "大模型"、"通用" 等
        """
        logger.info(
            f"[SettingsPanel] show() called with default_category={default_category}"
        )
        try:
            logger.info("[SettingsPanel] Getting categories...")
            categories = self._get_categories()
            logger.info(f"[SettingsPanel] categories: {categories}")

            if not categories:
                logger.error("[SettingsPanel] No categories available")
                self.show_error("No settings available")
                return

            # 如果有指定默认分类，切换到该分类
            if default_category and default_category in categories:
                self.current_category = default_category
                logger.info(
                    f"[SettingsPanel] Switched to default category: {default_category}"
                )
            elif self.current_category not in categories:
                logger.info(
                    f"[SettingsPanel] Current category {self.current_category} not in categories, setting to {list(categories.keys())[0]}"
                )
                self.current_category = list(categories.keys())[0]

            logger.info("[SettingsPanel] Building sidebar...")
            sidebar = ft.Container(
                content=ft.Column(
                    [
                        ft.Text(
                            "设置",
                            size=16,
                            weight=ft.FontWeight.W_600,
                            color="#e0e0e0",
                        ),
                        ft.Container(height=12),
                        self._build_category_sidebar(list(categories.keys())),
                    ],
                    spacing=4,
                ),
                width=180,
                padding=ft.Padding(16, 20, 16, 20),
                bgcolor="#252526",
                alignment=ft.Alignment(0, -1),
            )

            logger.info("[SettingsPanel] Building settings content...")
            try:
                settings_content_obj = self._build_settings_content()
                logger.info("[SettingsPanel] Settings content built successfully")
            except Exception as e:
                logger.error(
                    f"[SettingsPanel] Error building settings content: {e}",
                    exc_info=True,
                )
                self.show_error(f"Error building settings: {str(e)}")
                return

            settings_content = ft.Container(
                content=settings_content_obj,
                padding=ft.Padding(24, 20, 24, 20),
                bgcolor="#1e1e1e",
                alignment=ft.Alignment(0, -1),
                expand=True,
            )

            logger.info("[SettingsPanel] Creating dialog...")

            divider = ft.Container(width=1, bgcolor="#383838")

            content = ft.Row(
                [
                    sidebar,
                    divider,
                    settings_content,
                ],
                spacing=0,
                expand=True,
            )

            # 计算对话框宽度为父窗口的80%
            dialog_width = min(self.page.width * 0.8, 900) if self.page.width else 800
            dialog_height = (
                min(self.page.height * 0.7, 600) if self.page.height else 500
            )

            # 存储对话框高度供 Raw section 使用
            self._dialog_height = dialog_height

            dialog_container = ft.Container(
                content=content,
                width=dialog_width,
                height=dialog_height,
                border_radius=8,
                border=ft.border.all(0.5, "#000000"),
            )

            self.dialog = ft.AlertDialog(
                content=dialog_container,
                content_padding=0,
                bgcolor="#00000000",
                shape=ft.RoundedRectangleBorder(radius=8),
            )
            self.dialog.open = True
            self.page.overlay.append(self.dialog)
            self.page.update()

        except Exception as e:
            self.show_error(f"Failed to open settings: {str(e)}")

    def _init_update_manager(self):
        """Initialize auto update manager with callbacks."""
        self._update_manager = AutoUpdateManager()
        self._update_manager.set_callbacks(
            on_status_change=self._on_update_status_change,
            on_progress=self._on_update_progress,
            on_download_complete=self._on_download_complete,
            on_error=self._on_update_error,
        )

    def _on_update_status_change(self, status: UpdateStatus):
        """Handle update status changes."""
        logger.info(f"[SettingsPanel] Update status: {status.name}")

        if status == UpdateStatus.DOWNLOADING:
            self._update_progress.visible = True
            self._update_button.visible = False
            self._update_status_text.value = "Downloading..."
            self._update_status_text.color = ft.Colors.BLUE_400
        elif status == UpdateStatus.DOWNLOADED:
            self._update_progress.visible = False
            self._update_button.visible = True
            version_str = ""
            if self._update_manager and self._update_manager.current_update:
                version_str = self._update_manager.current_update.version
            self._update_status_text.value = t(
                "settings_panel.update_available",
                version=version_str,
            )
            self._update_status_text.color = ft.Colors.AMBER_400
        elif status == UpdateStatus.ERROR:
            self._update_progress.visible = False
            self._update_button.visible = True
            self._update_status_text.value = t("settings_panel.check_failed")
            self._update_status_text.color = ft.Colors.RED_400

        self.page.update()

    def _on_update_progress(self, progress: DownloadProgress):
        """Handle download progress updates."""
        # Progress is handled by the manager's UI
        pass

    def _on_download_complete(self, update_info: UpdateInfo):
        """Handle download completion."""
        logger.info(f"[SettingsPanel] Download complete: {update_info.version}")
        self._show_install_confirmation_dialog(update_info)

    def _on_update_error(self, message: str):
        """Handle update errors."""
        logger.error(f"[SettingsPanel] Update error: {message}")
        self.show_error(message)
        self._update_progress.visible = False
        self._update_button.visible = True
        self._update_status_text.value = t("settings_panel.check_failed")
        self._update_status_text.color = ft.Colors.RED_400
        self.page.update()

    def _get_config_value(self, key_path: str) -> Any:
        """Get config value by key path like 'llm.api_key'"""
        keys = key_path.split(".")
        value = self.settings_data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def _set_config_value(self, key_path: str, value: Any):
        """Set config value by key path and save immediately."""
        try:
            g_config.set(key_path, value, save=False)
            g_config.save()
            self.show_snackbar(t("settings_panel.messages.saved", key_path=key_path))
            if self.on_save_callback:
                self.on_save_callback()
        except Exception as e:
            self.show_error(t("settings_panel.messages.save_failed", error=str(e)))

    def _build_category_sidebar(self, categories: list[str]) -> ft.Column:
        """Build left sidebar with category list."""
        category_buttons = []

        # 分类键到翻译的映射
        category_trans_keys = {
            "general": "settings_panel.categories.general",
            "llm": "settings_panel.categories.llm",
            "skills": "settings_panel.categories.skills",
            "storage": "settings_panel.categories.storage",
            "advanced": "settings_panel.categories.advanced",
            "raw": "settings_panel.categories.raw",
        }

        for category in categories:
            is_selected = category == self.current_category
            # 获取翻译后的分类名称
            display_name = t(category_trans_keys.get(category, category))
            btn = ft.Container(
                content=ft.Row(
                    [
                        ft.Text(
                            display_name,
                            size=13,
                            weight=ft.FontWeight.W_500
                            if is_selected
                            else ft.FontWeight.W_400,
                            color=ft.Colors.WHITE if is_selected else "#808080",
                        )
                    ],
                    expand=True,
                ),
                padding=ft.Padding(12, 8, 12, 8),
                bgcolor="#3b82f6" if is_selected else "transparent",
                border_radius=ft.BorderRadius.all(4),
                on_click=lambda e, cat=category: self._on_category_click(cat),
                animate=ft.Animation(200, ft.AnimationCurve.EASE_OUT),
            )
            category_buttons.append(btn)

        return ft.Column(
            category_buttons,
            spacing=1,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )

    def _on_category_click(self, category: str):
        """Handle category selection."""
        self.current_category = category
        self._refresh_content()

    def _refresh_content(self):
        """Refresh the content area based on current category."""
        try:
            if not (self.dialog and self.dialog.content):
                return

            dialog_container = self.dialog.content
            if not (hasattr(dialog_container, "content") and dialog_container.content):
                return

            content_row = dialog_container.content
            if not (
                hasattr(content_row, "controls")
                and isinstance(content_row.controls, list)
                and len(content_row.controls) >= 3
            ):
                return

            controls_list = content_row.controls

            # Update sidebar
            sidebar = controls_list[0]
            if (
                hasattr(sidebar, "content")
                and sidebar.content
                and hasattr(sidebar.content, "controls")
            ):
                sidebar_content = sidebar.content
                sidebar_controls = sidebar_content.controls
                if isinstance(sidebar_controls, list) and len(sidebar_controls) > 2:
                    sidebar_controls[2] = self._build_category_sidebar(
                        list(self._get_categories().keys())
                    )

            # Update settings content
            settings_container = controls_list[2]
            if hasattr(settings_container, "content"):
                settings_container.content = self._build_settings_content()

        except Exception as e:
            logger.error(f"Failed to build settings UI: {e}", exc_info=True)
            # Try to display a user-friendly error in the settings panel
            if "settings_container" in locals() and hasattr(
                settings_container, "content"
            ):
                settings_container.content = ft.Column(
                    [
                        ft.Text("Error Loading Settings", color=ft.Colors.RED, size=16),
                        ft.Text(
                            "An error occurred while building the settings view. "
                            "Please check the logs for more details.",
                            size=12,
                        ),
                        ft.TextField(
                            value=str(e),
                            multiline=True,
                            read_only=True,
                            border_color=ft.Colors.RED,
                        ),
                    ]
                )
            else:
                self.show_error(f"Failed to refresh settings: {e}")
        finally:
            # Always try to update the page
            if self.page:
                self.page.update()

    def _get_categories(self) -> dict[str, list[tuple[str, Any, str]]]:
        """Organize settings into categories."""
        # 使用内部键（非翻译）来标识分类
        categories = {
            "general": [],
            "llm": [],
            "skills": [],
            "storage": [],
            "advanced": [],
            "raw": [],
        }

        if g_config is None:
            try:
                g_config.load()
            except Exception:
                pass

        if g_config:
            self.settings_data = g_config.to_json_dict()

            def add_to_category(key_path: str, value: Any):
                key_lower = key_path.lower()

                if "llm" in key_lower or "api_key" in key_lower or "model" in key_lower:
                    if isinstance(value, (str, int, float, bool)):
                        categories["llm"].append(
                            (key_path, value, self._get_field_type(value))
                        )
                elif "skill" in key_lower:
                    if isinstance(value, (str, int, float, bool)):
                        categories["skills"].append(
                            (key_path, value, self._get_field_type(value))
                        )
                elif (
                    "storage" in key_lower
                    or "database" in key_lower
                    or "db" in key_lower
                ):
                    if isinstance(value, (str, int, float, bool)):
                        categories["storage"].append(
                            (key_path, value, self._get_field_type(value))
                        )
                elif key_lower.startswith(("debug", "log", "verbose", "experimental")):
                    if isinstance(value, (str, int, float, bool)):
                        categories["advanced"].append(
                            (key_path, value, self._get_field_type(value))
                        )
                else:
                    if isinstance(value, (str, int, float, bool)):
                        categories["general"].append(
                            (key_path, value, self._get_field_type(value))
                        )

            def flatten(obj: Any, prefix: str = ""):
                if isinstance(obj, dict):
                    for key, val in obj.items():
                        new_key = f"{prefix}.{key}" if prefix else key

                        # --- Memento-S Change: Exclude LLM profiles from flattening ---
                        if new_key == "llm.profiles":
                            continue
                        # ---------------------------------------------------------

                        if isinstance(val, dict):
                            flatten(val, new_key)
                        else:
                            add_to_category(new_key, val)
                elif isinstance(obj, list):
                    pass

            flatten(self.settings_data)

        # ========== 配置开关：隐藏额外的分类选项 ==========
        # 设置为 True 则只显示 general 和 llm，隐藏 skills、storage、advanced
        # 设置为 False 则显示所有分类
        HIDE_EXTRA_CATEGORIES = True
        # =================================================

        if HIDE_EXTRA_CATEGORIES:
            hidden = {"skills", "storage", "advanced"}
            # 返回非空分类，但始终包含 raw 分类（即使为空）
            return {
                k: v
                for k, v in categories.items()
                if (v or k == "raw") and k not in hidden
            }
        else:
            # 返回所有分类，包括空的 raw 分类
            return {k: v for k, v in categories.items() if v or k == "raw"}

    def _get_field_type(self, value: Any) -> str:
        """Determine field type for UI control."""
        if isinstance(value, bool):
            return "bool"
        elif isinstance(value, int):
            return "int"
        elif isinstance(value, float):
            return "float"
        else:
            return "str"

    def _build_settings_content(self) -> ft.Column:
        """Build settings content for current category."""
        logger.info(
            f"[SettingsPanel] Building settings content for category: {self.current_category}"
        )

        categories = self._get_categories()
        logger.info(f"[SettingsPanel] Available categories: {list(categories.keys())}")

        settings = categories.get(self.current_category, [])
        logger.info(
            f"[SettingsPanel] Settings for {self.current_category}: {len(settings)} items"
        )

        controls = []

        # General category: split into Appearance, Cache, and Update sections
        if self.current_category == "general":
            logger.info("[SettingsPanel] Building General category sections")
            try:
                appearance = self._build_appearance_section()
                logger.info("[SettingsPanel] Appearance section built successfully")
            except Exception as e:
                logger.error(
                    f"[SettingsPanel] Error building appearance section: {e}",
                    exc_info=True,
                )
                raise
            try:
                api_keys = self._build_api_keys_section()
                logger.info("[SettingsPanel] API Keys section built successfully")
            except Exception as e:
                logger.error(
                    f"[SettingsPanel] Error building API keys section: {e}",
                    exc_info=True,
                )
                raise
            try:
                cache = self._build_cache_section()
                logger.info("[SettingsPanel] Cache section built successfully")
            except Exception as e:
                logger.error(
                    f"[SettingsPanel] Error building cache section: {e}", exc_info=True
                )
                raise
            try:
                update = self._build_update_section()
                logger.info("[SettingsPanel] Update section built successfully")
            except Exception as e:
                logger.error(
                    f"[SettingsPanel] Error building update section: {e}", exc_info=True
                )
                raise
            controls.append(appearance)
            controls.append(api_keys)
            controls.append(cache)
            controls.append(update)
            return ft.Column(
                controls,
                spacing=12,
                scroll=ft.ScrollMode.AUTO,
                alignment=ft.MainAxisAlignment.START,
            )

        # --- Memento-S Change: Use dedicated LLM settings builder ---
        if self.current_category == "llm":
            # Hide all llm settings above (active_profile is in the dropdown now)
            # Only show profile settings
            controls.append(self._build_llm_profile_settings())

            return ft.Column(
                controls,
                spacing=16,
                scroll=ft.ScrollMode.AUTO,
                expand=True,
            )
        # ----------------------------------------------------------

        # --- Memento-S Change: Use dedicated Skills settings builder ---
        if self.current_category == "skills":
            controls.append(self._build_skills_section())

            return ft.Column(
                controls,
                spacing=16,
                scroll=ft.ScrollMode.AUTO,
                expand=True,
            )
        # ----------------------------------------------------------

        # --- Memento-S Change: Use dedicated Storage settings builder ---
        if self.current_category == "storage":
            category_display = t("settings_panel.categories.storage")
            controls.append(
                self._build_generic_section(
                    t(
                        "settings_panel.messages.category_settings",
                        category=category_display,
                    ),
                    settings,
                )
            )

            return ft.Column(
                controls,
                spacing=16,
                scroll=ft.ScrollMode.AUTO,
                expand=True,
            )
        # ----------------------------------------------------------

        # --- Memento-S Change: Use dedicated Advanced settings builder ---
        if self.current_category == "advanced":
            category_display = t("settings_panel.categories.advanced")
            controls.append(
                self._build_generic_section(
                    t(
                        "settings_panel.messages.category_settings",
                        category=category_display,
                    ),
                    settings,
                )
            )

            return ft.Column(
                controls,
                spacing=16,
                scroll=ft.ScrollMode.AUTO,
                expand=True,
            )
        # ----------------------------------------------------------

        # --- Memento-S Change: Use dedicated Raw settings builder ---
        if self.current_category == "raw":
            controls.append(self._build_raw_section())

            return ft.Column(
                controls,
                spacing=16,
                scroll=ft.ScrollMode.AUTO,
                expand=True,
            )
        # ----------------------------------------------------------

        if not settings:
            # For other categories, if no settings, show message
            return ft.Column(
                [
                    ft.Container(
                        content=ft.Row(
                            [
                                ft.Text(
                                    t("settings_panel.empty_category"),
                                    color=ft.Colors.GREY_500,
                                    size=14,
                                )
                            ],
                            alignment=ft.MainAxisAlignment.CENTER,
                        ),
                        padding=40,
                    )
                ],
                expand=True,
            )

        # For all other categories, build controls normally
        for key_path, value, field_type in settings:
            control = self._create_setting_control(key_path, value, field_type)
            controls.append(control)

        return ft.Column(
            controls,
            spacing=16,
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )

    def _build_appearance_section(self) -> ft.Container:
        """Build appearance settings section (language, theme)."""
        logger.info("[SettingsPanel] _build_appearance_section called")

        current_config = g_config

        current_theme = (
            current_config.app.theme
            if current_config and current_config.app
            else "system"
        )
        current_language = (
            current_config.app.language
            if current_config and current_config.app
            else "en-US"
        )

        # Use translations for theme and language options
        theme_options = [
            ft.dropdown.Option(key, t(f"settings_panel.themes.{key}"))
            for key in ["system"]  # "light", "dark"
        ]
        # 必须包含所有可能的语言选项，否则当前语言不在选项中时会显示为空
        language_options = [
            ft.dropdown.Option(key, t(f"settings_panel.languages.{key}"))
            for key in ["en-US"]  # Hide "zh-CN",
        ]

        logger.info("[SettingsPanel] Creating appearance container")

        language_dropdown = ft.Dropdown(
            value=current_language,
            options=language_options,
            width=150,
            border_color="#404040",
            focused_border_color="#3b82f6",
            content_padding=10,
        )
        language_dropdown.on_select = lambda e: self._on_language_change(
            e.control.value
        )

        theme_dropdown = ft.Dropdown(
            value=current_theme,
            options=theme_options,
            width=150,
            border_color="#404040",
            focused_border_color="#3b82f6",
            content_padding=10,
        )
        theme_dropdown.on_select = lambda e: self._on_theme_change(e.control.value)

        return ft.Container(
            content=ft.Column(
                [
                    ft.Container(
                        content=ft.Text(
                            t("settings_panel.appearance"),
                            size=13,
                            weight=ft.FontWeight.W_500,
                            color="#a0a0a0",
                        ),
                        padding=ft.Padding(0, 0, 0, 8),
                        expand=True,
                    ),
                    ft.Container(
                        content=ft.Column(
                            [
                                ft.Container(
                                    content=ft.Row(
                                        [
                                            ft.Text(
                                                t("settings_panel.language"),
                                                size=13,
                                                color="#e0e0e0",
                                            ),
                                            language_dropdown,
                                        ],
                                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                                    ),
                                    padding=ft.Padding(12, 10, 12, 10),
                                ),
                                ft.Divider(height=1, color="#383838"),
                                ft.Container(
                                    content=ft.Row(
                                        [
                                            ft.Text(
                                                t("settings_panel.theme"),
                                                size=13,
                                                color="#e0e0e0",
                                            ),
                                            theme_dropdown,
                                        ],
                                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                                    ),
                                    padding=ft.Padding(12, 10, 12, 10),
                                ),
                            ],
                            spacing=0,
                        ),
                        bgcolor="#2d2d2d",
                        border=ft.Border(
                            top=ft.BorderSide(1, "#383838"),
                            bottom=ft.BorderSide(1, "#383838"),
                            left=ft.BorderSide(1, "#383838"),
                            right=ft.BorderSide(1, "#383838"),
                        ),
                        border_radius=ft.BorderRadius.all(6),
                    ),
                ],
                spacing=4,
            ),
        )

    def _build_update_section(self) -> ft.Container:
        """Build update section with version and check update button."""
        logger.info("[SettingsPanel] _build_update_section called")

        current_version = (
            self._update_manager._get_current_version()
            if self._update_manager
            else "1.0.0"
        )

        self._update_status_text = ft.Text(
            f"v{current_version}",
            size=13,
            color="#a0a0a0",
        )

        self._update_button = ft.TextButton(
            t("settings_panel.check_update"),
            style=ft.ButtonStyle(color="#3b82f6"),
            on_click=lambda e: asyncio.create_task(self._handle_update_check()),
        )

        self._update_progress = ft.ProgressRing(
            width=16,
            height=16,
            stroke_width=2,
            visible=False,
            color="#3b82f6",
        )

        logger.info("[SettingsPanel] Creating update container")

        return ft.Container(
            content=ft.Column(
                [
                    ft.Container(
                        content=ft.Text(
                            t("settings_panel.update"),
                            size=13,
                            weight=ft.FontWeight.W_500,
                            color="#a0a0a0",
                        ),
                        padding=ft.Padding(0, 16, 0, 8),
                        expand=True,
                    ),
                    ft.Container(
                        content=ft.Column(
                            [
                                ft.Container(
                                    content=ft.Row(
                                        [
                                            ft.Text(
                                                t("settings_panel.version"),
                                                size=13,
                                                color="#e0e0e0",
                                            ),
                                            self._update_status_text,
                                        ],
                                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                                    ),
                                    padding=ft.Padding(12, 10, 12, 10),
                                ),
                                ft.Divider(height=1, color="#383838"),
                                ft.Container(
                                    content=ft.Row(
                                        [
                                            ft.Text(
                                                t("settings_panel.check_update"),
                                                size=13,
                                                color="#e0e0e0",
                                            ),
                                            ft.Row(
                                                [
                                                    self._update_progress,
                                                    self._update_button,
                                                ],
                                                spacing=8,
                                            ),
                                        ],
                                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                                    ),
                                    padding=ft.Padding(12, 10, 12, 10),
                                ),
                            ],
                            spacing=0,
                        ),
                        bgcolor="#2d2d2d",
                        border=ft.Border(
                            top=ft.BorderSide(1, "#383838"),
                            bottom=ft.BorderSide(1, "#383838"),
                            left=ft.BorderSide(1, "#383838"),
                            right=ft.BorderSide(1, "#383838"),
                        ),
                        border_radius=ft.BorderRadius.all(6),
                    ),
                ],
                spacing=0,
            ),
        )

    def _build_api_keys_section(self) -> ft.Container:
        """Build API Keys section with TAVILY_API_KEY setting."""
        logger.info("[SettingsPanel] _build_api_keys_section called")

        # 从配置读取 TAVILY_API_KEY
        current_config = g_config
        tavily_api_key = (
            current_config.env.get("TAVILY_API_KEY", "")
            if current_config and current_config.env
            else ""
        )

        # 创建密码输入框（自动识别为密码类型）
        api_key_field = ft.TextField(
            value=tavily_api_key,
            password=True,
            can_reveal_password=True,
            width=250,
            height=38,
            content_padding=ft.Padding(left=10, top=4, right=10, bottom=4),
            border_color="#404040",
            focused_border_color="#3b82f6",
            hint_text="Enter your TAVILY API Key",
        )

        # 自动保存配置
        def on_api_key_change(e):
            new_value = e.control.value
            self._set_config_value("env.TAVILY_API_KEY", new_value)

        api_key_field.on_blur = on_api_key_change
        api_key_field.on_submit = on_api_key_change

        return ft.Container(
            content=ft.Column(
                [
                    ft.Container(
                        content=ft.Text(
                            t("settings_panel.api_keys"),
                            size=13,
                            weight=ft.FontWeight.W_500,
                            color="#a0a0a0",
                        ),
                        padding=ft.Padding(0, 0, 0, 8),
                        expand=True,
                    ),
                    ft.Container(
                        content=ft.Column(
                            [
                                ft.Container(
                                    content=ft.Row(
                                        [
                                            ft.Text(
                                                t("settings_panel.tavily_api_key"),
                                                size=13,
                                                color="#e0e0e0",
                                            ),
                                            api_key_field,
                                        ],
                                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                                    ),
                                    padding=ft.Padding(12, 10, 12, 10),
                                ),
                            ],
                            spacing=0,
                        ),
                        bgcolor="#2d2d2d",
                        border=ft.Border(
                            top=ft.BorderSide(1, "#383838"),
                            bottom=ft.BorderSide(1, "#383838"),
                            left=ft.BorderSide(1, "#383838"),
                            right=ft.BorderSide(1, "#383838"),
                        ),
                        border_radius=ft.BorderRadius.all(6),
                    ),
                ],
                spacing=4,
            ),
        )

    def _build_raw_section(self) -> ft.Container:
        """Build raw config editor section."""
        logger.info("[SettingsPanel] _build_raw_section called")

        from utils.path_manager import PathManager
        import json

        config_path = PathManager.get_config_file()

        # 读取当前配置文件内容
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_content = f.read()
        except Exception as e:
            config_content = f"Error reading config: {e}"

        # 计算 TextField 的高度
        # Column 包含 4 个元素，spacing=8，共 3 个 spacing:
        # 标题 ~18px + spacing 8px + 路径 ~15px + spacing 8px + 按钮行 ~40px + spacing 8px = ~97px
        # 边距: 20px
        # 总计: 97 + 20 = 117px
        fixed_height = 117
        available_height = 500  # 默认
        if hasattr(self, "_dialog_height") and self._dialog_height:
            available_height = self._dialog_height
        print(f"available_height: {available_height}")
        text_field_height = max(200, available_height - fixed_height) - 10
        logger.info(
            f"[SettingsPanel] Raw section text field height: {text_field_height}"
        )

        # 创建文本编辑器
        config_editor = ft.TextField(
            value=config_content,
            multiline=True,
            text_size=12,
            border_color="#404040",
            focused_border_color="#3b82f6",
            height=text_field_height,
            expand=True,
        )

        # 状态显示
        status_text = ft.Text("", size=12, color="#808080")

        # def validate_config(e):
        #     """验证配置格式"""
        #     try:
        #         config_dict = json.loads(config_editor.value)
        #         # 使用 Pydantic 验证
        #         g_config._validate(config_dict)
        #         status_text.value = "配置验证通过"
        #         status_text.color = "#4caf50"
        #     except json.JSONDecodeError as ex:
        #         status_text.value = f"JSON 格式错误: {str(ex)}"
        #         status_text.color = "#f44336"
        #     except Exception as ex:
        #         print(f"Validation error: {ex}")
        #         status_text.value = f"配置验证失败: {str(ex)}"
        #         status_text.color = "#f44336"
        #     status_text.update()

        def save_config(e):
            """保存配置"""
            try:
                # 解析 JSON 格式
                config_dict = json.loads(config_editor.value)

                # 调用 save_raw_config 保存配置
                error = g_config.save_raw_config(config_dict)
                if error:
                    status_text.value = f"保存失败: {error}"
                    status_text.color = "#f44336"
                    status_text.update()
                    return

                # 重新加载配置
                g_config.load()

                status_text.value = "配置保存成功"
                status_text.color = "#4caf50"

                # 刷新其他 UI 部分
                if self.on_save_callback:
                    self.on_save_callback()

            except json.JSONDecodeError as ex:
                status_text.value = f"JSON 格式错误: {str(ex)}"
                status_text.color = "#f44336"
            except Exception as ex:
                status_text.value = f"保存失败: {str(ex)}"
                status_text.color = "#f44336"
            status_text.update()

        def reset_config(e):
            """重置为原始内容"""
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_editor.value = f.read()
                config_editor.update()
                status_text.value = "已重置为原始内容"
                status_text.color = "#808080"
                status_text.update()
            except Exception as ex:
                status_text.value = f"重置失败: {str(ex)}"
                status_text.color = "#f44336"
                status_text.update()

        return ft.Container(
            content=ft.Column(
                [
                    ft.Container(
                        content=ft.Text(
                            t("settings_panel.categories.raw"),
                            size=13,
                            weight=ft.FontWeight.W_500,
                            color="#a0a0a0",
                        ),
                        # padding=ft.Padding(0, 0, 0, 8),
                    ),
                    ft.Text(
                        f"配置文件路径: {config_path}",
                        size=11,
                        color="#606060",
                    ),
                    config_editor,
                    ft.Row(
                        [
                            # ft.TextButton(
                            #     "验证",
                            #     on_click=validate_config,
                            # ),
                            ft.TextButton(
                                "重置",
                                on_click=reset_config,
                            ),
                            ft.ElevatedButton(
                                "保存",
                                on_click=save_config,
                                style=ft.ButtonStyle(color="#3b82f6"),
                            ),
                            status_text,
                        ],
                        alignment=ft.MainAxisAlignment.START,
                        spacing=12,
                    ),
                ],
                spacing=8,
                expand=True,
            ),
            expand=True,
        )

    def _build_cache_section(self) -> ft.Container:
        """Build cache clearing section with skills and workspace directories."""
        logger.info("[SettingsPanel] _build_cache_section called")

        from utils.path_manager import PathManager

        # Get directory paths
        skills_dir = PathManager.get_skills_dir()
        workspace_dir = PathManager.get_workspace_dir()

        skills_path_text = ft.Text(
            str(skills_dir),
            size=11,
            color="#808080",
            expand=True,
            overflow=ft.TextOverflow.ELLIPSIS,
        )

        workspace_path_text = ft.Text(
            str(workspace_dir),
            size=11,
            color="#808080",
            expand=True,
            overflow=ft.TextOverflow.ELLIPSIS,
        )

        def _get_dir_size(path: Path) -> str:
            """Calculate directory size in human readable format."""
            try:
                if not path.exists():
                    return "0 B"
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if not os.path.islink(fp):
                            total_size += os.path.getsize(fp)
                # Convert to human readable
                for unit in ["B", "KB", "MB", "GB"]:
                    if total_size < 1024.0:
                        return f"{total_size:.1f} {unit}"
                    total_size /= 1024.0
                return f"{total_size:.1f} TB"
            except Exception as e:
                logger.error(f"Error calculating directory size: {e}")
                return "Unknown"

        def _refresh_dir_sizes():
            """Refresh the displayed directory sizes."""
            skills_size_text.value = f"({_get_dir_size(skills_dir)})"
            workspace_size_text.value = f"({_get_dir_size(workspace_dir)})"
            if self.page:
                self.page.update()

        skills_size_text = ft.Text(
            f"({_get_dir_size(skills_dir)})",
            size=11,
            color="#606060",
        )

        workspace_size_text = ft.Text(
            f"({_get_dir_size(workspace_dir)})",
            size=11,
            color="#606060",
        )

        def _show_delete_confirm_dialog(dir_name: str, dir_path: Path):
            """Show confirmation dialog before deleting directory."""

            def confirm_delete(e):
                dialog.open = False
                self.page.update()
                try:
                    if dir_path.exists():
                        shutil.rmtree(dir_path)
                        os.makedirs(dir_path, exist_ok=True)
                        self.show_snackbar(
                            t("settings_panel.cache_cleared", name=dir_name)
                        )
                        _refresh_dir_sizes()
                    else:
                        self.show_error(
                            t("settings_panel.cache_not_found", name=dir_name)
                        )
                except Exception as ex:
                    logger.error(f"Error clearing {dir_name} cache: {ex}")
                    self.show_error(
                        t(
                            "settings_panel.cache_clear_failed",
                            name=dir_name,
                            error=str(ex),
                        )
                    )

            def cancel_delete(e):
                dialog.open = False
                self.page.update()

            dialog = ft.AlertDialog(
                title=ft.Text(t("settings_panel.confirm_clear_cache")),
                content=ft.Text(
                    t(
                        "settings_panel.confirm_clear_cache_message",
                        name=dir_name,
                        path=str(dir_path),
                    )
                ),
                actions=[
                    ft.TextButton(t("settings_panel.cancel"), on_click=cancel_delete),
                    ft.TextButton(
                        t("settings_panel.confirm"),
                        on_click=confirm_delete,
                        style=ft.ButtonStyle(color=ft.Colors.RED_400),
                    ),
                ],
                actions_alignment=ft.MainAxisAlignment.END,
                shape=ft.RoundedRectangleBorder(radius=8),
            )
            self.page.overlay.append(dialog)
            dialog.open = True
            self.page.update()

        skills_delete_btn = ft.IconButton(
            icon=ft.icons.Icons.DELETE_OUTLINE,
            icon_color=ft.Colors.RED_400,
            tooltip=t("settings_panel.clear_cache"),
            on_click=lambda e: _show_delete_confirm_dialog("Skills", skills_dir),
        )

        workspace_delete_btn = ft.IconButton(
            icon=ft.icons.Icons.DELETE_OUTLINE,
            icon_color=ft.Colors.RED_400,
            tooltip=t("settings_panel.clear_cache"),
            on_click=lambda e: _show_delete_confirm_dialog("Workspace", workspace_dir),
        )

        return ft.Container(
            content=ft.Column(
                [
                    ft.Container(
                        content=ft.Text(
                            t("settings_panel.cache_management"),
                            size=13,
                            weight=ft.FontWeight.W_500,
                            color="#a0a0a0",
                        ),
                        padding=ft.Padding(0, 16, 0, 8),
                        expand=True,
                    ),
                    ft.Container(
                        content=ft.Column(
                            [
                                # Skills directory row
                                ft.Container(
                                    content=ft.Column(
                                        [
                                            ft.Row(
                                                [
                                                    ft.Text(
                                                        t(
                                                            "settings_panel.skills_directory"
                                                        ),
                                                        size=13,
                                                        color="#e0e0e0",
                                                    ),
                                                    skills_size_text,
                                                ],
                                                spacing=8,
                                            ),
                                            ft.Row(
                                                [
                                                    skills_path_text,
                                                    skills_delete_btn,
                                                ],
                                                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                                            ),
                                        ],
                                        spacing=4,
                                    ),
                                    padding=ft.Padding(12, 10, 12, 10),
                                ),
                                ft.Divider(height=1, color="#383838"),
                                # Workspace directory row
                                ft.Container(
                                    content=ft.Column(
                                        [
                                            ft.Row(
                                                [
                                                    ft.Text(
                                                        t(
                                                            "settings_panel.workspace_directory"
                                                        ),
                                                        size=13,
                                                        color="#e0e0e0",
                                                    ),
                                                    workspace_size_text,
                                                ],
                                                spacing=8,
                                            ),
                                            ft.Row(
                                                [
                                                    workspace_path_text,
                                                    workspace_delete_btn,
                                                ],
                                                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                                            ),
                                        ],
                                        spacing=4,
                                    ),
                                    padding=ft.Padding(12, 10, 12, 10),
                                ),
                            ],
                            spacing=0,
                        ),
                        bgcolor="#2d2d2d",
                        border=ft.Border(
                            top=ft.BorderSide(1, "#383838"),
                            bottom=ft.BorderSide(1, "#383838"),
                            left=ft.BorderSide(1, "#383838"),
                            right=ft.BorderSide(1, "#383838"),
                        ),
                        border_radius=ft.BorderRadius.all(6),
                    ),
                ],
                spacing=4,
            ),
        )

    def _on_language_change(self, value: str | None):
        """Handle language change."""
        if not value:
            return
        try:
            # 使用 i18n 模块的 set_language 方法来切换语言并通知观察者
            from gui.i18n import set_language

            if set_language(value):
                lang_display = (
                    t(f"settings_panel.languages.{value}", default=value)
                    if value
                    else value
                )
                self.show_snackbar(
                    t("settings_panel.messages.language_changed", lang=lang_display)
                )
                self._refresh_content()
            else:
                self.show_error(
                    t("settings_panel.messages.language_change_failed", lang=value)
                )
        except Exception as e:
            self.show_error(t("settings_panel.messages.save_failed", error=str(e)))

    def _on_theme_change(self, value: str | None):
        """Handle theme change."""
        try:
            g_config.set("app.theme", value, save=False)
            g_config.save()

            if self.page:
                if value == "system":
                    self.page.theme_mode = ft.ThemeMode.SYSTEM
                elif value == "light":
                    self.page.theme_mode = ft.ThemeMode.LIGHT
                elif value == "dark":
                    self.page.theme_mode = ft.ThemeMode.DARK
                self.page.update()

            theme_display = (
                t(f"settings_panel.themes.{value}", default=value) if value else value
            )
            self.show_snackbar(
                t("settings_panel.messages.theme_changed", theme=theme_display)
            )
        except Exception as e:
            self.show_error(t("settings_panel.messages.save_failed", error=str(e)))

    def _create_setting_control(
        self, key_path: str, value: Any, field_type: str
    ) -> ft.Container:
        """Create a UI control for a given setting."""

        def on_change(e):
            new_value = e.control.value
            if field_type == "bool":
                new_value = bool(new_value)
            elif field_type == "int":
                try:
                    new_value = int(new_value)
                except (ValueError, TypeError):
                    self.show_error(f"Invalid integer for {key_path}: {new_value}")
                    return
            elif field_type == "float":
                try:
                    new_value = float(new_value)
                except (ValueError, TypeError):
                    self.show_error(f"Invalid float for {key_path}: {new_value}")
                    return

            self._set_config_value(key_path, new_value)

            # If we change the active profile, we need to refresh the whole view
            if key_path == "llm.active_profile":
                # A short delay to ensure config is saved and reloaded before refresh
                asyncio.create_task(self._delayed_refresh())

        # Special handling for the llm.active_profile dropdown
        if key_path == "llm.active_profile":
            profile_names = []
            if g_config and g_config.llm:
                profile_names = list(g_config.llm.profiles.keys())

            control = ft.Dropdown(
                value=str(value),
                options=[ft.dropdown.Option(name) for name in profile_names],
                dense=True,
                height=38,
                content_padding=ft.Padding(left=10, top=0, right=2, bottom=0),
            )
            control.on_change = on_change
        elif field_type == "bool":
            control = ft.Switch(value=bool(value), on_change=on_change)
        else:  # str, int, float
            is_password = "api_key" in key_path.lower() or "token" in key_path.lower()
            control = ft.TextField(
                value=str(value),
                on_submit=on_change,
                password=is_password,
                can_reveal_password=is_password,
                height=38,
                content_padding=ft.Padding(left=10, top=4, right=10, bottom=4),
            )
            # To save on blur as well for text fields
            control.on_blur = on_change

        # Use key_path as description
        title, description = _get_settings_title(key_path)

        return ft.Container(
            content=ft.Row(
                [
                    ft.Column(
                        [
                            ft.Text(title, size=13, weight=ft.FontWeight.W_500),
                            ft.Text(description, size=11, color=ft.Colors.GREY_500),
                        ],
                        expand=True,
                        spacing=2,
                    ),
                    control,
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            padding=ft.Padding(16, 12, 16, 12),
            border=ft.Border(bottom=ft.BorderSide(1, ft.Colors.GREY_800)),
            border_radius=ft.BorderRadius.all(4),
        )

    async def _delayed_refresh(self):
        await asyncio.sleep(0.1)
        # Reload config before refreshing UI
        try:
            g_config.load()
        except Exception as e:
            logger.error(f"Failed to reload config on refresh: {e}")
        self._refresh_content()

    def _build_llm_profile_settings(self) -> ft.Container:
        """Build the specific UI for editing the active LLM profile."""
        if not g_config:
            return ft.Container(content=ft.Text("Config not loaded"))

        profile_names = list(g_config.llm.profiles.keys())
        active_profile_name = g_config.llm.active_profile
        active_profile = g_config.llm.profiles.get(active_profile_name)

        if not active_profile:
            return ft.Container(
                content=ft.Text(
                    f"Error: Active profile '{active_profile_name}' not found.",
                    color=ft.Colors.RED,
                )
            )

        def on_profile_change(e):
            new_profile = e.control.value
            if new_profile and new_profile != active_profile_name:
                self._set_config_value("llm.active_profile", new_profile)
                self._refresh_content()

        def on_add_profile(e):
            self._show_add_profile_dialog()

        profile_selector = ft.Container(
            content=ft.Row(
                [
                    ft.Text(t("settings_panel.provider"), size=13, color="#e0e0e0"),
                    ft.Dropdown(
                        value=active_profile_name,
                        options=[ft.dropdown.Option(name) for name in profile_names],
                        # width=150,
                        border_color="#404040",
                        focused_border_color="#3b82f6",
                        content_padding=10,
                        on_select=on_profile_change,
                    ),
                    # ft.Row(
                    #     [
                    #         ft.Dropdown(
                    #             value=active_profile_name,
                    #             options=[
                    #                 ft.dropdown.Option(name) for name in profile_names
                    #             ],
                    #             width=150,
                    #             border_color="#404040",
                    #             focused_border_color="#3b82f6",
                    #             content_padding=10,
                    #             on_select=on_profile_change,
                    #         ),
                    #         ft.IconButton(
                    #             icon=ft.icons.Icons.ADD,
                    #             tooltip=t("settings_panel.add_service"),
                    #             on_click=on_add_profile,
                    #             icon_color="#3b82f6",
                    #         ),
                    #     ],
                    #     spacing=4,
                    # ),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            padding=ft.Padding(12, 10, 12, 10),
        )

        profile_rows = []
        # Create controls for the fields in the active LLMProfile
        for field_name, field_value in active_profile.model_dump().items():
            # Skip complex types like dicts and lists for now
            if not isinstance(field_value, (str, int, float, bool, type(None))):
                continue

            key_path = f"llm.profiles.{active_profile_name}.{field_name}"
            field_type = self._get_field_type(field_value)
            title, _ = _get_settings_title(key_path)

            is_password = "api_key" in key_path.lower() or "token" in key_path.lower()

            def make_on_change(kp, ft_type):
                def on_change(e):
                    new_val = e.control.value
                    if ft_type == "bool":
                        new_val = bool(new_val)
                    elif ft_type == "int":
                        try:
                            new_val = int(new_val)
                        except (ValueError, TypeError):
                            self.show_error(f"Invalid integer for {kp}: {new_val}")
                            return
                    elif ft_type == "float":
                        try:
                            new_val = float(new_val)
                        except (ValueError, TypeError):
                            self.show_error(f"Invalid float for {kp}: {new_val}")
                            return
                    self._set_config_value(kp, new_val)

                return on_change

            if field_type == "bool":
                control = ft.Switch(
                    value=bool(field_value),
                    on_change=make_on_change(key_path, field_type),
                )
            else:
                control = ft.TextField(
                    value=str(field_value) if field_value is not None else "",
                    on_submit=make_on_change(key_path, field_type),
                    on_blur=make_on_change(key_path, field_type),
                    password=is_password,
                    can_reveal_password=is_password,
                    height=38,
                    content_padding=ft.Padding(left=10, top=4, right=10, bottom=4),
                    border_color="#404040",
                    focused_border_color="#3b82f6",
                )

            row = ft.Container(
                content=ft.Row(
                    [
                        ft.Text(title, size=13, color="#e0e0e0"),
                        control,
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                padding=ft.Padding(12, 10, 12, 10),
            )
            profile_rows.append(row)

        profile_selector_container = ft.Container(
            content=ft.Column(
                [profile_selector] + profile_rows,
                spacing=0,
            ),
            bgcolor="#2d2d2d",
            border=ft.Border(
                top=ft.BorderSide(1, "#383838"),
                bottom=ft.BorderSide(1, "#383838"),
                left=ft.BorderSide(1, "#383838"),
                right=ft.BorderSide(1, "#383838"),
            ),
            border_radius=ft.BorderRadius.all(6),
        )

        return ft.Container(
            content=ft.Column(
                [
                    ft.Container(
                        content=ft.Text(
                            t("settings_panel.llm_config"),
                            size=13,
                            weight=ft.FontWeight.W_500,
                            color="#a0a0a0",
                        ),
                        padding=ft.Padding(0, 0, 0, 8),
                        expand=True,
                    ),
                    profile_selector_container,
                ],
                spacing=4,
            ),
        )

    def _build_skills_section(self) -> ft.Container:
        """Build Skills settings section with consistent UI style."""
        if not g_config:
            return ft.Container(content=ft.Text("Config not loaded"))

        skills_config = g_config.skills
        if not skills_config:
            return ft.Container(content=ft.Text("Skills config not available"))

        skills_rows = []

        # Flatten skills config for display
        def add_setting_field(key_path: str, field_name: str, field_value: Any):
            field_type = self._get_field_type(field_value)
            title, _ = _get_settings_title(key_path)

            is_password = "api_key" in key_path.lower() or "token" in key_path.lower()

            def make_on_change(kp, ft_type):
                def on_change(e):
                    new_val = e.control.value
                    if ft_type == "bool":
                        new_val = bool(new_val)
                    elif ft_type == "int":
                        try:
                            new_val = int(new_val)
                        except (ValueError, TypeError):
                            return
                    elif ft_type == "float":
                        try:
                            new_val = float(new_val)
                        except (ValueError, TypeError):
                            return
                    self._set_config_value(kp, new_val)

                return on_change

            if field_type == "bool":
                control = ft.Switch(
                    value=bool(field_value),
                    on_change=make_on_change(key_path, field_type),
                )
            else:
                control = ft.TextField(
                    value=str(field_value) if field_value is not None else "",
                    on_submit=make_on_change(key_path, field_type),
                    on_blur=make_on_change(key_path, field_type),
                    password=is_password,
                    can_reveal_password=is_password,
                    height=38,
                    content_padding=ft.Padding(left=10, top=4, right=10, bottom=4),
                    border_color="#404040",
                    focused_border_color="#3b82f6",
                )

            row = ft.Container(
                content=ft.Row(
                    [
                        ft.Text(title, size=13, color="#e0e0e0"),
                        control,
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                padding=ft.Padding(12, 10, 12, 10),
            )
            skills_rows.append(row)

        # Add top-level skills fields
        top_level_fields = ["catalog_path", "github_token", "cloud_catalog_url"]
        for field_name in top_level_fields:
            if hasattr(skills_config, field_name):
                field_value = getattr(skills_config, field_name)
                if isinstance(field_value, (str, int, float, bool, type(None))):
                    key_path = f"skills.{field_name}"
                    add_setting_field(key_path, field_name, field_value)

        # Add retrieval, execution, strategy fields
        for section in ["retrieval", "execution", "strategy"]:
            section_obj = getattr(skills_config, section, None)
            if section_obj and hasattr(section_obj, "model_dump"):
                for field_name, field_value in section_obj.model_dump().items():
                    if isinstance(field_value, (str, int, float, bool, type(None))):
                        key_path = f"skills.{section}.{field_name}"
                        add_setting_field(key_path, field_name, field_value)

        skills_container = ft.Container(
            content=ft.Column(skills_rows, spacing=0),
            bgcolor="#2d2d2d",
            border=ft.Border(
                top=ft.BorderSide(1, "#383838"),
                bottom=ft.BorderSide(1, "#383838"),
                left=ft.BorderSide(1, "#383838"),
                right=ft.BorderSide(1, "#383838"),
            ),
            border_radius=ft.BorderRadius.all(6),
        )

        return ft.Container(
            content=ft.Column(
                [
                    ft.Container(
                        content=ft.Text(
                            t("settings_panel.skills"),
                            size=13,
                            weight=ft.FontWeight.W_500,
                            color="#a0a0a0",
                        ),
                        padding=ft.Padding(0, 0, 0, 8),
                        expand=True,
                    ),
                    skills_container,
                ],
                spacing=4,
            ),
        )

    def _build_generic_section(self, title: str, settings: list) -> ft.Container:
        """Build a generic settings section with consistent UI style."""
        rows = []

        for key_path, field_value, field_type in settings:
            title_text, _ = _get_settings_title(key_path)

            is_password = "api_key" in key_path.lower() or "token" in key_path.lower()

            def make_on_change(kp, ft_type):
                def on_change(e):
                    new_val = e.control.value
                    if ft_type == "bool":
                        new_val = bool(new_val)
                    elif ft_type == "int":
                        try:
                            new_val = int(new_val)
                        except (ValueError, TypeError):
                            return
                    elif ft_type == "float":
                        try:
                            new_val = float(new_val)
                        except (ValueError, TypeError):
                            return
                    self._set_config_value(kp, new_val)

                return on_change

            if field_type == "bool":
                control = ft.Switch(
                    value=bool(field_value),
                    on_change=make_on_change(key_path, field_type),
                )
            else:
                control = ft.TextField(
                    value=str(field_value) if field_value is not None else "",
                    on_submit=make_on_change(key_path, field_type),
                    on_blur=make_on_change(key_path, field_type),
                    password=is_password,
                    can_reveal_password=is_password,
                    height=38,
                    content_padding=ft.Padding(left=10, top=4, right=10, bottom=4),
                    border_color="#404040",
                    focused_border_color="#3b82f6",
                )

            row = ft.Container(
                content=ft.Row(
                    [
                        ft.Text(title_text, size=13, color="#e0e0e0"),
                        control,
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                ),
                padding=ft.Padding(12, 10, 12, 10),
            )
            rows.append(row)

        container = ft.Container(
            content=ft.Column(rows, spacing=0),
            bgcolor="#2d2d2d",
            border=ft.Border(
                top=ft.BorderSide(1, "#383838"),
                bottom=ft.BorderSide(1, "#383838"),
                left=ft.BorderSide(1, "#383838"),
                right=ft.BorderSide(1, "#383838"),
            ),
            border_radius=ft.BorderRadius.all(6),
        )

        return ft.Container(
            content=ft.Column(
                [
                    ft.Container(
                        content=ft.Text(
                            title,
                            size=13,
                            weight=ft.FontWeight.W_500,
                            color="#a0a0a0",
                        ),
                        padding=ft.Padding(0, 0, 0, 8),
                        expand=True,
                    ),
                    container,
                ],
                spacing=4,
            ),
        )

    def _show_add_profile_dialog(self):
        """Show dialog to add a new LLM profile."""
        if not g_config:
            g_config.load()

        profile_name_field = ft.TextField(
            label=t("settings_panel.service_name"),
            hint_text=t("settings_panel.service_name_hint"),
            width=300,
            border_color="#404040",
            focused_border_color="#3b82f6",
        )

        model_field = ft.TextField(
            label=t("settings_panel.model"),
            hint_text=t("settings_panel.model_hint"),
            width=300,
            border_color="#404040",
            focused_border_color="#3b82f6",
        )

        api_key_field = ft.TextField(
            label=t("settings_panel.api_key"),
            hint_text=t("settings_panel.api_key_hint"),
            width=300,
            password=True,
            can_reveal_password=True,
            border_color="#404040",
            focused_border_color="#3b82f6",
        )

        # Error message display - shown at bottom of dialog when validation fails
        error_text = ft.Text(
            "",
            color=ft.Colors.RED_400,
            size=12,
            weight=ft.FontWeight.W_500,
        )

        base_url_field = ft.TextField(
            label=t("settings_panel.base_url"),
            hint_text=t("settings_panel.base_url_hint"),
            width=300,
            border_color="#404040",
            focused_border_color="#3b82f6",
        )

        # Container for error text - hidden by default
        error_container = ft.Container(
            content=error_text,
            padding=ft.Padding(top=8, left=0, right=0, bottom=0),
            visible=False,
        )

        add_dialog = ft.AlertDialog(
            title=ft.Text(t("settings_panel.add_service_title")),
            content=ft.Column(
                [
                    profile_name_field,
                    model_field,
                    api_key_field,
                    base_url_field,
                    error_container,
                ],
                spacing=12,
                tight=True,
            ),
            actions_alignment=ft.MainAxisAlignment.END,
            shape=ft.RoundedRectangleBorder(radius=8),
        )

        import re

        # Clear error message when user starts typing
        def clear_error(e):
            error_text.value = ""
            error_container.visible = False
            error_text.update()

        # Validate provider name - only allow letters, numbers, hyphens and underscores
        def validate_profile_name(e):
            clear_error(e)
            # Filter out special characters in real-time
            value = e.control.value
            # Allow: letters, numbers, spaces, hyphens, underscores
            # Block: other special characters like @, #, $, %, etc.
            filtered = re.sub(r"[^\w\s-]", "", value)
            if filtered != value:
                e.control.value = filtered
                e.control.update()

        profile_name_field.on_change = validate_profile_name
        model_field.on_change = clear_error
        api_key_field.on_change = clear_error
        base_url_field.on_change = clear_error

        # Save button handler with validation
        def on_save(e):
            profile_name = profile_name_field.value.strip()
            if not profile_name:
                error_text.value = t("settings_panel.validation.service_name_required")
                error_container.visible = True
                error_text.update()
                return

            if profile_name in g_config.llm.profiles:
                error_text.value = t("settings_panel.validation.service_name_exists")
                error_container.visible = True
                error_text.update()
                return

            # Check for special characters
            if not re.match(r"^[\w\s-]+$", profile_name):
                error_text.value = t(
                    "settings_panel.validation.service_name_invalid_chars"
                )
                error_container.visible = True
                error_text.update()
                return

            model = model_field.value.strip() if model_field.value else ""
            if not model:
                error_text.value = t("settings_panel.validation.model_required")
                error_container.visible = True
                error_text.update()
                return

            api_key = api_key_field.value.strip() if api_key_field.value else ""
            if not api_key:
                error_text.value = t("settings_panel.validation.api_key_required")
                error_container.visible = True
                error_text.update()
                return

            base_url = base_url_field.value.strip() if base_url_field.value else ""
            if not base_url:
                error_text.value = t("settings_panel.validation.base_url_required")
                error_container.visible = True
                error_text.update()
                return

            if not (base_url.startswith("http://") or base_url.startswith("https://")):
                error_text.value = t("settings_panel.validation.base_url_invalid")
                error_container.visible = True
                error_text.update()
                return

            try:
                new_profile = {
                    "model": profile_name + "/" + model,
                    "api_key": api_key,
                    "base_url": base_url,
                    "extra_headers": {},
                    "extra_body": {},
                    "context_window": 128000,
                    "max_tokens": 8192,
                    "temperature": 0.5,
                    "timeout": 120,
                }

                g_config.set(f"llm.profiles.{profile_name}", new_profile, save=True)
                g_config.set("llm.active_profile", profile_name)
                self._refresh_content()
                add_dialog.open = False
                self.page.update()
                self.show_snackbar(t("settings_panel.service_added", name=profile_name))

                # 触发保存回调，通知外部刷新UI（如模型选择器）
                print(f"[SettingsPanel] Triggering on_save_callback...")
                if self.on_save_callback:
                    try:
                        self.on_save_callback()
                        print(
                            f"[SettingsPanel] on_save_callback completed successfully"
                        )
                    except Exception as e:
                        print(f"[SettingsPanel] ERROR in on_save_callback: {e}")
                        import traceback

                        traceback.print_exc()
            except Exception as ex:
                print(f"[SettingsPanel] ERROR in on_save: {ex}")
                import traceback

                traceback.print_exc()
                self.show_error(t("settings_panel.add_failed", error=str(ex)))

        def on_cancel(e):
            add_dialog.open = False
            self.page.update()

        add_dialog.actions = [
            ft.TextButton(t("settings_panel.cancel"), on_click=on_cancel),
            ft.TextButton(t("settings_panel.add"), on_click=on_save),
        ]
        add_dialog.open = True
        self.page.overlay.append(add_dialog)
        self.page.update()

    # ==================== Update Feature ====================

    async def _handle_update_check(self):
        """Handle update check action using AutoUpdateManager."""
        if not self._update_manager:
            self.show_error("Update manager not initialized")
            return

        self._update_progress.visible = True
        self._update_button.visible = False
        self.page.update()

        try:
            # Check if we already have a cached update
            if self._update_manager.has_cached_update:
                logger.info("[SettingsPanel] Found cached update")
                if self._update_manager.current_update:
                    self._show_install_confirmation_dialog(
                        self._update_manager.current_update
                    )
                cache_version = ""
                if self._update_manager._cache:
                    cache_version = self._update_manager._cache.version
                self._update_status_text.value = t(
                    "settings_panel.update_available",
                    version=cache_version,
                )
                self._update_status_text.color = ft.Colors.AMBER_400
                return

            # Check for updates using manager
            update_info = await self._update_manager.check_for_update()

            if update_info:
                self._show_update_dialog(update_info)
            else:
                current_ver = ""
                if self._update_manager:
                    current_ver = self._update_manager._get_current_version()
                self._update_status_text.value = f"v{current_ver}"
                self._update_status_text.color = ft.Colors.GREEN_400
                self.show_snackbar(t("settings_panel.no_update"))

        except Exception as e:
            self._update_status_text.value = t("settings_panel.check_failed")
            self._update_status_text.color = ft.Colors.RED_400
            self.show_error(f"Failed to check updates: {str(e)}")
        finally:
            self._update_progress.visible = False
            self._update_button.visible = True
            self.page.update()

    def _show_update_dialog(self, update_info: UpdateInfo):
        """Show update available dialog with download option."""
        version = update_info.version
        current = update_info.current_version

        def on_download_click(e):
            """Start download and installation process."""
            dialog.open = False
            self.page.update()
            # Start download in background using manager
            asyncio.create_task(self._download_and_install_update(update_info))

        def on_close(e):
            dialog.open = False
            self.page.update()

        platform_name = platform.system()

        dialog_content = ft.Column(
            [
                ft.Text(f"New version: {version}", size=14),
                ft.Text(f"Current: {current}", size=12, color=ft.Colors.GREY_500),
                ft.Divider(height=16, color=ft.Colors.TRANSPARENT),
                ft.Text(
                    f"A new version is available for {platform_name}. Click 'Update Now' to download and install automatically.",
                    size=13,
                ),
            ],
            spacing=8,
            tight=True,
        )
        dialog_actions = [
            ft.TextButton("Later", on_click=on_close),
            ft.ElevatedButton(
                "Update Now",
                icon=ft.icons.Icons.DOWNLOAD,
                on_click=on_download_click,
            ),
        ]

        dialog = ft.AlertDialog(
            title=ft.Text("Update Available", size=16, weight=ft.FontWeight.BOLD),
            content=dialog_content,
            actions=dialog_actions,
            shape=ft.RoundedRectangleBorder(radius=8),
        )

        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()

    def _show_install_confirmation_dialog(self, update_info: UpdateInfo):
        """Show install confirmation dialog after download completes."""

        def on_confirm(e):
            dialog.open = False
            self.page.update()
            # Start installation
            asyncio.create_task(self._do_install_update())

        def on_cancel(e):
            dialog.open = False
            self.page.update()
            self.show_snackbar("Update will be installed later")

        dialog = ft.AlertDialog(
            title=ft.Text(
                t("update.confirm_title"), size=16, weight=ft.FontWeight.BOLD
            ),
            content=ft.Text(
                t(
                    "update.confirm_desc",
                    version=update_info.version,
                    current=update_info.current_version,
                )
            ),
            actions=[
                ft.TextButton(t("common.later"), on_click=on_cancel),
                ft.ElevatedButton(
                    t("update.restart_and_install"),
                    icon=ft.icons.Icons.RESTART_ALT,
                    on_click=on_confirm,
                ),
            ],
            shape=ft.RoundedRectangleBorder(radius=8),
        )

        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()

    async def _do_install_update(self):
        """Execute installation using AutoUpdateManager."""
        if not self._update_manager:
            self.show_error("Update manager not initialized")
            return

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

        try:
            # Perform installation
            success = await self._update_manager.install_update(
                page=self.page,
                on_complete=lambda: self._on_install_complete(),
            )

            installing_dialog.open = False
            self.page.update()

            if success:
                # Close app for restart
                self.show_snackbar(t("update.install_complete"))
                await asyncio.sleep(1)
                self.page.window.close()
            else:
                self.show_error("Installation failed")

        except Exception as e:
            installing_dialog.open = False
            self.page.update()
            logger.error(f"[SettingsPanel] Install error: {e}", exc_info=True)
            self.show_error(f"Installation failed: {str(e)}")

    def _on_install_complete(self):
        """Called when installation completes."""
        self.show_snackbar(t("update.install_complete"))

    async def _download_and_install_update(self, update_info: UpdateInfo):
        """Download and install update using AutoUpdateManager."""
        if not self._update_manager:
            self.show_error("Update manager not initialized")
            return

        try:
            # Download update
            success = await self._update_manager.download_update(update_info)

            if success:
                # Download complete, show install confirmation
                logger.info("[SettingsPanel] Download complete, showing install dialog")
                self._show_install_confirmation_dialog(update_info)
            else:
                self.show_error("Download failed")

        except asyncio.CancelledError:
            logger.warning("[SettingsPanel] Update cancelled by user")
            self.show_snackbar("Update cancelled")
        except Exception as e:
            logger.exception(f"[SettingsPanel] Update failed: {e}")
            self.show_error(f"Update failed: {str(e)}")
