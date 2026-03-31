from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

import flet as ft

from middleware.config import g_config
from gui.i18n import t
from gui.modules.auto_update_manager import (
    AutoUpdateManager,
    CallbackGroup,
    UpdateStatus,
    UpdateInfo,
    DownloadProgress,
)

if TYPE_CHECKING:
    from gui.modules.auth_service import AuthService


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
    # Provider fields
    "search": "搜索",
    "skills": "技能",
    "storage": "存储",
    "advanced": "高级",
    # IM fields
    "enabled": "启用",
    "app_id": "App ID",
    "app_secret": "App Secret",
    "app_key": "App Key",
    "corp_id": "Corp ID",
    "agent_id": "Agent ID",
    "secret": "Secret",
    "bot_id": "Bot ID",
    "encrypt_key": "加密密钥",
    "verification_token": "验证令牌",
    "webhook_url": "Webhook URL",
    "base_url": "Base URL",
    "platform": "平台",
    "token": "Token",
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
        update_manager: AutoUpdateManager | None = None,
        auth_service: AuthService | None = None,
        on_login_click: Callable[[], None] | None = None,
        on_logout_click: Callable[[], None] | None = None,
    ):
        self.page = page
        self.show_error = show_error
        self.show_snackbar = show_snackbar
        self.on_save_callback = on_save_callback
        self.dialog = None
        # 使用内部键（非翻译文本）来标识当前分类
        self.current_category = "general"
        self.settings_data = {}
        self._auth_service = auth_service
        self._on_login_click = on_login_click
        self._on_logout_click = on_logout_click
        self._update_status_reset_task: asyncio.Task | None = None
        self._sidebar_container: ft.Container | None = None
        self._settings_content_inner: ft.Container | None = None
        self._content_title: ft.Text | None = None

        self._update_manager: AutoUpdateManager | None = None
        self._update_listener: CallbackGroup | None = None
        self._init_update_manager(update_manager)

        # 微信会话过期状态追踪
        self._wechat_session_expired = False

        # 注册语言切换观察者
        from gui.i18n import add_observer

        add_observer(self._on_language_changed)

    def _on_language_changed(self, new_lang: str):
        """语言切换时的回调 - 刷新设置面板"""
        if self.dialog and self.dialog.open:
            self._refresh_content()

    def _close_dialog(self):
        """Close the settings dialog."""
        if self.dialog and self.dialog.open:
            self.dialog.open = False
            if self.page:
                self.page.update()

    def _on_dialog_dismiss(self):
        """Clean up when the settings dialog is closed."""
        if self._update_manager and self._update_listener:
            self._update_manager.remove_listener(self._update_listener)
            self._update_listener = None

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
            self._sidebar_container = ft.Container(
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
                ),
            )

            self._content_title = ft.Text(
                t(f"settings_panel.categories.{self.current_category}"),
                size=16,
                weight=ft.FontWeight.W_600,
                color="#e0e0e0",
            )

            self._settings_content_inner = ft.Container(
                content=settings_content_obj,
                padding=ft.Padding(0, 0, 0, 20),
                expand=True,
            )

            settings_content_container = ft.Container(
                content=ft.Column(
                    [
                        self._content_title,
                        self._settings_content_inner,
                    ],
                    spacing=12,
                    expand=True,
                ),
                padding=ft.Padding(24, 20, 24, 0),
                bgcolor="#1e1e1e",
                expand=True,
            )

            logger.info("[SettingsPanel] Creating dialog...")

            divider = ft.Container(width=1, bgcolor="#383838")

            content = ft.Row(
                [
                    self._sidebar_container,
                    divider,
                    settings_content_container,
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
                content=ft.Stack(
                    [
                        content,
                        ft.Container(
                            content=close_button,
                            right=8,
                            top=8,
                        ),
                    ],
                ),
                width=dialog_width,
                height=dialog_height,
                border_radius=8,
                border=ft.border.all(0.5, "#000000"),
                clip_behavior=ft.ClipBehavior.HARD_EDGE,
            )

            self.dialog = ft.AlertDialog(
                content=dialog_container,
                content_padding=0,
                bgcolor="#00000000",
                shape=ft.RoundedRectangleBorder(radius=8),
                on_dismiss=lambda _: self._on_dialog_dismiss(),
            )
            self.dialog.open = True
            self.page.overlay.append(self.dialog)
            self.page.update()

        except Exception as e:
            self.show_error(f"Failed to open settings: {str(e)}")

    def _init_update_manager(self, update_manager: AutoUpdateManager | None = None):
        """Initialize auto update manager with callbacks.

        Args:
            update_manager: Optional existing AutoUpdateManager to reuse.
                           If not provided, creates a new one.
        """
        if update_manager is not None:
            self._update_manager = update_manager
            logger.info("[SettingsPanel] Reusing existing AutoUpdateManager")
        else:
            self._update_manager = AutoUpdateManager()
            logger.info("[SettingsPanel] Created new AutoUpdateManager")

        self._update_listener = self._update_manager.add_listener(
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
            self._update_status_text.value = t("update.downloading")
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
            if self._update_manager and self._update_manager.current_update:
                version_str = self._update_manager.current_update.version
                self._update_status_text.value = t(
                    "settings_panel.update_available",
                    version=version_str,
                )
                self._update_status_text.color = ft.Colors.AMBER_400
            else:
                self._update_status_text.value = t("settings_panel.check_failed")
                self._update_status_text.color = ft.Colors.RED_400
        elif status == UpdateStatus.CANCELLED:
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
            else:
                current_ver = ""
                if self._update_manager:
                    current_ver = self._update_manager._get_current_version()
                self._update_status_text.value = f"v{current_ver}"
                self._update_status_text.color = "#a0a0a0"

        self.page.update()

    def _on_update_progress(self, progress: DownloadProgress):
        """Handle download progress updates."""
        # Progress is handled by the manager's UI
        pass

    def _on_download_complete(self, update_info: UpdateInfo):
        """Handle download completion.

        Only updates status text here. The install confirmation dialog is shown
        by _download_and_install_update (manual) or _handle_update_check (cached),
        avoiding duplicate popups.
        """
        logger.info(f"[SettingsPanel] Download complete: {update_info.version}")

    def _on_update_error(self, message: str):
        """Handle update errors."""
        logger.error(f"[SettingsPanel] Update error: {message}")
        self._show_update_blocked_message(message)
        if self._update_manager and self._update_manager.current_update:
            version_str = self._update_manager.current_update.version
            self._update_status_text.value = t(
                "settings_panel.update_available",
                version=version_str,
            )
            self._update_status_text.color = ft.Colors.AMBER_400
        else:
            self._update_status_text.value = t("settings_panel.check_failed")
            self._update_status_text.color = ft.Colors.RED_400
        self.page.update()

    def _show_update_blocked_message(self, message: str, version: str | None = None):
        """Show a visible update-blocked message inside the settings flow."""
        self._cancel_update_status_reset_task()
        if hasattr(self, "_update_status_text") and self._update_status_text:
            if version:
                self._update_status_text.value = t(
                    "settings_panel.update_available",
                    version=version,
                )
                self._update_status_text.color = ft.Colors.AMBER_400
            else:
                self._update_status_text.value = message
                self._update_status_text.color = ft.Colors.RED_400
        if hasattr(self, "_update_progress") and self._update_progress:
            self._update_progress.visible = False
        if hasattr(self, "_update_button") and self._update_button:
            self._update_button.visible = True
        self.show_snackbar(message, type="error", duration=5000)
        self.page.update()

    def _cancel_update_status_reset_task(self):
        """Cancel any pending delayed restore of the update status text."""
        if self._update_status_reset_task and not self._update_status_reset_task.done():
            self._update_status_reset_task.cancel()
        self._update_status_reset_task = None

    async def _restore_update_status_after_delay(self, delay_seconds: float = 3.0):
        """Restore the update status text back to the current version after a delay."""
        try:
            await asyncio.sleep(delay_seconds)
            if not self._update_manager or not hasattr(self, "_update_status_text"):
                return

            current_ver = self._update_manager._get_current_version()
            self._update_status_text.value = f"v{current_ver}"
            self._update_status_text.color = "#a0a0a0"
            self.page.update()
        except asyncio.CancelledError:
            return
        finally:
            self._update_status_reset_task = None

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

    def _set_config_value(self, key_path: str, value: Any, refresh: bool = True):
        """Set config value by key path and save immediately."""
        try:
            g_config.set(key_path, value, save=False)
            g_config.save()
            # Update local settings_data immediately
            self.settings_data = g_config.to_json_dict()
            self.show_snackbar(t("settings_panel.messages.saved", key_path=key_path))
            if self.on_save_callback:
                self.on_save_callback()
            if refresh:
                self._refresh_content()
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
            "im": "settings_panel.categories.im",
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

    def _refresh_content(self, reload_config: bool = False):
        """Refresh the content area based on current category.

        Args:
            reload_config: Whether to reload config from file. Set to True when
                          config might have been changed externally.
        """
        try:
            # Only reload config if explicitly requested
            # This avoids race conditions where we overwrite unsaved changes
            if reload_config and g_config:
                g_config.load()

            # Always sync with g_config to ensure we have the latest data
            if g_config:
                self.settings_data = g_config.to_json_dict()

            if not (self.dialog and self.dialog.open):
                return

            # Update sidebar
            if self._sidebar_container and self._sidebar_container.content:
                sidebar_content = self._sidebar_container.content
                sidebar_controls = sidebar_content.controls
                if isinstance(sidebar_controls, list) and len(sidebar_controls) > 2:
                    sidebar_controls[2] = self._build_category_sidebar(
                        list(self._get_categories().keys())
                    )

            # Update content title
            if self._content_title:
                self._content_title.value = t(
                    f"settings_panel.categories.{self.current_category}"
                )

            # Update settings content
            if self._settings_content_inner:
                self._settings_content_inner.content = (
                    self._build_settings_content()
                )

        except Exception as e:
            logger.error(f"Failed to build settings UI: {e}", exc_info=True)
            # Try to display a user-friendly error in the settings panel
            if self._settings_content_inner:
                self._settings_content_inner.content = ft.Column(
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
            "im": [],
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
                elif key_lower.startswith(
                    ("im.", "feishu", "dingtalk", "wecom", "lark")
                ):
                    if isinstance(value, (str, int, float, bool)):
                        categories["im"].append(
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
            user_info = self._build_user_info_section()
            controls.append(user_info)
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
            controls.append(self._build_generic_section(None, settings))

            return ft.Column(
                controls,
                spacing=16,
                scroll=ft.ScrollMode.AUTO,
                expand=True,
            )
        # ----------------------------------------------------------

        # --- Memento-S Change: Use dedicated IM settings builder ---
        if self.current_category == "im":
            controls.append(self._build_im_section())

            return ft.Column(
                controls,
                spacing=16,
                scroll=ft.ScrollMode.AUTO,
                expand=True,
            )
        # ----------------------------------------------------------

        # --- Memento-S Change: Use dedicated Advanced settings builder ---
        if self.current_category == "advanced":
            controls.append(self._build_generic_section(None, settings))

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

    def _build_user_info_section(self) -> ft.Container:
        """Build user info section at the top of General settings."""
        logged_in = self._auth_service and self._auth_service.is_logged_in
        display_name = self._auth_service.display_name if self._auth_service else ""

        if logged_in and display_name:
            user_info = self._auth_service.user_info if self._auth_service else {}
            account_id = user_info.get("email") or user_info.get("phone") or ""

            logout_btn = ft.OutlinedButton(
                t("auth.logout"),
                icon=ft.Icons.LOGOUT,
                icon_color=ft.Colors.RED_400,
                style=ft.ButtonStyle(
                    color=ft.Colors.RED_400,
                    side=ft.BorderSide(1, "#504040"),
                    shape=ft.RoundedRectangleBorder(radius=6),
                    padding=ft.Padding(12, 6, 12, 6),
                    overlay_color=ft.Colors.with_opacity(0.1, ft.Colors.RED_400),
                    mouse_cursor=ft.MouseCursor.CLICK,
                ),
                height=32,
                on_click=lambda e: (
                    self._on_logout_click() if self._on_logout_click else None
                ),
            )

            content = ft.Row(
                [
                    ft.Icon(ft.Icons.ACCOUNT_CIRCLE, color=ft.Colors.BLUE_400, size=36),
                    ft.Column(
                        [
                            ft.Text(
                                display_name,
                                size=14,
                                weight=ft.FontWeight.W_500,
                                color="#e0e0e0",
                                max_lines=1,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                            ft.Text(
                                account_id if account_id else display_name,
                                size=11,
                                color="#808080",
                                max_lines=1,
                                overflow=ft.TextOverflow.ELLIPSIS,
                            ),
                        ],
                        spacing=2,
                        expand=True,
                    ),
                    logout_btn,
                ],
                spacing=12,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )
        else:
            login_btn = ft.ElevatedButton(
                t("auth.login"),
                icon=ft.Icons.LOGIN,
                style=ft.ButtonStyle(
                    bgcolor="#3b82f6",
                    color="#ffffff",
                    shape=ft.RoundedRectangleBorder(radius=6),
                    padding=ft.Padding(16, 6, 16, 6),
                    overlay_color=ft.Colors.with_opacity(0.15, ft.Colors.WHITE),
                    mouse_cursor=ft.MouseCursor.CLICK,
                ),
                height=32,
                on_click=lambda e: (
                    self._on_login_click() if self._on_login_click else None
                ),
            )

            content = ft.Row(
                [
                    ft.Icon(
                        ft.Icons.ACCOUNT_CIRCLE_OUTLINED,
                        color="#606060",
                        size=36,
                    ),
                    ft.Column(
                        [
                            ft.Text(
                                t("settings_panel.not_logged_in"),
                                size=14,
                                weight=ft.FontWeight.W_500,
                                color="#a0a0a0",
                            ),
                            # ft.Text(
                            #     t("settings_panel.login_hint"),
                            #     size=11,
                            #     color="#606060",
                            # ),
                        ],
                        spacing=2,
                        expand=True,
                    ),
                    login_btn,
                ],
                spacing=12,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )

        return ft.Container(
            content=ft.Column(
                [
                    ft.Container(
                        content=ft.Text(
                            t("settings_panel.account"),
                            size=13,
                            weight=ft.FontWeight.W_500,
                            color="#a0a0a0",
                        ),
                        padding=ft.Padding(0, 0, 0, 8),
                    ),
                    ft.Container(
                        content=content,
                        bgcolor="#2d2d2d",
                        border=ft.Border(
                            top=ft.BorderSide(1, "#383838"),
                            bottom=ft.BorderSide(1, "#383838"),
                            left=ft.BorderSide(1, "#383838"),
                            right=ft.BorderSide(1, "#383838"),
                        ),
                        border_radius=ft.BorderRadius.all(6),
                        padding=ft.Padding(16, 12, 16, 12),
                    ),
                ],
                spacing=4,
            ),
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

        import json

        config_path = g_config.user_config_path

        # 读取当前配置文件内容
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_content = f.read()
        except Exception as e:
            config_content = f"Error reading config: {e}"

        # 计算 TextField 的高度（需要固定高度才能在内部滚动）
        # 层级从外到内:
        #   settings_content_container padding: top=20, bottom=0 → 20px
        #   Column spacing=12（标题与 _settings_content_inner 之间） → 12px
        #   标题 Text(size=16) → ~22px
        #   _settings_content_inner padding: top=0, bottom=20 → 20px
        #   内部 Column spacing=8, 3 个子元素, 2 个 spacing:
        #     路径 Text(size=11) → ~15px
        #     spacing → 8px
        #     config_editor（待计算）
        #     spacing → 8px
        #     按钮行 Row → ~36px
        # 总固定占用: 20 + 22 + 12 + 20 + 15 + 8 + 8 + 36 = 141px
        fixed_height = 141
        available_height = self._dialog_height if hasattr(self, "_dialog_height") and self._dialog_height else 500
        text_field_height = max(200, available_height - fixed_height)

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

                # 调用 replace_user_config 保存配置（仅用户域）
                error = g_config.replace_user_config(config_dict)
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

        # Get directory paths from config
        skills_dir = g_config.paths.skills_dir
        workspace_dir = g_config.paths.workspace_dir

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

    def _build_generic_section(
        self, title: str | None, settings: list
    ) -> ft.Container:
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

        section_controls = [container]
        if title:
            section_controls.insert(
                0,
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
            )

        return ft.Container(
            content=ft.Column(
                section_controls,
                spacing=4,
            ),
        )

    def _show_add_profile_dialog(self):
        """Show dialog to add a new LLM profile."""
        if not g_config:
            g_config.load()

        profile_name_field = ft.TextField(
            label="Profile Name",
            hint_text="输入Profile名称",
            width=300,
            border_color="#404040",
            focused_border_color="#3b82f6",
        )

        provider_name_field = ft.TextField(
            label="Provider Name",
            hint_text="输入Provider名称",
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

        max_tokens_field = ft.TextField(
            label="Max Tokens",
            hint_text="例如: 8192",
            width=300,
            border_color="#404040",
            focused_border_color="#3b82f6",
            keyboard_type=ft.KeyboardType.NUMBER,
            value="8192",
        )

        temperature_field = ft.TextField(
            label="Temperature",
            hint_text="例如: 0.5 (0.0-2.0)",
            width=300,
            border_color="#404040",
            focused_border_color="#3b82f6",
            keyboard_type=ft.KeyboardType.NUMBER,
            value="0.5",
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
                    provider_name_field,
                    model_field,
                    api_key_field,
                    base_url_field,
                    max_tokens_field,
                    temperature_field,
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
        provider_name_field.on_change = clear_error
        model_field.on_change = clear_error
        api_key_field.on_change = clear_error
        base_url_field.on_change = clear_error
        max_tokens_field.on_change = clear_error
        temperature_field.on_change = clear_error

        # Save button handler with validation
        def on_save(e):
            profile_name = profile_name_field.value.strip()
            if not profile_name:
                error_text.value = "Profile名称不能为空"
                error_container.visible = True
                error_text.update()
                return

            if profile_name in g_config.llm.profiles:
                error_text.value = t("settings_panel.validation.service_name_exists")
                error_container.visible = True
                error_text.update()
                return

            # Check for special characters in profile name
            if not re.match(r"^[\w\s-]+$", profile_name):
                error_text.value = "Profile名称只能包含字母、数字、空格、下划线和连字符"
                error_container.visible = True
                error_text.update()
                return

            provider_name = provider_name_field.value.strip()
            if not provider_name:
                error_text.value = "Provider名称不能为空"
                error_container.visible = True
                error_text.update()
                return
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

            # Parse numeric values
            try:
                max_tokens = (
                    int(max_tokens_field.value.strip())
                    if max_tokens_field.value
                    else 8192
                )
            except ValueError:
                max_tokens = 8192

            try:
                temperature = (
                    float(temperature_field.value.strip())
                    if temperature_field.value
                    else 0.5
                )
            except ValueError:
                temperature = 0.5

            try:
                new_profile = {
                    "model": model,
                    "api_key": api_key,
                    "base_url": base_url,
                    "litellm_provider": provider_name,
                    "extra_headers": {},
                    "extra_body": {},
                    "context_window": 128000,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "timeout": 120,
                }

                g_config.set(f"llm.profiles.{profile_name}", new_profile, save=False)
                g_config.set("llm.active_profile", profile_name, save=False)
                g_config.save()
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
            self.show_error(t("update.manager_not_initialized"))
            return

        if self._update_manager.status == UpdateStatus.DOWNLOADING:
            self._cancel_update_status_reset_task()
            logger.info(
                "[SettingsPanel] Download in progress, re-showing progress dialog"
            )
            self._update_status_text.value = t("update.downloading")
            self._update_status_text.color = ft.Colors.BLUE_400
            self._update_progress.visible = True
            self._update_button.visible = False
            self._update_manager._is_auto_check = False
            self._update_manager._set_status(UpdateStatus.DOWNLOADING)
            self.page.update()
            return

        self._update_progress.visible = True
        self._update_button.visible = False
        self._cancel_update_status_reset_task()
        self.page.update()

        try:
            self._update_manager._is_auto_check = False

            # Check if we already have a cached update
            if self._update_manager.has_cached_update:
                if self._update_manager._is_cache_stale():
                    logger.info("[SettingsPanel] Cached update is stale, re-checking")
                    self._update_manager.clear_cache()
                else:
                    logger.info("[SettingsPanel] Found cached update")
                    self._update_manager._mark_checked()
                    update_info = self._update_manager.current_update
                    if not update_info and self._update_manager._cache:
                        update_info = UpdateInfo(
                            version=self._update_manager._cache.version,
                            current_version=self._update_manager._get_current_version(),
                            download_url="",
                            force_update=self._update_manager._cache.force_update,
                        )
                        self._update_manager._current_update = update_info
                    cache_version = ""
                    if self._update_manager._cache:
                        cache_version = self._update_manager._cache.version
                    self._update_status_text.value = t(
                        "settings_panel.update_available",
                        version=cache_version,
                    )
                    self._update_status_text.color = ft.Colors.AMBER_400
                    if update_info:
                        blocked_reason = (
                            self._update_manager.get_macos_update_block_reason()
                        )
                        if blocked_reason:
                            self._show_update_blocked_message(
                                blocked_reason,
                                version=update_info.version,
                            )
                            return
                        if self._update_manager.is_force_update:
                            self._show_force_update_dialog(update_info)
                        else:
                            self._show_install_confirmation_dialog(update_info)
                    return

            update_info = await self._update_manager.manual_check_for_update()
            logger.info(f"update_info: {update_info}")

            if update_info:
                blocked_reason = self._update_manager.get_macos_update_block_reason()
                if blocked_reason:
                    self._show_update_blocked_message(
                        blocked_reason,
                        version=update_info.version,
                    )
                    return
                if self._update_manager.status == UpdateStatus.DOWNLOADED:
                    if self._update_manager.is_force_update:
                        self._show_force_update_dialog(update_info)
                    else:
                        self._show_install_confirmation_dialog(update_info)
                else:
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
            self.show_error(t("update.check_failed_detail", error=str(e)))
            self._update_status_reset_task = asyncio.create_task(
                self._restore_update_status_after_delay(3.0)
            )
        finally:
            self._update_progress.visible = False
            self._update_button.visible = True
            self.page.update()

    def _show_update_dialog(self, update_info: UpdateInfo):
        """Show update available dialog with download option."""
        version = update_info.version
        current = update_info.current_version
        release_notes = update_info.release_notes

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
                ft.Text(t("update.new_version", version=version), size=14),
                ft.Text(
                    t("update.current_version", version=current),
                    size=12,
                    color=ft.Colors.GREY_500,
                ),
                ft.Text(
                    t("update.release_notes_label", notes=release_notes),
                    size=12,
                    color=ft.Colors.GREY_500,
                ),
                ft.Divider(height=16, color=ft.Colors.TRANSPARENT),
                ft.Text(
                    t("update.update_hint", platform=platform_name),
                    size=13,
                ),
            ],
            spacing=8,
            tight=True,
            scroll=ft.ScrollMode.AUTO,
        )
        dialog_actions = [
            ft.TextButton(t("common.later"), on_click=on_close),
            ft.ElevatedButton(
                t("update.update_now"),
                icon=ft.icons.Icons.DOWNLOAD,
                on_click=on_download_click,
            ),
        ]

        dialog = ft.AlertDialog(
            title=ft.Text(t("update.available"), size=16, weight=ft.FontWeight.BOLD),
            content=dialog_content,
            actions=dialog_actions,
            shape=ft.RoundedRectangleBorder(radius=8),
        )

        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()

    def _show_force_update_dialog(self, update_info: UpdateInfo):
        """Show a modal, non-dismissible force-update dialog.

        install_confirmation is bypassed: installation starts immediately on confirm.
        """

        def on_install_click(e):
            dialog.open = False
            self.page.update()
            asyncio.create_task(self._do_install_update())

        dialog = ft.AlertDialog(
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
            shape=ft.RoundedRectangleBorder(radius=8),
        )

        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()

    def _show_install_confirmation_dialog(self, update_info: UpdateInfo):
        """Show install confirmation dialog after download completes.

        install_confirmation config only affects the auto-update flow (UpdateNotifier).
        Manual checks always show this dialog regardless of the config value,
        since the user explicitly initiated the action and should confirm before restart.
        """

        def on_confirm(e):
            dialog.open = False
            self.page.update()
            # Start installation
            asyncio.create_task(self._do_install_update())

        def on_cancel(e):
            dialog.open = False
            self.page.update()
            self.show_snackbar(t("update.install_later_hint"))

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
            self.show_error(t("update.manager_not_initialized"))
            return

        version = None
        if self._update_manager and self._update_manager.current_update:
            version = self._update_manager.current_update.version
        elif self._update_manager and self._update_manager._cache:
            version = self._update_manager._cache.version

        if version:
            blocked_reason = self._update_manager.get_macos_update_block_reason()
            if blocked_reason:
                self._show_update_blocked_message(blocked_reason, version=version)
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
                await self.page.window.close()
            else:
                self.show_error(t("update.install_failed"))

        except Exception as e:
            installing_dialog.open = False
            self.page.update()
            logger.error(f"[SettingsPanel] Install error: {e}", exc_info=True)
            self.show_error(f"Installation failed: {str(e)}")

    def _on_install_complete(self):
        """Called when installer script is launched successfully."""
        self.show_snackbar(t("update.install_complete"))

    async def _download_and_install_update(self, update_info: UpdateInfo):
        """Download and install update using AutoUpdateManager."""
        if not self._update_manager:
            self.show_error(t("update.manager_not_initialized"))
            return

        try:
            # Download update
            success = await self._update_manager.download_update(update_info)

            if success:
                # Download complete, show install confirmation
                logger.info("[SettingsPanel] Download complete, showing install dialog")
                self._show_install_confirmation_dialog(update_info)
            else:
                self.show_error(t("update.download_failed"))

        except asyncio.CancelledError:
            logger.warning("[SettingsPanel] Update cancelled by user")
            self.show_snackbar(t("update.cancelled"))
        except Exception as e:
            logger.exception(f"[SettingsPanel] Update failed: {e}")
            self.show_error(f"Update failed: {str(e)}")

    def _build_im_section(self) -> ft.Container:
        """Build IM (Instant Messaging) platform settings section."""
        logger.info("[SettingsPanel] Building IM section")

        # Platform configurations
        platforms = [
            (
                "wechat",
                t("settings_panel.im.platforms.wechat", default="微信 (WeChat)"),
                ["enabled", "token"],
            ),
            (
                "feishu",
                t("settings_panel.im.platforms.feishu", default="飞书 (Feishu)"),
                [
                    "enabled",
                    "app_id",
                    "app_secret",
                    "encrypt_key",
                    "verification_token",
                ],
            ),
            (
                "dingtalk",
                t("settings_panel.im.platforms.dingtalk", default="钉钉 (DingTalk)"),
                ["enabled", "app_key", "app_secret", "webhook_url"],
            ),
            (
                "wecom",
                t("settings_panel.im.platforms.wecom", default="企业微信 (WeCom)"),
                ["enabled", "bot_id", "secret"],
            ),
        ]

        platform_controls = []

        # ========== Gateway 状态显示（仅显示，不可操作） ==========
        # Gateway 默认开启，用户无需手动操作
        gateway_title = ft.Text(
            t("settings_panel.im.gateway_title", default="IM Gateway"),
            size=14,
            weight=ft.FontWeight.W_600,
            color="#e0e0e0",
        )

        # 刷新按钮 - 使用带边框的图标按钮更醒目
        refresh_btn = ft.IconButton(
            icon=ft.Icons.REFRESH,
            icon_size=18,
            icon_color="#3b82f6",
            tooltip="刷新状态",
            on_click=lambda _: self._refresh_gateway_status(),
            style=ft.ButtonStyle(
                bgcolor={
                    ft.ControlState.DEFAULT: "#2a2a2a",
                    ft.ControlState.HOVERED: "#3a3a3a",
                },
                shape=ft.RoundedRectangleBorder(radius=4),
                side=ft.BorderSide(1, "#3b82f6"),
                padding=ft.Padding(8, 8, 8, 8),
            ),
        )

        # 测试按钮 - 模拟微信会话过期（仅在开发/测试时使用）
        simulate_expire_btn = ft.IconButton(
            icon=ft.Icons.BUG_REPORT,
            icon_size=16,
            icon_color="#ff9800",
            tooltip="模拟微信会话过期（测试用）",
            on_click=lambda _: self._simulate_wechat_session_expired(),
            style=ft.ButtonStyle(
                bgcolor={
                    ft.ControlState.DEFAULT: "#2a2a2a",
                    ft.ControlState.HOVERED: "#3a3a3a",
                },
                shape=ft.RoundedRectangleBorder(radius=4),
                side=ft.BorderSide(1, "#ff9800"),
                padding=ft.Padding(6, 6, 6, 6),
            ),
            visible=False,  # 隐藏测试按钮（功能保留但不在UI显示）
        )

        gateway_enabled = g_config.gateway.enabled

        if gateway_enabled:
            status_icon = ft.Icon(ft.Icons.CHECK_CIRCLE, color="#6bcf7f", size=16)
            status_text = ft.Text(
                "运行中 - IM 接入已启用",
                size=12,
                color="#6bcf7f",
            )
        else:
            status_icon = ft.Icon(ft.Icons.CANCEL, color="#999999", size=16)
            status_text = ft.Text(
                "已停用 - IM 接入未启用",
                size=12,
                color="#999999",
            )

        gateway_status_row = ft.Row(
            [
                status_icon,
                status_text,
            ],
            spacing=4,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        # 检查 GatewayManager 启动错误
        startup_error = self._get_gateway_startup_error()
        error_controls = []
        if startup_error:
            error_controls.extend(
                [
                    ft.Container(height=8),
                    ft.Container(
                        content=ft.Row(
                            [
                                ft.Icon(ft.Icons.ERROR, color="#ff6b6b", size=16),
                                ft.Text(
                                    t(
                                        "settings_panel.im.gateway_error_title",
                                        default="启动失败",
                                    ),
                                    size=12,
                                    color="#ff6b6b",
                                    weight=ft.FontWeight.W_500,
                                ),
                            ],
                            spacing=4,
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                    ),
                    ft.Container(height=4),
                    ft.Container(
                        content=ft.Text(
                            startup_error,
                            size=11,
                            color="#ff9999",
                            selectable=True,
                        ),
                        padding=ft.Padding(8, 6, 8, 6),
                        bgcolor="#3d2828",
                        border_radius=4,
                    ),
                ]
            )

        # 标题行（包含刷新按钮和测试按钮）
        gateway_header_row = ft.Row(
            [
                gateway_title,
                ft.Row(
                    [
                        simulate_expire_btn,
                        refresh_btn,
                    ],
                    spacing=8,
                ),
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        gateway_section = ft.Container(
            content=ft.Column(
                [
                    gateway_header_row,
                    ft.Container(height=8),
                    gateway_status_row,
                    *error_controls,
                ],
                spacing=0,
            ),
            padding=ft.Padding(16, 16, 16, 16),
            bgcolor="#252526",
            border_radius=8,
            border=ft.border.all(1, "#383838"),
        )

        platform_controls.append(gateway_section)
        platform_controls.append(ft.Container(height=16))
        # =========================================================

        for platform_key, platform_name, fields in platforms:
            field_controls = []

            for field in fields:
                config_key = f"im.{platform_key}.{field}"
                current_value = self._get_nested_value(self.settings_data, config_key)

                # 定义每个渠道的必填字段
                required_fields = {
                    "feishu": {"app_id", "app_secret"},
                    "dingtalk": {"app_key", "app_secret"},
                    "wecom": {"bot_id", "secret"},
                    "wechat": {"token"},
                }
                platform_required = required_fields.get(platform_key, set())

                # 根据字段是否在必填列表中决定标记
                is_required = field in platform_required
                required_marker = " *" if is_required else " (可选)"
                field_label = SETTINGS_TITLE_MAP.get(field, field) + required_marker

                if field == "enabled":
                    # Boolean switch for enabled status
                    control = ft.Row(
                        [
                            ft.Text(
                                SETTINGS_TITLE_MAP.get(field, field),
                                size=13,
                                color="#cccccc",
                                width=120,
                            ),
                            ft.Switch(
                                value=bool(current_value)
                                if current_value is not None
                                else False,
                                on_change=lambda e, key=config_key, pid=platform_key: (
                                    self._on_im_enabled_toggle(
                                        key, pid, e.control.value
                                    )
                                ),
                                active_color="#3b82f6",
                                disabled=not gateway_enabled,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.START,
                    )
                else:
                    # Text field for other values
                    control = ft.Row(
                        [
                            ft.Text(
                                field_label,
                                size=13,
                                color="#cccccc",
                                width=120,
                            ),
                            ft.TextField(
                                value=str(current_value)
                                if current_value is not None
                                else "",
                                on_blur=lambda e, key=config_key: (
                                    self._set_config_value(key, e.control.value)
                                ),
                                border_color="#404040",
                                focused_border_color="#3b82f6",
                                text_size=13,
                                expand=True,
                                password=field
                                in ["app_secret", "secret", "encrypt_key", "token"],
                                can_reveal_password=field
                                in ["app_secret", "secret", "encrypt_key", "token"],
                                disabled=not gateway_enabled,
                                read_only=not gateway_enabled,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.START,
                    )

                field_controls.append(control)
                field_controls.append(ft.Container(height=8))

            # 为微信添加扫码登录按钮和状态显示
            if platform_key == "wechat":
                wechat_status, wechat_msg = self._get_wechat_login_status()

                # 检查微信是否启用
                wechat_enabled = (
                    self._get_nested_value(self.settings_data, "im.wechat.enabled")
                    or False
                )
                btn_disabled = not wechat_enabled

                # 状态指示器
                if wechat_status == "logged_in":
                    status_icon = ft.Icon(
                        ft.Icons.CHECK_CIRCLE, color="#6bcf7f", size=16
                    )
                    status_text = ft.Text("已登录", size=12, color="#6bcf7f")
                elif wechat_status == "expired":
                    status_icon = ft.Icon(ft.Icons.ERROR, color="#ff6b6b", size=16)
                    status_text = ft.Text("登录已过期", size=12, color="#ff6b6b")
                else:
                    status_icon = ft.Icon(
                        ft.Icons.INFO_OUTLINE, color="#888888", size=16
                    )
                    status_text = ft.Text("未登录", size=12, color="#888888")

                status_row = ft.Row(
                    [status_icon, status_text],
                    spacing=4,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                )
                field_controls.append(ft.Container(height=4))
                field_controls.append(status_row)

                # 按钮行：登录按钮 + 刷新按钮
                if wechat_status == "logged_in":
                    btn_text = "重新登录"
                    btn_bgcolor = "#666666" if not btn_disabled else "#444444"
                else:
                    btn_text = "扫码登录"
                    btn_bgcolor = "#3b82f6" if not btn_disabled else "#444444"

                login_btn = ft.TextButton(
                    content=ft.Text(
                        btn_text,
                        size=13,
                        color=ft.Colors.WHITE,
                    ),
                    style=ft.ButtonStyle(
                        bgcolor=btn_bgcolor,
                        padding=ft.Padding(16, 8, 16, 8),
                        shape=ft.RoundedRectangleBorder(radius=6),
                    ),
                    on_click=lambda e: self._on_wechat_login_click(),
                    disabled=btn_disabled,
                )

                refresh_btn = ft.IconButton(
                    icon=ft.Icons.REFRESH,
                    icon_color="#888888" if not btn_disabled else "#555555",
                    icon_size=20,
                    tooltip="刷新登录状态",
                    on_click=lambda e: self._refresh_wechat_status(),
                    disabled=btn_disabled,
                )

                btn_row = ft.Row(
                    [login_btn, refresh_btn],
                    spacing=8,
                )
                field_controls.append(ft.Container(height=8))
                field_controls.append(btn_row)

                # 显示禁用提示
                if btn_disabled:
                    hint_text = "请先启用微信"
                    field_controls.append(
                        ft.Text(
                            hint_text,
                            size=11,
                            color="#666666",
                            italic=True,
                        )
                    )

            # Platform section container
            platform_section = ft.Container(
                content=ft.Column(
                    [
                        ft.Text(
                            platform_name,
                            size=14,
                            weight=ft.FontWeight.W_600,
                            color="#e0e0e0",
                        ),
                        ft.Container(height=12),
                        *field_controls,
                    ],
                    spacing=0,
                ),
                padding=ft.Padding(16, 16, 16, 16),
                bgcolor="#252526",
                border_radius=8,
                border=ft.border.all(1, "#383838"),
            )

            platform_controls.append(platform_section)
            if platform_key != platforms[-1][0]:
                platform_controls.append(ft.Container(height=16))

        return ft.Container(
            content=ft.Column(
                [
                    ft.Text(
                        t(
                            "settings_panel.im.description",
                            default="配置即时通讯平台接入，支持微信、飞书、钉钉、企业微信",
                        ),
                        size=12,
                        color="#888888",
                    ),
                    ft.Container(height=20),
                    *platform_controls,
                ],
                spacing=0,
                scroll=ft.ScrollMode.AUTO,
            ),
        )

    def _get_nested_value(self, data: dict, key_path: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        keys = key_path.split(".")
        value = data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    def _get_gateway_startup_error(self) -> str | None:
        """获取 GatewayManager 启动错误信息。"""
        try:
            from middleware.im.gateway_starter import get_gateway_manager

            gateway_mgr = get_gateway_manager()
            if gateway_mgr:
                return gateway_mgr.get_startup_error()
        except Exception:
            pass
        return None

    def _on_im_enabled_toggle(
        self, config_key: str, platform_id: str, value: bool
    ) -> None:
        """Handle IM platform enabled toggle switch change.

        Saves the config and notifies Gateway to refresh channels.
        """
        try:
            logger.info(
                f"[SettingsPanel] _on_im_enabled_toggle: config_key={config_key}, "
                f"platform_id={platform_id}, value={value}"
            )

            # Save the enabled state to config
            self._set_config_value(config_key, value, refresh=False)

            # Verify the config was saved
            from middleware.config import g_config

            saved_value = g_config._runtime_config.to_json_dict()
            keys = config_key.split(".")
            current = saved_value
            for k in keys:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    current = None
                    break
            logger.info(
                f"[SettingsPanel] Config saved, verified {config_key}={current}"
            )

            # Notify Gateway to refresh IM channels
            # This allows dynamic start/stop without restarting Gateway
            try:
                from middleware.im.gateway_starter import get_gateway_manager

                gateway_mgr = get_gateway_manager()

                logger.debug(f"[SettingsPanel] Gateway manager: {gateway_mgr}")

                if gateway_mgr:
                    logger.debug(
                        f"[SettingsPanel] Gateway is_running: {gateway_mgr.is_running}"
                    )

                    if gateway_mgr.is_running:
                        logger.info(
                            f"[SettingsPanel] Calling refresh_channels_sync for {platform_id}"
                        )
                        gateway_mgr.refresh_channels_sync()
                        logger.info(
                            f"[SettingsPanel] Notified Gateway to refresh channels after {platform_id} toggle"
                        )
                    else:
                        logger.warning(
                            f"[SettingsPanel] Gateway exists but not running, config saved for {platform_id}"
                        )
                else:
                    logger.warning(
                        f"[SettingsPanel] Gateway manager is None, config saved for {platform_id}"
                    )
            except Exception as e:
                logger.error(f"[SettingsPanel] Failed to notify Gateway: {e}")
                import traceback

                logger.error(f"[SettingsPanel] Traceback: {traceback.format_exc()}")
                # Don't show error to user - config is already saved

            # Refresh the UI to show/hide platform fields
            self._refresh_content()

        except Exception as e:
            logger.error(f"[SettingsPanel] Failed to toggle {platform_id}: {e}")
            self.show_error(f"Failed to toggle {platform_id}: {e}")

    def _get_wechat_login_status(self) -> tuple[str, str]:
        token = self._get_nested_value(self.settings_data, "im.wechat.token")
        if not token:
            return ("not_logged_in", "未登录")
        # 检查会话是否过期
        if self._wechat_session_expired:
            return ("expired", "登录已过期")
        return ("logged_in", "已登录")

    def set_wechat_session_expired(self, expired: bool = True) -> None:
        """设置微信会话过期状态。

        当收到 session_expired 事件时调用，用于更新设置面板中的登录状态显示。

        Args:
            expired: True 表示会话已过期，False 表示重置状态
        """
        logger.info(f"[SettingsPanel] Setting wechat session expired: {expired}")
        self._wechat_session_expired = expired
        # 如果设置面板当前打开，刷新显示
        if self.dialog and self.dialog.open:
            self._refresh_content()

    async def _verify_wechat_token_async(
        self, base_url: str, token: str
    ) -> tuple[bool, str]:
        """异步验证微信token。"""
        import sys
        from pathlib import Path

        # 添加3rd目录到路径
        _3RD_DIR = Path(__file__).resolve().parent.parent.parent / "3rd"
        if str(_3RD_DIR) not in sys.path:
            sys.path.insert(0, str(_3RD_DIR))

        from weixin_sdk.client import WeixinClient
        from weixin_sdk.exceptions import WeixinSessionExpiredError, WeixinAuthError

        client = WeixinClient(base_url=base_url, token=token)
        try:
            # 调用get_config验证token
            await client.get_config()
            return (True, "Token有效")
        except WeixinSessionExpiredError:
            return (False, "Token已过期")
        except WeixinAuthError as e:
            if e.code == -14:
                return (False, "Token已过期")
            elif e.code in (401, 403):
                return (False, "Token无效")
            return (False, f"验证失败: {e.message}")
        except Exception as e:
            logger.error(f"[SettingsPanel] Token verification error: {e}")
            return (False, f"验证错误: {str(e)}")
        finally:
            await client.close()

    def _on_wechat_login_click(self):
        """处理微信扫码登录按钮点击。"""
        from gui.widgets.wechat_login_dialog import WechatLoginDialog

        wechat_status, _ = self._get_wechat_login_status()
        is_relogin = wechat_status == "logged_in"

        def on_success(token: str):
            logger.info("[SettingsPanel] WeChat login successful, saving token")
            try:
                self._set_config_value("im.wechat.token", token, refresh=False)

                # 重新登录后需要重启微信渠道，使新 token 生效
                try:
                    from middleware.im.gateway_starter import get_gateway_manager
                    from middleware.config import g_config

                    gateway_mgr = get_gateway_manager()
                    if gateway_mgr and gateway_mgr.is_running:
                        logger.info(
                            "[SettingsPanel] Restarting WeChat channel after login"
                        )

                        # 先禁用微信，触发停止
                        g_config.set("im.wechat.enabled", False, save=True)
                        gateway_mgr.refresh_channels_sync()

                        # 等待一下确保停止完成
                        import time

                        time.sleep(0.5)

                        # 再启用微信，触发启动（使用新token）
                        g_config.set("im.wechat.enabled", True, save=True)
                        gateway_mgr.refresh_channels_sync()

                        logger.info("[SettingsPanel] WeChat channel restarted")
                except Exception as e:
                    logger.error(f"[SettingsPanel] Failed to restart WeChat: {e}")

                # 重置微信会话过期状态
                self._wechat_session_expired = False

                if is_relogin:
                    self.show_snackbar("微信重新登录成功！")
                else:
                    self.show_snackbar("微信登录成功！")
                self._refresh_content()
            except Exception as e:
                logger.error(f"[SettingsPanel] Failed to save WeChat token: {e}")
                self.show_error(f"保存Token失败: {e}")

        def on_failed(error: str):
            logger.error(f"[SettingsPanel] WeChat login failed: {error}")
            self.show_error(f"微信登录失败: {error}")

        dialog = WechatLoginDialog(
            page=self.page,
            on_login_success=on_success,
            on_login_failed=on_failed,
            is_relogin=is_relogin,
        )
        dialog.show()

    def _refresh_gateway_status(self) -> None:
        """刷新 IM Gateway 状态并显示详细信息。"""
        logger.info("[SettingsPanel] Refreshing Gateway status")
        try:
            from middleware.im.gateway_starter import get_gateway_manager

            gateway_mgr = get_gateway_manager()
            if not gateway_mgr:
                self.show_snackbar("Gateway 尚未初始化")
                return

            is_running = gateway_mgr.is_running
            startup_error = gateway_mgr.get_startup_error()

            if startup_error:
                status_msg = f"Gateway 启动失败: {startup_error}"
                self.show_error(status_msg)
            elif is_running:
                # 获取启用的渠道数量
                channels = []
                if g_config.im.wechat.enabled:
                    channels.append("微信")
                if g_config.im.feishu.enabled:
                    channels.append("飞书")
                if g_config.im.dingtalk.enabled:
                    channels.append("钉钉")
                if g_config.im.wecom.enabled:
                    channels.append("企业微信")

                if channels:
                    status_msg = f"Gateway 运行中，已启用渠道: {', '.join(channels)}"
                else:
                    status_msg = "Gateway 运行中，暂无启用的渠道"
                self.show_snackbar(status_msg)
            else:
                self.show_snackbar("Gateway 未运行")

            # 刷新 UI 以显示最新的错误状态
            self._refresh_content()

        except Exception as e:
            logger.error(f"[SettingsPanel] Failed to refresh Gateway status: {e}")
            self.show_error(f"刷新 Gateway 状态失败: {e}")

    def _simulate_wechat_session_expired(self) -> None:
        """模拟微信会话过期（用于测试自动登录功能）。"""
        logger.info("[SettingsPanel] Simulating WeChat session expiration")
        try:
            from middleware.im.gateway.channels.wechat_ilinkai import (
                simulate_wechat_session_expired,
            )

            result = simulate_wechat_session_expired()
            if result:
                self.show_snackbar("已触发微信会话过期模拟，等待自动弹出登录对话框...")
            else:
                self.show_error("模拟失败：微信适配器未运行或未找到")
        except Exception as e:
            logger.error(f"[SettingsPanel] Failed to simulate session expiration: {e}")
            self.show_error(f"模拟会话过期失败: {e}")

    def _refresh_wechat_status(self):
        logger.info("[SettingsPanel] Refreshing WeChat status")
        try:
            from middleware.config import g_config

            self.settings_data = g_config._runtime_config.to_json_dict()
            self._refresh_content()
            self.show_snackbar("状态已刷新")
        except Exception as e:
            logger.error(f"[SettingsPanel] Failed to refresh WeChat status: {e}")
            self.show_error(f"刷新状态失败: {e}")
