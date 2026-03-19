from __future__ import annotations

import asyncio

import flet as ft

from middleware.config import g_config
from gui.i18n import t
from gui.widgets.sidebar import SessionSidebar
from gui.widgets.toolbar import Toolbar


class AppLayoutBuilder:
    """Build and wire Flet controls for main GUI."""

    def __init__(self, app):
        self.app = app
        # 注册语言切换观察者
        from gui.i18n import add_observer

        add_observer(self._on_language_changed)

        # 初始化下拉菜单相关属性
        self._overlay_entry = None
        self._dropdown_position = None
        self._model_filter = ""
        self._is_editing_models = False

    def _on_language_changed(self, new_lang: str):
        """语言切换时的回调 - 刷新所有 UI 文本"""
        # 刷新输入框占位符
        if hasattr(self.app, "message_input") and self.app.message_input:
            self.app.message_input.hint_text = t("input.placeholder")

        # 刷新按钮工具提示
        if hasattr(self.app, "send_button") and self.app.send_button:
            self.app.send_button.tooltip = t("input.send")
        if hasattr(self.app, "stop_button") and self.app.stop_button:
            self.app.stop_button.tooltip = t("input.stop")

        # 刷新状态文本
        if hasattr(self.app, "status_text") and self.app.status_text:
            # 保持当前状态，但如果是默认状态则刷新
            if self.app.status_text.value == t(
                "input.status_ready", locale="zh-CN"
            ) or self.app.status_text.value == t("input.status_ready", locale="en-US"):
                self.app.status_text.value = t("input.status_ready")

        # 刷新工具栏
        if hasattr(self.app, "toolbar") and self.app.toolbar:
            if hasattr(self.app.toolbar, "_on_language_changed"):
                self.app.toolbar._on_language_changed(new_lang)

        # 更新页面
        if self.app.page:
            self.app.page.update()

    def create_sidebar(self):
        self.app.sidebar = SessionSidebar(
            on_new_chat=self.app._on_new_chat,
            on_select_session=lambda sid: asyncio.create_task(
                self.app._on_select_session(sid)
            ),
            on_delete_session=lambda sid: asyncio.create_task(
                self.app._on_delete_session(sid)
            ),
            on_rename_session=lambda sid: asyncio.create_task(
                self.app._on_rename_session(sid)
            ),
            on_load_more=lambda: asyncio.create_task(self.app._on_load_more_sessions()),
        )

    def create_main_area(self):
        self.app.toolbar = Toolbar(
            on_settings=self.app._show_settings,
            on_clear=self.app._on_clear_chat,
            on_export=self.app._on_export_chat,
        )

        self.app.chat_list = ft.ListView(
            expand=True,
            spacing=12,
            padding=20,
            auto_scroll=True,
        )

        self.app.loading_indicator = ft.ProgressRing(
            width=20,
            height=20,
            stroke_width=2,
            visible=False,
        )

        self.app.status_text = ft.Text(
            t("input.status_ready"),
            size=12,
            color=ft.Colors.GREY_500,
        )

        self.app.selected_command_index = -1
        self.app.filtered_commands = []
        self.app.command_hint_items = []

        self.app.command_hints = ft.ListView(
            height=180,
            spacing=2,
            padding=ft.Padding.only(left=12, right=12, top=8, bottom=8),
        )

        self.app.message_input = ft.TextField(
            hint_text=t("input.placeholder"),
            multiline=True,
            min_lines=3,
            max_lines=10,
            expand=True,
            border_radius=ft.BorderRadius.all(6),
            filled=False,
            border=ft.InputBorder.NONE,
            cursor_color=ft.Colors.WHITE,
            selection_color=ft.Colors.BLUE_400,
            on_change=self.app._on_input_change,
        )

        self.app.send_button = ft.IconButton(
            icon=ft.icons.Icons.SEND,
            icon_color=ft.Colors.BLUE_400,
            on_click=lambda e: asyncio.create_task(self.app._on_send_message(e)),
            tooltip=t("input.send"),
        )

        self.app.stop_button = ft.IconButton(
            icon=ft.icons.Icons.STOP,
            icon_color=ft.Colors.RED_400,
            on_click=lambda e: self.app._on_stop_generation(e),
            tooltip=t("input.stop"),
            visible=False,
        )

        input_with_buttons = ft.Container(
            content=ft.Column(
                [
                    ft.Container(
                        content=self.app.message_input,
                        expand=True,
                        padding=ft.Padding.only(left=6, right=6, top=4),
                    ),
                    ft.Container(
                        content=ft.Row(
                            [
                                ft.Container(expand=True),
                                self.app.stop_button,
                                self.app.send_button,
                            ],
                            spacing=4,
                        ),
                        padding=ft.Padding.only(left=6, right=6, bottom=4),
                    ),
                ],
                spacing=0,
                expand=True,
            ),
            bgcolor=ft.Colors.GREY_800,
            border_radius=ft.BorderRadius.all(8),
            padding=4,
        )

        self.app.command_hints_container = ft.Container(
            content=self.app.command_hints,
            bgcolor=ft.Colors.GREY_800,
            border_radius=ft.BorderRadius.all(8),
            border=ft.Border.all(1, ft.Colors.GREY_700),
            padding=8,
            visible=False,
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=8,
                color=ft.Colors.BLACK38,
                offset=ft.Offset(0, -4),
            ),
        )

        input_column = ft.Column(
            [
                ft.Container(
                    content=self.app.command_hints_container,
                    padding=ft.Padding.only(left=16, right=16, bottom=4),
                ),
                ft.Container(
                    content=input_with_buttons,
                    padding=ft.Padding.symmetric(horizontal=16, vertical=8),
                    bgcolor=ft.Colors.GREY_900,
                ),
            ],
            spacing=0,
        )

        # 从配置加载模型信息
        self._load_model_config()

        self.app.token_text_bottom = ft.Text(
            "0 / 80000", size=11, color=ft.Colors.GREY_400
        )
        self.app.token_progress_bottom = ft.ProgressBar(
            width=80,
            value=0,
            color=ft.Colors.BLUE_400,
            bgcolor=ft.Colors.GREY_800,
        )

        # 加载模型配置
        self._load_model_config()

        # 创建模型选择下拉菜单
        self.app.model_selector = self._build_model_selector()

        bottom_info_bar = ft.Container(
            content=ft.Row(
                [
                    ft.Row(
                        [
                            ft.Icon(
                                ft.icons.Icons.MODEL_TRAINING,
                                size=14,
                                color=ft.Colors.BLUE_400,
                            ),
                            self.app.model_selector,
                        ],
                        spacing=6,
                    ),
                    ft.VerticalDivider(width=1, color=ft.Colors.GREY_800),
                    ft.Row(
                        [
                            ft.Icon(
                                ft.icons.Icons.TOKEN,
                                size=12,
                                color=ft.Colors.GREY_500,
                            ),
                            self.app.token_text_bottom,
                            self.app.token_progress_bottom,
                        ],
                        spacing=6,
                    ),
                    ft.Container(expand=True),
                    ft.Container(
                        content=ft.Row(
                            [
                                self.app.loading_indicator,
                                ft.Container(width=8),
                                self.app.status_text,
                            ],
                            spacing=4,
                        ),
                    ),
                ],
                spacing=12,
            ),
            padding=ft.Padding.symmetric(horizontal=20, vertical=8),
            bgcolor=ft.Colors.GREY_900,
            border=ft.Border.only(top=ft.BorderSide(1, ft.Colors.GREY_800)),
        )

        self.app.main_area = ft.Column(
            [
                self.app.toolbar,
                self.app.chat_list,
                input_column,
                bottom_info_bar,
            ],
            expand=True,
            spacing=0,
        )

    def _load_model_config(self):
        """从配置加载模型信息"""
        try:
            if g_config is None:
                g_config.load()

            if g_config and g_config.llm:
                self.current_profile = g_config.llm.active_profile
                self.available_profiles = list(g_config.llm.profiles.keys())
                current_model = g_config.llm.current
                self.current_model_name = (
                    current_model.model if current_model else self.current_profile
                )
            else:
                self.current_profile = "default"
                self.available_profiles = ["default"]
                self.current_model_name = "Kimi-K2.5"
        except Exception as e:
            self.current_profile = "default"
            self.available_profiles = ["default"]
            self.current_model_name = "Kimi-K2.5"
            print(f"[Layout] Failed to load model config: {e}")

    def _build_model_selector(self) -> ft.Control:
        """构建设模型选择下拉菜单（自定义实现）"""
        # 截断当前模型名称显示
        display_current_model = self.current_model_name
        if len(display_current_model) > 20:
            display_current_model = display_current_model[:17] + "..."

        # 创建模型列表容器（添加高度限制）
        self.model_list_column = ft.Column(
            spacing=0,
            scroll=ft.ScrollMode.AUTO,
            height=150,  # 设置固定高度
        )

        # 搜索框
        self.search_field = ft.TextField(
            hint_text="搜索模型...",
            hint_style=ft.TextStyle(size=11, color=ft.Colors.GREY_500),
            text_style=ft.TextStyle(size=11, color=ft.Colors.WHITE),
            border_color=ft.Colors.GREY_700,
            focused_border_color=ft.Colors.BLUE_400,
            height=32,
            content_padding=ft.Padding(left=8, right=8, top=0, bottom=0),
            expand=True,
            on_change=lambda e: self._on_search_change(e.control.value),
        )

        # 添加按钮
        add_btn = ft.IconButton(
            icon=ft.icons.Icons.ADD,
            icon_size=19,
            icon_color=ft.Colors.GREEN_400,
            tooltip="添加模型",
            on_click=lambda e: self._on_add_model_click(),
            style=ft.ButtonStyle(
                padding=ft.Padding(left=2, right=2, top=2, bottom=2),
            ),
        )

        # 编辑按钮
        self.edit_btn = ft.IconButton(
            icon=ft.icons.Icons.EDIT,
            icon_size=18,
            icon_color=ft.Colors.BLUE_400,
            tooltip="编辑模型",
            on_click=lambda e: self._on_edit_model_click(),
            style=ft.ButtonStyle(
                padding=ft.Padding(left=2, right=2, top=2, bottom=2),
            ),
        )
        edit_btn = self.edit_btn

        # 搜索和操作按钮行
        search_row = ft.Row(
            [
                self.search_field,
                add_btn,
                edit_btn,
            ],
            spacing=2,
        )

        # 下拉菜单内容
        self.dropdown_menu = ft.Container(
            content=ft.Column(
                [
                    ft.Container(
                        content=search_row,
                        padding=ft.Padding(left=8, right=4, top=8, bottom=8),
                    ),
                    ft.Divider(height=1, color=ft.Colors.GREY_700),
                    self.model_list_column,
                    ft.Divider(height=1, color=ft.Colors.GREY_700),
                    ft.Container(
                        content=ft.Row(
                            [
                                ft.Icon(
                                    ft.icons.Icons.SETTINGS,
                                    size=14,
                                    color=ft.Colors.GREY_400,
                                ),
                                ft.Text(
                                    t("toolbar.model_settings"),
                                    size=11,
                                    color=ft.Colors.GREY_300,
                                ),
                            ],
                            spacing=8,
                        ),
                        padding=ft.Padding(left=12, right=12, top=8, bottom=8),
                        on_click=lambda e: (
                            self._hide_dropdown(),
                            self.app._show_settings_with_category("llm"),
                        ),
                    ),
                ],
                spacing=0,
                tight=True,
            ),
            width=320,
            height=240,  # 设置固定总高度
            bgcolor=ft.Colors.GREY_900,
            border_radius=4,
            border=ft.border.all(1, ft.Colors.GREY_700),
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=10,
                color=ft.Colors.BLACK54,
            ),
            visible=False,
        )

        # 选择器按钮 - 保存引用以便获取宽度
        self._selector_button_container = ft.Container(
            content=ft.Row(
                [
                    ft.Text(
                        display_current_model,
                        size=11,
                        color=ft.Colors.GREY_300,
                        overflow=ft.TextOverflow.ELLIPSIS,
                        no_wrap=True,
                        expand=True,
                    ),
                    ft.Icon(
                        ft.icons.Icons.ARROW_DROP_DOWN,
                        size=16,
                        color=ft.Colors.GREY_400,
                    ),
                    ft.Container(width=4),  # 图标右侧间距
                ],
                spacing=2,
            ),
            padding=ft.Padding(left=8, right=4, top=4, bottom=4),
            border_radius=4,
            border=ft.border.all(1, ft.Colors.GREY_700),
            bgcolor=ft.Colors.GREY_800,
            width=150,
        )

        selector_button = ft.GestureDetector(
            content=self._selector_button_container,
            on_tap=self._toggle_dropdown,
        )

        return ft.Container(
            content=selector_button,
            tooltip=t("settings.sections.provider"),
        )

    def _on_model_selected(self, profile_name: str):
        """处理模型选择"""
        if profile_name == self.current_profile:
            return

        try:
            # 隐藏下拉菜单
            self._hide_dropdown()

            # 更新配置中的活动profile
            g_config.set("llm.active_profile", profile_name)

            # 重新加载配置
            self._load_model_config()

            # 更新UI - 刷新整个底部状态栏
            self._refresh_bottom_info_bar()

            # 显示提示
            if hasattr(self.app, "_show_snackbar"):
                self.app._show_snackbar(
                    t("settings.messages.model_changed", name=profile_name)
                )

            # 触发应用层的模型切换回调（如果有）
            if hasattr(self.app, "_on_model_changed"):
                asyncio.create_task(self.app._on_model_changed(profile_name))

        except Exception as e:
            print(f"[Layout] Failed to switch model: {e}")
            if hasattr(self.app, "_show_error"):
                self.app._show_error(
                    t("settings.messages.model_change_failed", error=e)
                )

    def _toggle_dropdown(self, e=None):
        """切换下拉菜单显示状态（使用 Overlay）"""
        if self._overlay_entry and self._overlay_entry in self.app.page.overlay:
            # 已显示，关闭它
            self._hide_dropdown()
        else:
            # 未显示，打开它
            self._show_dropdown()

    def _show_dropdown(self):
        """显示下拉菜单（使用 Overlay）"""
        # 重置搜索
        self.search_field.value = ""
        self._model_filter = ""
        # 重置编辑状态
        self._is_editing_models = False
        # 重置编辑按钮图标（先设置属性，等添加到页面后再更新）
        if hasattr(self, "edit_btn"):
            self.edit_btn.icon = ft.icons.Icons.EDIT
            self.edit_btn.tooltip = "编辑模型"
        self._update_model_list()

        # 确保下拉菜单可见
        self.dropdown_menu.visible = True

        # 获取按钮的实际宽度
        button_width = 200  # 默认宽度
        if (
            hasattr(self, "_selector_button_container")
            and self._selector_button_container
        ):
            # 尝试获取按钮的实际宽度
            if (
                hasattr(self._selector_button_container, "width")
                and self._selector_button_container.width
            ):
                button_width = self._selector_button_container.width
            elif (
                hasattr(self._selector_button_container, "content")
                and self._selector_button_container.content
            ):
                # 估算宽度：文本 + 图标 + 内边距
                button_width = len(self.current_model_name) * 7 + 40

        # 下拉菜单宽度 = 按钮宽度 + 20px
        dropdown_width = button_width + 20
        dropdown_height = 240

        # 更新下拉菜单容器宽度
        self.dropdown_menu.width = dropdown_width

        # 获取页面尺寸
        page_width = (
            self.app.page.window.width
            if hasattr(self.app.page.window, "width")
            else 1200
        )
        page_height = (
            self.app.page.window.height
            if hasattr(self.app.page.window, "height")
            else 800
        )

        if self._dropdown_position:
            # 下拉菜单中心对齐按钮中心
            left = self._dropdown_position[0] - (dropdown_width / 2)
            top = self._dropdown_position[1] - dropdown_height - 5  # 正上方，5px 间距
        else:
            # 默认位置
            left = 180
            top = page_height - dropdown_height - 80

        # 确保不超出屏幕边界
        left = max(10, min(left, page_width - dropdown_width - 10))
        top = max(10, min(top, page_height - dropdown_height - 10))

        # 创建带透明背景的全屏 Stack，包含定位的下拉菜单
        self._overlay_entry = ft.Stack(
            [
                # 透明背景层，点击可关闭菜单
                ft.GestureDetector(
                    content=ft.Container(
                        bgcolor=ft.Colors.TRANSPARENT,
                        expand=True,
                    ),
                    on_tap=lambda e: self._hide_dropdown(),
                ),
                # 定位的下拉菜单
                ft.Container(
                    content=self.dropdown_menu,
                    left=left + 120,
                    top=top + 10,
                ),
            ],
            expand=True,
        )

        # 添加到 overlay
        self.app.page.overlay.append(self._overlay_entry)
        self.app.page.update()

    def _hide_dropdown(self):
        """隐藏下拉菜单"""
        if hasattr(self, "_overlay_entry") and self._overlay_entry:
            if self._overlay_entry in self.app.page.overlay:
                self.app.page.overlay.remove(self._overlay_entry)
            self._overlay_entry = None
            self.app.page.update()

    def _update_model_list(self, filter_text: str = ""):
        """更新模型列表"""
        print(
            f"[_update_model_list] called with filter: '{filter_text}', editing: {self._is_editing_models}"
        )
        print(f"[_update_model_list] available_profiles: {self.available_profiles}")
        model_items = []

        # 根据搜索词过滤
        for profile_name in self.available_profiles:
            if filter_text and filter_text.lower() not in profile_name.lower():
                continue

            profile = g_config.llm.profiles.get(profile_name) if g_config else None
            model_display = profile.model if profile else profile_name

            # 截断显示名称
            display_name = model_display
            if len(display_name) > 30:
                display_name = display_name[:27] + "..."

            is_active = profile_name == self.current_profile

            # 构建模型项的行内容
            row_controls = [
                ft.Icon(
                    ft.icons.Icons.CHECK_CIRCLE
                    if is_active
                    else ft.icons.Icons.CIRCLE_OUTLINED,
                    size=14,
                    color=ft.Colors.BLUE_400 if is_active else ft.Colors.GREY_500,
                ),
                ft.Column(
                    [
                        ft.Text(
                            profile_name,
                            size=11,
                            weight=ft.FontWeight.W_500
                            if is_active
                            else ft.FontWeight.W_400,
                            color=ft.Colors.WHITE if is_active else ft.Colors.GREY_300,
                        ),
                        ft.Text(
                            display_name,
                            size=9,
                            color=ft.Colors.GREY_500,
                        ),
                    ],
                    spacing=0,
                    expand=True,
                ),
            ]

            # 编辑模式下添加删除按钮（至少保留一个模型）
            if self._is_editing_models and len(self.available_profiles) > 1:
                delete_btn = ft.IconButton(
                    icon=ft.icons.Icons.DELETE_OUTLINE,
                    icon_size=16,
                    icon_color=ft.Colors.RED_400,
                    tooltip="删除模型",
                    on_click=lambda e, name=profile_name: self._on_delete_model_click(
                        name
                    ),
                    style=ft.ButtonStyle(
                        padding=ft.Padding(left=4, right=4, top=4, bottom=4),
                    ),
                )
                row_controls.append(delete_btn)

            item = ft.Container(
                content=ft.Row(
                    row_controls,
                    spacing=8,
                ),
                padding=ft.Padding(left=12, right=12, top=6, bottom=6),
                on_click=lambda e, name=profile_name: self._on_model_selected(name)
                if not self._is_editing_models
                else None,
                bgcolor=ft.Colors.GREY_800 if is_active else None,
            )
            model_items.append(item)

        if not model_items:
            model_items.append(
                ft.Container(
                    content=ft.Text(
                        "未找到匹配的模型",
                        size=11,
                        color=ft.Colors.GREY_500,
                    ),
                    padding=ft.Padding(left=12, right=12, top=12, bottom=12),
                )
            )

        self.model_list_column.controls = model_items
        print(f"[_update_model_list] set {len(model_items)} items to column")
        # 刷新页面以显示新内容
        if self.app.page:
            print(f"[_update_model_list] calling page.update()")
            self.app.page.update()
            print(f"[_update_model_list] page updated")

    def _on_search_change(self, value: str):
        """搜索框内容变化时处理"""
        # 实时更新模型列表（不关闭菜单）
        self._update_model_list(value)

    def _on_add_model_click(self):
        """点击添加模型按钮"""
        print(f"[Layout] _on_add_model_click started")
        self._hide_dropdown()
        print(f"[Layout] Dropdown hidden")

        # 创建设置面板并显示添加模型对话框
        from gui.modules.settings_panel import SettingsPanel

        def on_model_added():
            """模型添加后的回调"""
            print(f"[Layout] on_model_added callback started")
            self._load_model_config()
            print(
                f"[Layout] _load_model_config completed, current profiles: {self.available_profiles}"
            )
            self._refresh_bottom_info_bar()
            print(f"[Layout] _refresh_bottom_info_bar completed")

        settings_panel = SettingsPanel(
            page=self.app.page,
            show_error=self.app._show_error
            if hasattr(self.app, "_show_error")
            else print,
            show_snackbar=self.app._show_snackbar
            if hasattr(self.app, "_show_snackbar")
            else print,
            on_save_callback=on_model_added,
        )
        settings_panel._show_add_profile_dialog()

    def _on_edit_model_click(self):
        """点击编辑模型按钮 - 切换编辑状态"""
        self._is_editing_models = not self._is_editing_models
        print(f"[Layout] Edit mode: {self._is_editing_models}")

        # 切换编辑按钮图标
        if hasattr(self, "edit_btn"):
            if self._is_editing_models:
                self.edit_btn.icon = ft.icons.Icons.CHECK
                self.edit_btn.tooltip = "完成编辑"
            else:
                self.edit_btn.icon = ft.icons.Icons.EDIT
                self.edit_btn.tooltip = "编辑模型"
            self.edit_btn.update()

        # 刷新列表显示编辑状态
        self._update_model_list()

    def _on_delete_model_click(self, profile_name: str):
        """点击删除按钮 - 显示确认对话框"""
        print(f"[Layout] Delete clicked for: {profile_name}")

        def confirm_delete(e):
            print(f"[Layout] Confirm delete: {profile_name}")
            dialog.open = False
            self._delete_profile(profile_name)
            self.app.page.update()

        def cancel_delete(e):
            print(f"[Layout] Cancel delete: {profile_name}")
            dialog.open = False
            self.app.page.update()

        dialog = ft.AlertDialog(
            title=ft.Text("确认删除"),
            content=ft.Text(f"确定要删除模型 '{profile_name}' 吗？\n此操作不可撤销。"),
            actions=[
                ft.TextButton("取消", on_click=cancel_delete),
                ft.TextButton(
                    "删除",
                    on_click=confirm_delete,
                    style=ft.ButtonStyle(color=ft.Colors.RED_400),
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
            shape=ft.RoundedRectangleBorder(radius=8),
        )
        self.app.page.overlay.append(dialog)
        dialog.open = True
        self.app.page.update()

    def _delete_profile(self, profile_name: str):
        """删除模型配置"""
        try:
            print(f"[Layout] Deleting profile: {profile_name}")

            # 检查是否是最后一个模型
            if len(self.available_profiles) <= 1:
                print(f"[Layout] Cannot delete last profile")
                if hasattr(self.app, "_show_snackbar"):
                    self.app._show_snackbar("不能删除最后一个模型")
                return

            # 从配置中删除
            if profile_name in g_config.llm.profiles:
                # 先获取剩余模型列表（转换为普通字典）
                profiles = {}
                for name, profile in g_config.llm.profiles.items():
                    if name != profile_name:
                        # 将 Pydantic 模型转换为字典
                        if hasattr(profile, "model_dump"):
                            profiles[name] = profile.model_dump()
                        elif hasattr(profile, "dict"):
                            profiles[name] = profile.dict()
                        else:
                            profiles[name] = dict(profile)

                # 如果删除的是当前选中模型，先切换 active_profile
                if profile_name == self.current_profile:
                    remaining_profiles = list(profiles.keys())
                    if remaining_profiles:
                        new_profile = remaining_profiles[0]
                        print(f"[Layout] Switching to new profile: {new_profile}")
                        # 先更新 active_profile，避免验证失败
                        g_config.set("llm.active_profile", new_profile, save=False)

                # 再删除 profile 并保存
                g_config.set("llm.profiles", profiles, save=True)

                # 刷新配置和 UI
                self._load_model_config()
                if profile_name == self.current_profile:
                    self._refresh_bottom_info_bar()

                # 刷新列表
                self._update_model_list()

                # 显示提示
                if hasattr(self.app, "_show_snackbar"):
                    self.app._show_snackbar(f"模型 '{profile_name}' 已删除")

                print(f"[Layout] Profile deleted successfully")
            else:
                print(f"[Layout] Profile not found: {profile_name}")

        except Exception as e:
            print(f"[Layout] Failed to delete profile: {e}")
            import traceback

            traceback.print_exc()
            if hasattr(self.app, "_show_error"):
                self.app._show_error(f"删除模型失败: {e}")

    def _refresh_bottom_info_bar(self):
        """刷新底部状态栏的模型选择器显示"""
        try:
            # 重建模型选择器
            new_selector = self._build_model_selector()
            self.app.model_selector = new_selector

            # 需要找到 bottom_info_bar 并更新其内容
            # bottom_info_bar 是 main_area 的最后一个子元素
            if hasattr(self.app, "main_area") and self.app.main_area:
                # 获取 main_area 的 controls 列表
                controls = self.app.main_area.controls
                if controls:
                    # 最后一个控件是 bottom_info_bar
                    old_bottom_bar = controls[-1]
                    if old_bottom_bar and hasattr(old_bottom_bar, "content"):
                        # 重建 bottom_info_bar
                        new_bottom_bar = self._build_bottom_info_bar()
                        # 替换 main_area 中的 bottom_info_bar
                        controls[-1] = new_bottom_bar

                        if self.app.page:
                            self.app.page.update()
                            print(
                                f"[Layout] Bottom info bar refreshed with model: {self.current_model_name}"
                            )
        except Exception as e:
            print(f"[Layout] Error refreshing bottom info bar: {e}")

    def _build_bottom_info_bar(self) -> ft.Container:
        """构建底部信息栏"""
        return ft.Container(
            content=ft.Row(
                [
                    ft.Row(
                        [
                            ft.Icon(
                                ft.icons.Icons.MODEL_TRAINING,
                                size=14,
                                color=ft.Colors.BLUE_400,
                            ),
                            self.app.model_selector,
                        ],
                        spacing=6,
                    ),
                    ft.VerticalDivider(width=1, color=ft.Colors.GREY_800),
                    ft.Row(
                        [
                            ft.Icon(
                                ft.icons.Icons.TOKEN,
                                size=12,
                                color=ft.Colors.GREY_500,
                            ),
                            self.app.token_text_bottom,
                            self.app.token_progress_bottom,
                        ],
                        spacing=6,
                    ),
                    ft.Container(expand=True),
                    ft.Container(
                        content=ft.Row(
                            [
                                self.app.loading_indicator,
                                ft.Container(width=8),
                                self.app.status_text,
                            ],
                            spacing=4,
                        ),
                    ),
                ],
                spacing=12,
            ),
            padding=ft.Padding.symmetric(horizontal=20, vertical=8),
            bgcolor=ft.Colors.GREY_900,
            border=ft.Border.only(top=ft.BorderSide(1, ft.Colors.GREY_800)),
            clip_behavior=ft.ClipBehavior.NONE,  # 不裁剪子控件
        )

    def refresh_model_selector(self):
        """刷新模型选择器（当配置发生变化时调用）"""
        self._load_model_config()
        self._refresh_bottom_info_bar()
