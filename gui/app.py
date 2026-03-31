"""
Memento-S GUI Application - Improved Version with Full TUI Features

Features:
    - Auto title generation with LLM
    - Time-based conversation grouping (Today/Yesterday/This Week/This Month/Earlier)
    - Auto context compression when approaching token limits
    - Complete keyboard shortcuts (ESC double-press interrupt)
    - Loading indicators and status bar
    - Better message styling with copy support
    - Full command system support
"""

from __future__ import annotations

import asyncio
import platform
import sys
import warnings
from pathlib import Path
from typing import Optional

import flet as ft

# 抑制无害的 Lark SDK 警告
warnings.filterwarnings(
    "ignore",
    message=".*Task exception was never retrieved.*",
    category=RuntimeWarning,
)

from bootstrap import bootstrap_sync
from shared.chat import ChatManager

# Core imports
from core.memento_s import MementoSAgent
from core.skill import SkillGateway
from gui.i18n import t
from gui.modules.auth_service import AuthService
from gui.modules.command_controller import CommandController
from gui.modules.conversation_controller import ConversationController
from gui.modules.input_controller import InputController
from gui.modules.keyboard_controller import KeyboardController
from gui.modules.layout import AppLayoutBuilder
from gui.modules.main_layout import MainLayout
from gui.modules.message_controller import MessageController
from gui.modules.settings_panel import SettingsPanel
from gui.modules.ui_feedback_controller import UIFeedbackController
from gui.modules.update_notifier import UpdateNotifier
from gui.modules.auto_update_manager import AutoUpdateManager

# GUI imports
from gui.widgets.sidebar import SessionSidebar
from middleware.config import g_config
from middleware.llm import LLMClient
from utils.logger import logger


class MementoSGUIImproved:
    """改进的 GUI 应用类"""

    # ESC 双击检测间隔（秒）
    _DOUBLE_ESC_INTERVAL: float = 0.4

    # Slash commands (like TUI)
    COMMANDS = {
        "/clear": t("commands.clear"),
        "/context": t("commands.context"),
        "/compress": t("commands.compress"),
        "/reset": t("commands.reset"),
        "/skills": t("commands.skills"),
        "/reload": t("commands.reload"),
        "/history": t("commands.history"),
        "/save": t("commands.save"),
        "/new": t("commands.new"),
        "/load": t("commands.load"),
        "/rename": t("commands.rename"),
        "/delete": t("commands.delete"),
        "/exit": t("commands.exit"),
        "/feishu": "Start/stop Feishu WebSocket bridge (start|stop|status)",
        "/help": t("commands.help"),
    }

    def __init__(self):
        self.page: Optional[ft.Page] = None

        # Splash / startup overlay
        self.startup_overlay: Optional[ft.Container] = None

        # Core components
        self._llm: Optional[LLM] = None
        self._skill_gateway: Optional[SkillGateway] = None
        self._agent: Optional[MementoSAgent] = None

        # Session & Conversation management
        # Architecture: 1 Session -> N Conversations
        # Each user message and AI reply creates a new Conversation
        self.current_session_id: Optional[str] = None  # Current active session
        self.messages: list[dict] = []  # Messages currently displayed in UI
        self.total_tokens: int = 0

        # UI State
        self.is_processing: bool = False
        self._current_task: Optional[asyncio.Task] = None
        self._last_esc_time: float = 0

        # Settings
        self.theme_mode = ft.ThemeMode.DARK
        self.auto_save = True

        # Track Flet child processes for clean exit
        self._flet_process_ids: list[int] = []

        # Components (initialized in build)
        self.sidebar: Optional[ft.Container] = None
        self.session_sidebar: Optional[SessionSidebar] = None
        self.user_info_bar = None
        self.chat_list: Optional[ft.ListView] = None
        self.message_input: Optional[ft.TextField] = None
        self.loading_indicator: Optional[ft.ProgressRing] = None
        self.status_text: Optional[ft.Text] = None

        # Modular helpers
        self.layout_builder = AppLayoutBuilder(self)
        self.settings_panel: Optional[SettingsPanel] = None
        self.conversation_controller = ConversationController(self, logger)
        self.keyboard_controller = KeyboardController(self)
        self.input_controller = InputController(self)
        self.message_controller = MessageController(self, logger)
        self.command_controller = CommandController(self)
        self.ui_feedback_controller = UIFeedbackController(self)
        self.update_manager: Optional[AutoUpdateManager] = None
        self.update_notifier: Optional[UpdateNotifier] = None
        self.auth_service = AuthService()
        self._login_dialog = None

        # 注册语言切换观察者
        from gui.i18n import add_observer

        add_observer(self._on_language_changed)

        # 订阅 IM 事件（显示 Toast 通知）
        self._setup_im_event_handlers()

        # 订阅 401 认证失效事件
        self._setup_auth_event_handlers()

    def _setup_auth_event_handlers(self):
        """订阅 AUTH_REQUIRED 事件，API 返回 401 时清空 token 并弹出登录框。"""
        try:
            from utils.event_bus import event_bus, EventType

            def on_auth_required(event):
                try:
                    self.auth_service.clear_token()
                    if self.user_info_bar:
                        self.user_info_bar.update_user_area(
                            logged_in=False, display_name=""
                        )
                    if self.layout_builder:
                        self.layout_builder.update_sidebar_user_state(
                            logged_in=False, display_name=""
                        )
                    if self.page:
                        self._show_login_dialog_for_401()
                except Exception as e:
                    logger.warning(f"[App] Error handling AUTH_REQUIRED: {e}")

            event_bus.subscribe(EventType.AUTH_REQUIRED, on_auth_required)
            logger.info("[App] AUTH_REQUIRED event handler registered")
        except Exception as e:
            logger.warning(f"[App] Failed to setup auth event handlers: {e}")

    def _show_login_dialog_for_401(self):
        """在主线程安全地弹出登录框（供事件回调调用）。"""
        if not self.page:
            return

        def _open():
            try:
                self._ensure_login_dialog()
                self._login_dialog.show()
            except Exception as e:
                logger.warning(f"[App] Failed to show login dialog for 401: {e}")

        try:
            self.page.call_later(0, _open)
        except Exception:
            _open()

    def _setup_im_event_handlers(self):
        """设置 IM 事件处理器"""
        try:
            from utils.event_bus import event_bus, EventType

            def on_im_event(event):
                """处理 IM 事件"""
                try:
                    data = event.data or {}
                    platform = data.get("platform", data.get("platform_id", "未知平台"))
                    error = data.get("error")

                    if event.type == EventType.IM_SERVICE_STARTED:
                        self._show_snackbar(f"✓ {platform} 服务已启动")
                    elif event.type == EventType.IM_SERVICE_STOPPED:
                        self._show_snackbar(f"✓ {platform} 服务已停止")
                    elif event.type == EventType.IM_SERVICE_START_FAILED:
                        msg = f"✗ {platform} 启动失败"
                        if error:
                            msg += f": {error[:50]}"
                        self._show_error(msg)

                        # 检查是否需要重新登录（微信 session 过期）
                        # 同时更新设置面板中的微信登录状态
                        if self.settings_panel:
                            self.settings_panel.set_wechat_session_expired(True)
                            logger.info(
                                "[App] Updated settings panel wechat status to expired"
                            )
                        if data.get("event_type") == "session_expired" and data.get(
                            "requires_relogin"
                        ):
                            logger.info(
                                "[App] WeChat session expired, showing login dialog"
                            )
                            self._show_wechat_login_dialog(is_relogin=True)
                    elif event.type == EventType.IM_SERVICE_STOP_FAILED:
                        msg = f"✗ {platform} 停止失败"
                        if error:
                            msg += f": {error[:50]}"
                        self._show_error(msg)
                except Exception as e:
                    logger.warning(f"[App] Error handling IM event: {e}")

            # 订阅所有 IM 相关事件
            event_bus.subscribe(EventType.IM_SERVICE_STARTED, on_im_event)
            event_bus.subscribe(EventType.IM_SERVICE_STOPPED, on_im_event)
            event_bus.subscribe(EventType.IM_SERVICE_START_FAILED, on_im_event)
            event_bus.subscribe(EventType.IM_SERVICE_STOP_FAILED, on_im_event)

            logger.info("[App] IM event handlers registered")
        except Exception as e:
            logger.warning(f"[App] Failed to setup IM event handlers: {e}")

    def _on_language_changed(self, new_lang: str):
        """语言切换时的回调 - 刷新所有 UI 文本"""
        print(f"[App] Language changed to: {new_lang}")
        if self.page:
            # 刷新命令描述（这些不会被自动刷新）
            self.COMMANDS = {
                "/clear": t("commands.clear"),
                "/context": t("commands.context"),
                "/compress": t("commands.compress"),
                "/reset": t("commands.reset"),
                "/skills": t("commands.skills"),
                "/reload": t("commands.reload"),
                "/history": t("commands.history"),
                "/save": t("commands.save"),
                "/new": t("commands.new"),
                "/load": t("commands.load"),
                "/rename": t("commands.rename"),
                "/delete": t("commands.delete"),
                "/exit": t("commands.exit"),
                "/help": t("commands.help"),
            }

            # 其他组件通过各自的观察者自动刷新
            self.page.update()

    async def initialize(self):
        """Initialize core components"""
        try:
            logger.info("[INIT] Step 1: Starting GUI initialization...")
            self._set_startup_status(t("app.init_core"))

            # Bootstrap is already done in main(), just verify config is loaded
            logger.info("[INIT] Step 2: Verifying bootstrap status...")
            try:
                # Verify config is accessible
                workspace_path = g_config.get_workspace_dir()
                logger.info("[INIT] Bootstrap already completed")
            except RuntimeError:
                # If not loaded, run bootstrap
                logger.info("[INIT] Running bootstrap...")
                await asyncio.to_thread(bootstrap_sync)
                logger.info("[INIT] Bootstrap completed")
                workspace_path = g_config.get_workspace_dir()
            skills_directory = g_config.get_skills_path()
            logger.info(f"[INIT] Workspace path: {workspace_path}")
            logger.info(f"[INIT] Skills directory: {skills_directory}")

            self._set_startup_status(t("app.init_llm"))

            # Initialize LLM
            logger.info("[INIT] Step 3: Initializing LLM...")
            try:
                self._llm = LLMClient()
                logger.info("[INIT] LLM initialized successfully")
            except Exception as llm_error:
                import traceback

                logger.error(f"[INIT] Failed to initialize LLM: {llm_error}")
                logger.error(f"[INIT] Traceback:\n{traceback.format_exc()}")
                raise

            # Create skill gateway
            logger.info("[INIT] Step 4: Creating skill gateway...")
            self._set_startup_status("Loading skill system...")
            try:
                self._skill_gateway = await SkillGateway.from_config()
                logger.info("[INIT] Skill gateway created successfully")
            except Exception as e:
                logger.warning(f"[INIT] Failed to create skill provider: {e}")
                self._skill_gateway = None

            if self._skill_gateway is None:
                manifests = []
                logger.warning(
                    "[INIT] Skill gateway unavailable, continue with no skills"
                )
            else:
                manifests = await self._skill_gateway.discover()
            skill_count = len(manifests)
            logger.info(f"[INIT] Skills loaded: {skill_count} skills")
            skill_names = [m.name for m in manifests]
            logger.info(f"[INIT] Skill names: {skill_names}")
            self._set_startup_status(f"✓ Skill system ready ({skill_count} skills)")

            # Initialize agent
            logger.info("[INIT] Step 6: Initializing MementoSAgent...")
            try:
                self._agent = MementoSAgent(
                    skill_gateway=self._skill_gateway,
                )
                logger.info("[INIT] MementoSAgent initialized successfully")
            except Exception as agent_error:
                import traceback

                logger.error(
                    f"[INIT] Failed to initialize MementoSAgent: {agent_error}"
                )
                logger.error(f"[INIT] Traceback:\n{traceback.format_exc()}")
                raise

            # 2) Load history messages/session
            logger.info("[INIT] Step 7: Loading recent session...")
            self._set_startup_status(t("app.init_session"))
            try:
                await self._load_most_recent_session()
                logger.info("[INIT] Recent session loaded successfully")
            except Exception as session_load_error:
                import traceback

                logger.error(
                    f"[INIT] Failed to load recent session: {session_load_error}"
                )
                logger.error(f"[INIT] Traceback:\n{traceback.format_exc()}")
                # Don't raise here, just log the error

            # 8) Finalize startup
            logger.info("[INIT] Step 8: Finalizing...")
            self._hide_startup_overlay()
            logger.info("[INIT] GUI initialized successfully!")

            # 9) Initialize auto-update
            logger.info("[INIT] Step 9: Initializing auto-update...")
            try:
                self.update_manager = AutoUpdateManager()
                self.update_manager.log_runtime_environment_info()
                notifier = UpdateNotifier(
                    page=self.page,
                    manager=self.update_manager,
                    show_error=self._show_error,
                    show_snackbar=self._show_snackbar,
                )
                self.update_notifier = notifier
                await notifier.initialize()
                logger.info("[INIT] Auto-update initialized")
            except Exception as e:
                logger.warning(f"[INIT] Auto-update init failed: {e}")

        except Exception as e:
            error_msg = t("error.init_failed", error=str(e))
            logger.error(f"[INIT] Failed to initialize: {e}")
            import traceback

            logger.error(f"[INIT] Traceback:\n{traceback.format_exc()}")
            self._show_error(error_msg)
            self._set_startup_status(error_msg)

    def build(self, page: ft.Page):
        """Build the GUI"""
        self.page = page
        self.page.title = "Memento-S"
        self.page.theme_mode = self.theme_mode
        self.page.padding = 0

        # 拦截系统默认的窗口关闭行为，以便执行自定义的清理和强制退出逻辑
        try:
            # Flet 0.21.0 及以上版本的新 API (page.window)
            self.page.window.prevent_close = True
            self._apply_window_icon()

            # 保底：也绑定综合事件 (防止部分 0.21.x 版本遗漏)
            self.page.window.on_event = self._on_window_event

        except AttributeError:
            # 兼容老版本 Flet (0.20 及以下)
            self.page.window_prevent_close = True
            self.page.on_window_event = self._on_window_event

        # 【非常重要】修改 window 属性后，必须调用 update() 推送到前端
        self.page.update()

        # Keyboard shortcuts
        self.page.on_keyboard_event = self._on_keyboard

        # Create UI
        self._create_sidebar()
        self._create_main_area()

        # Record Flet child process IDs for clean exit
        self._record_flet_processes()

        # Create main layout with file browser support
        self.main_layout = MainLayout(
            page=self.page,
            sidebar=self.sidebar,
            main_area=self.main_area,
            workspace_path=g_config.get_workspace_dir(),
        )
        base_layout = self.main_layout.get_layout()

        # Startup waiting overlay (visible immediately)
        self.startup_overlay = ft.Container(
            content=ft.Container(
                content=ft.Column(
                    [
                        ft.ProgressRing(width=36, height=36, stroke_width=3),
                        ft.Text(
                            t("app.startup_title"), size=18, weight=ft.FontWeight.W_600
                        ),
                        ft.Text(
                            t("app.startup_subtitle"), size=13, color=ft.Colors.GREY_400
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=16,
                ),
                width=420,
                height=220,
                bgcolor=ft.Colors.GREY_900,
                border_radius=16,
                padding=20,
                alignment=ft.Alignment.CENTER,
            ),
            expand=True,
            bgcolor=ft.Colors.with_opacity(0.72, ft.Colors.BLACK),
            alignment=ft.Alignment.CENTER,
            visible=True,
        )

        # Layout
        self.page.add(
            ft.Stack(
                [
                    base_layout,
                    self.startup_overlay,
                ],
                expand=True,
            )
        )

        # Initialize
        asyncio.create_task(self.initialize())

    def _apply_window_icon(self):
        """Apply runtime icon to the native desktop window on Windows."""
        if platform.system() != "Windows" or not self.page:
            return

        icon_path = self._resolve_window_icon_path()
        if not icon_path:
            logger.warning("[App] Window icon not found, skipping runtime icon setup")
            return

        try:
            self.page.window.icon = str(icon_path)
            logger.info(f"[App] Applied runtime window icon: {icon_path}")
        except Exception as e:
            logger.warning(f"[App] Failed to apply runtime window icon: {e}")

    def _get_runtime_icon_name(self) -> str:
        """Return the platform-specific desktop icon filename."""
        system = platform.system()
        if system == "Windows":
            return "icon.ico"
        if system == "Darwin":
            return "icon.icns"
        return "icon.png"

    def _resolve_window_icon_path(self) -> Path | None:
        """Resolve the platform-specific icon for both source and packaged runtimes."""
        candidates: list[Path] = []
        icon_name = self._get_runtime_icon_name()

        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / "assets" / icon_name)

        executable = getattr(sys, "executable", None)
        if executable:
            candidates.append(Path(executable).resolve().parent / "assets" / icon_name)

        candidates.append(Path(__file__).resolve().parents[1] / "assets" / icon_name)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return None

    def _create_sidebar(self):
        """Create and initialize sidebar components."""
        self.layout_builder.create_sidebar()
        # Restore login state if previously logged in
        if self.auth_service.is_logged_in:
            self.user_info_bar.update_user_area(
                logged_in=True,
                display_name=self.auth_service.display_name,
            )
            self.layout_builder.update_sidebar_user_state(
                logged_in=True,
                display_name=self.auth_service.display_name,
            )

    def _ensure_login_dialog(self):
        """Lazily create (or recreate) the shared LoginDialog instance."""
        from gui.widgets.login_dialog import LoginDialog

        if self._login_dialog is None or self._login_dialog.page is not self.page:
            self._login_dialog = LoginDialog(
                page=self.page,
                auth_service=self.auth_service,
                on_login_success=self._on_login_success,
            )

    def _on_login_click(self):
        """Handle login button click in sidebar."""
        self._ensure_login_dialog()
        self._login_dialog.show()

    def _on_login_success(self):
        """Handle successful login."""
        self.user_info_bar.update_user_area(
            logged_in=True,
            display_name=self.auth_service.display_name,
        )
        self.layout_builder.update_sidebar_user_state(
            logged_in=True,
            display_name=self.auth_service.display_name,
        )
        self._show_snackbar(t("auth.login_success"))
        self._refresh_settings_panel_if_open()

    def _on_logout_click(self):
        """Handle logout button click in sidebar."""
        asyncio.create_task(self._do_logout())

    async def _do_logout(self):
        """Call logout API and reset UI state."""
        try:
            success, msg = await self.auth_service.logout()
        except Exception as e:
            logger.warning(f"[App] Logout error: {e}")
            success = False
            msg = str(e)

        if success:
            self.user_info_bar.update_user_area(logged_in=False, display_name="")
            self.layout_builder.update_sidebar_user_state(
                logged_in=False, display_name=""
            )
            self._show_snackbar(t("auth.logout_success"))
            self._refresh_settings_panel_if_open()
        else:
            self._show_snackbar(t("auth.logout_failed", error=msg or ""))

    def _refresh_settings_panel_if_open(self):
        """Refresh the settings panel content if it is currently open."""
        try:
            if (
                self.settings_panel
                and self.settings_panel.dialog
                and self.settings_panel.dialog.open
            ):
                self.settings_panel._refresh_content()
        except Exception:
            pass

    def _create_main_area(self):
        """Create main chat area"""
        self.layout_builder.create_main_area()

    # ========== Event Handlers ==========

    def _on_keyboard(self, e: ft.KeyboardEvent):
        """Handle keyboard shortcuts"""
        self.keyboard_controller.on_keyboard(e)

    def _on_window_event(self, e):
        """处理窗口事件，兼容各类 Flet 版本的事件对象结构 (包含 Enum 变化)"""

        # 将事件的 type 和 data 强制转为字符串并全部小写
        e_type_str = str(getattr(e, "type", "")).lower()
        e_data_str = str(getattr(e, "data", "")).lower()

        # logger.info(f"[App] 检查窗口事件 -> type: '{e_type_str}', data: '{e_data_str}'")

        # 判断：如果类型枚举包含 "close"（新版 Flet），或者 data 等于 "close"（旧版 Flet）
        if "close" in e_type_str or e_data_str == "close":
            logger.info(
                "[App] 匹配到窗口关闭 (close) 动作，正在调用 exit_app() 强制退出..."
            )
            self.exit_app()

    def exit_app(self):
        """Public method to properly exit the application"""
        logger.info("[App] Exit requested, starting async shutdown...")

        # 启动异步退出流程（给 Flet 时间发送 update 指令）
        task = asyncio.create_task(self._async_exit())
        # 添加回调以避免 "Task exception was never retrieved" 警告
        task.add_done_callback(lambda t: None)

    async def _async_exit(self):
        """异步退出：先让子进程自己退出，避免窗口闪烁"""
        # 1. 解除拦截并隐藏窗口，让子进程可以自己正常退出
        try:
            if hasattr(self.page, "window"):
                self.page.window.prevent_close = False  # 允许子进程自己退出
                self.page.window.visible = False
            else:
                self.page.window_prevent_close = False
                self.page.window_visible = False
            self.page.update()
            logger.info("[App] 窗口已隐藏，等待子进程自己退出...")
        except Exception as e:
            logger.error(f"[App] 隐藏窗口失败: {e}")

        # 2. 停止正在进行的任务
        if hasattr(self, "_current_task") and self._current_task:
            self._current_task.cancel()

        # 3. 给子进程时间自己退出（避免闪烁）
        await asyncio.sleep(1.0)

        # 4. 强制杀死残留的进程（如果有）
        logger.info("[App] 清理残留进程...")
        self._kill_flet_processes()

        # 5. 主进程退出
        logger.info("[App] 主进程退出")
        import os
        import sys

        if os.name == "nt":
            os._exit(0)
        else:
            sys.exit(0)

    def _kill_flet_processes(self):
        """终止 Flet 前端进程，防止残留（无弹窗）"""
        try:
            import os

            current_pid = os.getpid()
            logger.info(f"[App] 正在清理 Flet 子进程，当前 PID: {current_pid}")

            # 方法1: 使用 psutil（无弹窗，推荐）
            try:
                import psutil

                self._kill_with_psutil()
                return
            except ImportError:
                logger.debug("[App] psutil 未安装，回退到 subprocess 方法")

            # 方法2: 使用 subprocess（隐藏窗口）
            self._kill_with_subprocess()

        except Exception as e:
            logger.error(f"[App] 清理子进程时出错: {e}")

    def _kill_with_psutil(self):
        """使用 psutil 终止进程（无弹窗）"""
        logger.info("[App] 使用 psutil 方式终止进程（无弹窗）")
        try:
            import psutil
            import os

            current_pid = os.getpid()

            # 终止记录的 PID
            for pid in self._flet_process_ids:
                try:
                    proc = psutil.Process(pid)
                    proc.terminate()
                    proc.wait(timeout=1)
                    logger.info(f"[App] 已终止进程 PID: {pid}")
                except psutil.NoSuchProcess:
                    pass
                except Exception as e:
                    logger.warning(f"[App] 终止进程 PID {pid} 失败: {e}")

            # 按名称查找并终止
            for proc in psutil.process_iter(["pid", "name", "ppid"]):
                try:
                    proc_info = proc.info
                    proc_name = proc_info.get("name", "").lower()
                    proc_ppid = proc_info.get("ppid")

                    if ("flet" in proc_name or "flutter" in proc_name) and proc_info[
                        "pid"
                    ] != current_pid:
                        if proc_ppid == current_pid or "flet" in proc_name:
                            proc.terminate()
                            logger.info(
                                f"[App] 已终止 {proc_name} (PID: {proc_info['pid']})"
                            )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

        except Exception as e:
            logger.error(f"[App] psutil 终止失败: {e}")
            raise

    def _kill_with_subprocess(self):
        """使用 subprocess 终止进程（隐藏窗口，无弹窗）"""
        logger.info("[App] 使用 subprocess 方式终止进程（隐藏窗口）")
        import subprocess
        import os

        # Windows: 使用 creationflags 隐藏窗口
        if os.name == "nt":
            # 隐藏窗口的标志
            CREATE_NO_WINDOW = 0x08000000

            # 终止记录的 PID
            for pid in self._flet_process_ids:
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/PID", str(pid)],
                        capture_output=True,
                        timeout=2,
                        creationflags=CREATE_NO_WINDOW,
                    )
                    logger.info(f"[App] 已终止进程 PID: {pid}")
                except Exception as e:
                    logger.warning(f"[App] 终止进程 PID {pid} 失败: {e}")

            # 按名称终止
            for proc_name in ["flet.exe", "flutter_windows.exe"]:
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/IM", proc_name, "/T"],
                        capture_output=True,
                        timeout=2,
                        creationflags=CREATE_NO_WINDOW,
                    )
                    logger.info(f"[App] 已终止 {proc_name}")
                except:
                    pass
        else:
            # Unix: 直接终止
            for pid in self._flet_process_ids:
                try:
                    subprocess.run(
                        ["kill", "-9", str(pid)],
                        capture_output=True,
                        timeout=2,
                    )
                except:
                    pass

    def _record_flet_processes(self):
        """Record Flet child process IDs on startup for clean exit"""
        try:
            import os
            import subprocess

            current_pid = os.getpid()
            self._flet_process_ids = []

            if os.name == "nt":
                # Windows: Use wmic to find child processes
                try:
                    result = subprocess.run(
                        [
                            "wmic",
                            "process",
                            "where",
                            f"ParentProcessId={current_pid}",
                            "get",
                            "ProcessId",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    # Parse output to get PIDs
                    lines = result.stdout.strip().split("\n")
                    for line in lines[1:]:  # Skip header
                        pid_str = line.strip()
                        if pid_str and pid_str.isdigit():
                            pid = int(pid_str)
                            if pid != current_pid:
                                self._flet_process_ids.append(pid)
                                logger.info(f"[App] 记录子进程 PID: {pid}")
                except Exception as e:
                    logger.warning(f"[App] 记录子进程失败: {e}")
            else:
                # Unix: Use ps to find child processes
                try:
                    result = subprocess.run(
                        ["pgrep", "-P", str(current_pid)],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    for line in result.stdout.strip().split("\n"):
                        if line:
                            try:
                                pid = int(line.strip())
                                self._flet_process_ids.append(pid)
                                logger.info(f"[App] 记录子进程 PID: {pid}")
                            except:
                                pass
                except Exception as e:
                    logger.warning(f"[App] 记录子进程失败: {e}")

            logger.info(f"[App] 共记录 {len(self._flet_process_ids)} 个子进程")

        except Exception as e:
            logger.error(f"[App] 记录进程信息时出错: {e}")

    async def _on_send_message(self, e=None):
        """Handle send button click"""
        await self.input_controller.on_send_message(e)

    def _on_input_change(self, e):
        """Show/hide command hints when user types / commands"""
        self.input_controller.on_input_change(e)

    async def _on_command_hint_selected(self, command: str):
        """Auto-fill input with selected command"""
        await self.input_controller.on_command_hint_selected(command)

    def _update_command_hints_highlight(self):
        """Update visual highlighting of command hints and auto-scroll to selected"""
        self.input_controller.update_command_hints_highlight()

    async def _send_current_message(self):
        """Send current message"""
        await self.message_controller.send_current_message()

    # Note: process_message is now handled within send_current_message

    def _on_stop_generation(self, e):
        """Stop generation immediately"""
        self.message_controller.stop_generation()

    # ========== Session & Conversation Management ==========

    async def _ensure_session(self):
        """Ensure current session exists"""
        return await self.conversation_controller.ensure_session()

    def _on_new_chat(self):
        """Start new chat - creates new session"""
        asyncio.create_task(self.conversation_controller.on_new_chat())

    async def _on_select_session(self, session_id: str):
        """Load session and its conversations"""
        await self.conversation_controller.on_select_session(session_id)

    async def _on_delete_session(self, session_id: str):
        """Delete session"""
        await self.conversation_controller.on_delete_session(session_id)

    async def _on_rename_session(self, session_id: str):
        """Rename session"""
        await self.conversation_controller.on_rename_session(session_id)

    async def _on_load_more_sessions(self):
        """Load more sessions (pagination)"""
        await self.conversation_controller.load_more_sessions()

    async def _load_most_recent_session(self):
        """Load most recent session on startup"""
        return await self.conversation_controller.load_most_recent_session()

    async def _generate_conversation_title(self, content: str = None):
        """Generate title using LLM"""
        await self.conversation_controller.generate_conversation_title(content)

    def _get_current_model(self) -> str:
        """Get current LLM model name"""
        if g_config and g_config.llm and g_config.llm.current:
            return g_config.llm.current.model
        return ""

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for mixed language text."""
        from utils.token_utils import count_tokens

        return count_tokens(text)

    # ========== Command Handling ==========

    async def _handle_command(self, cmd: str):
        """Handle slash commands"""
        await self.command_controller.handle_command(cmd)

    # ========== UI Helpers ==========

    def _on_clear_chat(self):
        """Clear chat display"""
        self.ui_feedback_controller.on_clear_chat()

    def _add_system_message(self, text: str):
        """Add system message"""
        self.ui_feedback_controller.add_system_message(text)

    def _update_token_display(self, current: int = None, max_tokens: int | None = None):
        """Update token display in bottom info bar"""
        if max_tokens is None:
            max_tokens = g_config.llm.current_profile.input_budget
        self.ui_feedback_controller.update_token_display(current, max_tokens)

    def _update_toolbar_title(self, title: str = None):
        """Update toolbar title with conversation title"""
        self.ui_feedback_controller.update_toolbar_title(title)

    def _set_status(self, text: str):
        """Update status bar"""
        self.ui_feedback_controller.set_status(text)

    def _on_settings_saved(self):
        """Callback executed when settings are saved."""
        try:
            # 刷新模型选择器
            if hasattr(self, "layout_builder"):
                self.layout_builder.refresh_model_selector()

            # 重新初始化LLM客户端（配置可能已更改）
            if self._llm:
                self._llm._load_config()
                logger.info(f"[App] LLM config reloaded after settings saved")

            # 重新初始化Agent的LLM
            if self._agent:
                from middleware.llm import LLMClient

                self._agent.llm = LLMClient()
                logger.info(f"[App] Agent LLM reinitialized after settings saved")

            self._show_snackbar(t("settings.saved"))
        except Exception as e:
            logger.error(f"Error updating model selector in UI: {e}")
            self._show_error(t("settings.update_failed"))

    async def _on_model_changed(self, profile_name: str):
        """处理模型切换

        Args:
            profile_name: 新选择的模型profile名称
        """
        try:
            logger.info(f"[App] Model changed to: {profile_name}")

            # 重新初始化LLM客户端
            if self._llm:
                # 重新加载配置
                self._llm._load_config()
                logger.info(f"[App] LLM client reloaded with new config")

            # 重新初始化Agent的LLM（如果存在）
            if self._agent:
                # 创建新的LLMClient实例，让Agent使用新的配置
                from middleware.llm import LLMClient

                self._agent.llm = LLMClient()
                logger.info(f"[App] Agent LLM reinitialized with new config")

            # 刷新模型选择器UI
            if hasattr(self, "layout_builder"):
                self.layout_builder.refresh_model_selector()

            # 更新状态栏显示
            self._update_toolbar_title()

        except Exception as e:
            logger.error(f"[App] Error handling model change: {e}")
            self._show_error(t("settings.model_change_failed", error=str(e)))

    def _show_settings(self):
        """Show settings dialog - create new instance each time"""
        self.settings_panel = SettingsPanel(
            page=self.page,
            show_error=self._show_error,
            show_snackbar=self._show_snackbar,
            on_save_callback=self._on_settings_saved,
            update_manager=self.update_manager,
            auth_service=self.auth_service,
            on_login_click=self._on_login_click,
            on_logout_click=self._on_logout_click,
        )
        self.settings_panel.show()

    def _show_settings_with_category(self, category: str):
        """Show settings dialog with specific category selected

        Args:
            category: Category name to show, e.g., "大模型", "通用"
        """
        self.settings_panel = SettingsPanel(
            page=self.page,
            show_error=self._show_error,
            show_snackbar=self._show_snackbar,
            on_save_callback=self._on_settings_saved,
            update_manager=self.update_manager,
            auth_service=self.auth_service,
            on_login_click=self._on_login_click,
            on_logout_click=self._on_logout_click,
        )
        self.settings_panel.show(default_category=category)

    def _set_startup_status(self, message: str):
        """Update startup overlay status text"""
        if (
            self.startup_overlay
            and self.startup_overlay.content
            and hasattr(self.startup_overlay.content, "content")
        ):
            panel = self.startup_overlay.content
            if panel and panel.content and hasattr(panel.content, "controls"):
                controls = panel.content.controls
                if len(controls) >= 3 and isinstance(controls[2], ft.Text):
                    controls[2].value = message
                    if self.page:
                        self.page.update()

    def _hide_startup_overlay(self):
        """Hide startup waiting overlay"""
        if self.startup_overlay:
            self.startup_overlay.visible = False
            if self.page:
                self.page.update()

    def _on_export_chat(self):
        """Export conversation"""
        self._add_system_message(t("status.export_not_implemented"))

    def _on_markket(self):
        """Open Markket dialog"""
        from gui.widgets.market_dialog import MarkketDialog

        dialog = MarkketDialog(self)
        dialog.show()

    def _show_error(self, message: str):
        """Show error"""
        self.ui_feedback_controller.show_error(message)

    def _show_snackbar(self, message: str, type: str = "info", duration: int = 3000):
        """显示 Toast 提示

        Args:
            message: 提示消息
            type: 消息类型 - "info"/"warning"/"error"，默认 "info"
            duration: 显示时长（毫秒），默认 3000ms（3秒）
        """
        self.ui_feedback_controller.show_snackbar(message, type=type, duration=duration)

    def _show_wechat_login_dialog(self, is_relogin: bool = False):
        """显示微信登录对话框

        Args:
            is_relogin: 是否为重新登录
        """
        try:
            from gui.widgets.wechat_login_dialog import WechatLoginDialog
            from middleware.config import g_config
            from middleware.im.gateway_starter import get_gateway_manager

            def on_success(token: str):
                logger.info("[App] WeChat login successful, saving token")
                try:
                    # 保存 token
                    g_config.set("im.wechat.token", token, save=True)

                    # 重新启动微信渠道（需要先停止再启动，确保适配器重新初始化）
                    gateway_mgr = get_gateway_manager()
                    if gateway_mgr and gateway_mgr.is_running:
                        logger.info("[App] Restarting WeChat channel after login")

                        # 在后台线程中异步重启，不阻塞 UI
                        import threading

                        def restart_channel():
                            try:
                                # 先禁用微信渠道，触发停止
                                g_config.set("im.wechat.enabled", False, save=True)
                                gateway_mgr.refresh_channels_sync()
                                logger.info("[App] WeChat channel stopped")

                                # 等待确保停止完成
                                import time

                                time.sleep(1)

                                # 再启用微信渠道，触发启动（使用新 token）
                                g_config.set("im.wechat.enabled", True, save=True)
                                gateway_mgr.refresh_channels_sync()
                                logger.info("[App] WeChat channel restarted")
                            except Exception as e:
                                logger.error(
                                    f"[App] Failed to restart WeChat channel: {e}"
                                )

                        threading.Thread(target=restart_channel, daemon=True).start()

                    # 重置微信会话过期状态
                    if self.settings_panel:
                        self.settings_panel.set_wechat_session_expired(False)
                        logger.info(
                            "[App] Reset wechat session expired status after successful login"
                        )

                    self._show_snackbar("微信登录成功！渠道正在重启...")
                except Exception as e:
                    logger.error(f"[App] Failed to save WeChat token: {e}")
                    self._show_error(f"保存Token失败: {e}")

            def on_failed(error: str):
                logger.error(f"[App] WeChat login failed: {error}")
                self._show_error(f"微信登录失败: {error}")

            dialog = WechatLoginDialog(
                page=self.page,
                on_login_success=on_success,
                on_login_failed=on_failed,
                is_relogin=is_relogin,
            )
            dialog.show()

        except Exception as e:
            logger.error(f"[App] Failed to show WeChat login dialog: {e}")
            self._show_error(f"无法显示登录对话框: {e}")

    def _toggle_file_browser(self):
        """Toggle file browser drawer visibility"""
        if hasattr(self, "main_layout"):
            self.main_layout.toggle_file_browser()


def main():
    """Entry point"""
    import multiprocessing

    multiprocessing.freeze_support()

    import bootstrap as _bootstrap

    _bootstrap.bootstrap_sync()

    # 使用 utils.logger 初始化日志（支持按天记录）
    from utils.logger import setup_logger

    setup_logger(log_file="gui", enable_console=True)

    gui = MementoSGUIImproved()
    icon_path = gui._resolve_window_icon_path()

    def on_page_init(page: ft.Page):
        gui.build(page)

    ft.run(
        main=on_page_init,
        assets_dir=str(icon_path.parent) if icon_path else "assets",
    )


if __name__ == "__main__":
    main()
