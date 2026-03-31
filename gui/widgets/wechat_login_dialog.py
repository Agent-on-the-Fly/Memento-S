import asyncio
import base64
import io
import logging
import sys
from pathlib import Path
from typing import Callable, Optional

import flet as ft

from utils.logger import get_logger

logger = get_logger(__name__)

TRANSPARENT_PIXEL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


class WechatLoginDialog:
    def __init__(
        self,
        page: ft.Page,
        on_login_success: Optional[Callable[[str], None]] = None,
        on_login_failed: Optional[Callable[[str], None]] = None,
        is_relogin: bool = False,
    ):
        self.page = page
        self.on_login_success = on_login_success
        self.on_login_failed = on_login_failed
        self.is_relogin = is_relogin

        self._dialog: Optional[ft.AlertDialog] = None
        self._login_task: Optional[asyncio.Task] = None
        self._timer_task: Optional[asyncio.Task] = None
        self._is_open = False
        self._login_mgr = None
        self._qr_ticket = None

        self._status_text = ft.Text(
            "正在获取二维码...",
            size=14,
            color="#cccccc",
            text_align=ft.TextAlign.CENTER,
        )
        self._qr_image = ft.Image(
            src=TRANSPARENT_PIXEL,
            width=200,
            height=200,
            fit="contain",
            visible=False,
        )
        self._progress_ring = ft.ProgressRing(
            width=40,
            height=40,
            stroke_width=3,
            color="#3b82f6",
            visible=True,
        )
        self._timer_text = ft.Text(
            "二维码有效中...",
            size=12,
            color="#888888",
        )

    def show(self):
        if self._is_open:
            return

        self._is_open = True
        self._reset_ui()

        close_btn = ft.IconButton(
            icon=ft.Icons.CLOSE,
            icon_color="#a0a0a0",
            icon_size=18,
            on_click=self._on_close_click,
            tooltip="关闭",
            style=ft.ButtonStyle(padding=ft.Padding(4, 4, 4, 4)),
        )

        title_text = "微信重新登录" if self.is_relogin else "微信扫码登录"
        title_bar = ft.Row(
            [
                ft.Text(
                    title_text, size=16, weight=ft.FontWeight.W_600, color="#e0e0e0"
                ),
                close_btn,
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        )

        content = ft.Column(
            [
                title_bar,
                ft.Container(height=20),
                ft.Container(
                    content=self._progress_ring, alignment=ft.alignment.Alignment(0, 0)
                ),
                ft.Container(
                    content=self._qr_image, alignment=ft.alignment.Alignment(0, 0)
                ),
                ft.Container(height=12),
                self._status_text,
                ft.Container(height=8),
                self._timer_text,
                ft.Container(height=12),
                ft.Text("请使用微信扫描二维码登录", size=12, color="#666666"),
            ],
            spacing=0,
            tight=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )

        dialog_card = ft.Container(
            content=content,
            width=320,
            padding=ft.Padding(24, 24, 24, 24),
            bgcolor="#1e1e1e",
            border_radius=8,
            border=ft.border.all(1, "#383838"),
        )

        self._dialog = ft.AlertDialog(
            modal=True,
            bgcolor=ft.Colors.with_opacity(0.5, ft.Colors.BLACK),
            content_padding=0,
            inset_padding=20,
            content=dialog_card,
        )

        self.page.show_dialog(self._dialog)
        self.page.update()

        self._start_login()

    def _reset_ui(self):
        self._status_text.value = "正在获取二维码..."
        self._status_text.color = "#cccccc"
        self._qr_image.visible = False
        self._progress_ring.visible = True
        self._timer_text.value = "二维码有效中..."
        self._timer_text.visible = True

    def _on_close_click(self, e):
        self.close()

    def close(self):
        logger.info("[WechatLoginDialog] Closing dialog")
        self._is_open = False

        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()
        if self._login_task and not self._login_task.done():
            self._login_task.cancel()

        if self._dialog:
            try:
                self.page.pop_dialog()
                logger.info("[WechatLoginDialog] Dialog popped from page")
            except Exception as e:
                logger.error(f"[WechatLoginDialog] Error closing dialog: {e}")
            self._dialog = None

        # 强制刷新页面以确保对话框关闭
        try:
            self.page.update()
        except Exception as e:
            logger.warning(f"[WechatLoginDialog] Page update error: {e}")

    def _start_login(self):
        try:
            self._login_task = asyncio.create_task(self._do_login())
        except RuntimeError:
            loop = asyncio.get_event_loop()
            self._login_task = loop.create_task(self._do_login())

    def _start_timer(self, start_time: float, timeout: float):
        """后台监控超时，不在 UI 显示倒计时（服务器控制实际过期时间）。"""

        async def timer_loop():
            while self._is_open:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout:
                    # 超时了，但让 poll_status 来处理过期状态
                    break
                await asyncio.sleep(5)

        try:
            self._timer_task = asyncio.create_task(timer_loop())
        except RuntimeError:
            loop = asyncio.get_event_loop()
            self._timer_task = loop.create_task(timer_loop())

    def _page_update_safe(self):
        try:
            self.page.update()
        except Exception:
            pass

    async def _do_login(self):
        retry_count = 0
        max_retries = 3

        while retry_count < max_retries and self._is_open:
            try:
                await self._try_login_once()
                break
            except asyncio.CancelledError:
                logger.info("[WechatLoginDialog] Login cancelled")
                break
            except Exception as e:
                retry_count += 1
                logger.error(f"[WechatLoginDialog] Attempt {retry_count} failed: {e}")
                if retry_count >= max_retries:
                    self._status_text.value = f"获取失败: {str(e)[:25]}"
                    self._status_text.color = "#ff6b6b"
                    self._progress_ring.visible = False
                    self._page_update_safe()
                    if self.on_login_failed:
                        self.on_login_failed(str(e))
                else:
                    self._status_text.value = f"重试中...({retry_count}/{max_retries})"
                    self._page_update_safe()
                    await asyncio.sleep(2)

    async def _try_login_once(self):
        _3RD_DIR = Path(__file__).resolve().parent.parent.parent / "3rd"
        if str(_3RD_DIR) not in sys.path:
            sys.path.insert(0, str(_3RD_DIR))

        from weixin_sdk.auth.qr_login import QRLoginManager
        import qrcode

        self._login_mgr = QRLoginManager("https://ilinkai.weixin.qq.com")

        self._status_text.value = "正在获取二维码..."
        self._page_update_safe()

        qr_response = await self._login_mgr.fetch_qr_code()
        self._qr_ticket = qr_response.ticket

        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=2,
        )
        qr.add_data(qr_response.qr_url)
        qr.make(fit=True)

        img_buffer = io.BytesIO()
        img = qr.make_image(fill_color="black", back_color="white")
        img.save(img_buffer, format="PNG")
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

        self._qr_image.src = f"data:image/png;base64,{img_base64}"
        self._qr_image.visible = True
        self._progress_ring.visible = False
        self._status_text.value = "请使用微信扫描二维码"
        self._status_text.color = "#3b82f6"
        self._page_update_safe()

        # 启动独立计时器任务
        start_time = asyncio.get_event_loop().time()
        self._start_timer(start_time, 300)

        # 轮询登录状态
        while self._is_open:
            try:
                status, data = await self._login_mgr.poll_status(self._qr_ticket)

                if status.value == "wait":
                    self._status_text.value = "等待扫码..."
                elif status.value == "scaned":
                    self._status_text.value = "已扫码，请在手机上确认"
                    self._status_text.color = "#ffd93d"
                elif status.value == "confirmed":
                    await self._handle_confirmed(data)
                    return
                elif status.value == "expired":
                    self._status_text.value = "二维码已过期，请关闭后重新登录"
                    self._status_text.color = "#ff6b6b"
                    self._timer_text.visible = False
                    self._page_update_safe()
                    return

                self._page_update_safe()

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"[WechatLoginDialog] Poll error: {e}")

    async def _handle_confirmed(self, data):
        bot_token = data.get("bot_token") if data else None
        if bot_token:
            self._status_text.value = "登录成功！"
            self._status_text.color = "#6bcf7f"
            self._timer_text.visible = False
            self._page_update_safe()
            await asyncio.sleep(1)
            self.close()
            if self.on_login_success:
                self.on_login_success(bot_token)
        else:
            self._status_text.value = "登录失败：未获取到token"
            self._status_text.color = "#ff6b6b"
            self._page_update_safe()
            if self.on_login_failed:
                self.on_login_failed("未获取到token")
