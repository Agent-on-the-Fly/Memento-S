"""
Webhook HTTP 服务器。

提供统一的 Webhook 端点：
- POST /webhook/{channel_type}/{account_id}

适配器只需实现 parse_webhook 和 verify_webhook 方法。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable

from aiohttp import web

from .protocol import (
    ChannelType,
    GatewayMessage,
)

logger = logging.getLogger(__name__)


class WebhookHandler:
    """Webhook 处理器，负责处理特定渠道的 Webhook 请求。"""

    def __init__(
        self,
        channel_type: ChannelType,
        account_id: str,
        adapter: Any,
    ):
        self.channel_type = channel_type
        self.account_id = account_id
        self.adapter = adapter
        self._message_callback: Callable[[list[GatewayMessage]], None] | None = None

    def on_message(self, callback: Callable[[list[GatewayMessage]], None]) -> None:
        """设置消息回调。"""
        self._message_callback = callback

    async def handle(self, request: web.Request) -> web.Response:
        """处理 Webhook 请求。"""
        try:
            body = await request.read()
            signature = self._extract_signature(request)

            # 验证签名
            if signature:
                headers = dict(request.headers)
                valid = await self.adapter.verify_webhook(signature, body, headers)
                if not valid:
                    logger.warning(
                        "Webhook signature verification failed: %s/%s",
                        self.channel_type.value,
                        self.account_id,
                    )
                    return web.Response(status=401, text="Unauthorized")

            # 解析消息
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                payload = {"raw": body.decode("utf-8", errors="replace")}

            messages = await self.adapter.parse_webhook(payload)

            # 触发回调
            if messages and self._message_callback:
                try:
                    result = self._message_callback(messages)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error("Webhook callback error: %s", e)

            return self._create_success_response()

        except Exception as e:
            logger.exception("Webhook handler error: %s", e)
            return web.Response(status=500, text="Internal Server Error")

    def _extract_signature(self, request: web.Request) -> str:
        """提取签名。"""
        return (
            request.headers.get("X-Hub-Signature-256", "")
            or request.headers.get("X-Hub-Signature", "")
            or request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
            or request.headers.get("X-Signature", "")
        )

    def _create_success_response(self) -> web.Response:
        """创建成功响应。"""
        if self.channel_type == ChannelType.FEISHU:
            return web.json_response({"code": 0, "msg": "success"})
        return web.Response(status=200, text="OK")


class WebhookServer:
    """Webhook HTTP 服务器，提供统一的 Webhook 端点入口。"""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        base_path: str = "/webhook",
    ):
        self.host = host
        self.port = port
        self.base_path = base_path.rstrip("/")
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._handlers: dict[tuple[str, str], WebhookHandler] = {}
        self._health_callback: Callable[[], dict] | None = None

    async def start(self) -> None:
        """启动 Webhook 服务器。"""
        self._app = web.Application()
        self._app.router.add_post(
            "/webhook/{channel_type}/{account_id}",
            self._handle_webhook,
        )
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get("/", self._handle_root)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()

        logger.info("Webhook server started: http://%s:%d", self.host, self.port)

    async def stop(self) -> None:
        """停止 Webhook 服务器。"""
        if self._runner:
            await self._runner.cleanup()
        self._app = None
        self._runner = None
        self._site = None
        logger.info("Webhook server stopped")

    def register(
        self,
        channel_type: ChannelType,
        account_id: str,
        adapter: Any,
    ) -> WebhookHandler:
        """注册 Webhook 处理器。"""
        key = (channel_type.value, account_id)
        if key in self._handlers:
            logger.warning("Replacing existing webhook handler: %s", key)

        handler = WebhookHandler(channel_type, account_id, adapter)
        self._handlers[key] = handler

        logger.info(
            "Webhook handler registered: /webhook/%s/%s",
            channel_type.value,
            account_id,
        )
        return handler

    def unregister(self, channel_type: ChannelType, account_id: str) -> None:
        """注销 Webhook 处理器。"""
        key = (channel_type.value, account_id)
        self._handlers.pop(key, None)
        logger.info(
            "Webhook handler unregistered: /webhook/%s/%s",
            channel_type.value,
            account_id,
        )

    def get_handler(self, channel_type: str, account_id: str) -> WebhookHandler | None:
        """获取处理器。"""
        return self._handlers.get((channel_type, account_id))

    def set_health_callback(self, callback: Callable[[], dict]) -> None:
        """设置健康检查回调。"""
        self._health_callback = callback

    async def _handle_webhook(self, request: web.Request) -> web.Response:
        """处理 Webhook 请求。"""
        channel_type = request.match_info.get("channel_type", "")
        account_id = request.match_info.get("account_id", "")
        handler = self.get_handler(channel_type, account_id)

        if not handler:
            logger.warning("No handler for webhook: %s/%s", channel_type, account_id)
            return web.Response(status=404, text="Not Found")

        return await handler.handle(request)

    async def _handle_health(self, request: web.Request) -> web.Response:
        """处理健康检查请求。"""
        if self._health_callback:
            try:
                health = self._health_callback()
                if asyncio.iscoroutine(health):
                    health = await health
                return web.json_response(health)
            except Exception as e:
                logger.error("Health check error: %s", e)
                return web.json_response(
                    {"status": "error", "message": str(e)},
                    status=500,
                )

        return web.json_response({
            "status": "healthy",
            "handlers": len(self._handlers),
        })

    async def _handle_root(self, request: web.Request) -> web.Response:
        """处理根路径请求。"""
        return web.json_response({
            "service": "Memento-S Gateway Webhook Server",
            "version": "1.0.0",
            "endpoints": [
                "POST /webhook/{channel_type}/{account_id}",
                "GET /health",
            ],
        })

    def get_stats(self) -> dict:
        """获取统计信息。"""
        return {
            "host": self.host,
            "port": self.port,
            "handlers": len(self._handlers),
            "registered_channels": [
                {"channel": ct, "account": aid}
                for (ct, aid) in self._handlers.keys()
            ],
        }

    def get_webhook_url(
        self,
        channel_type: ChannelType,
        account_id: str,
        base_url: str = "",
    ) -> str:
        """获取 Webhook URL。"""
        path = f"{self.base_path}/{channel_type.value}/{account_id}"
        if base_url:
            return f"{base_url.rstrip('/')}{path}"
        return f"http://{self.host}:{self.port}{path}"
