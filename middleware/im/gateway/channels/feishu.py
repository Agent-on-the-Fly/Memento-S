"""
飞书渠道适配器。

复用现有的 im_platform/feishu 长连接实现：
- platform.py: API 主动调用
- receiver.py: WebSocket 长连接接收消息

支持连接模式：
- websocket: 使用 lark-oapi SDK 的 WebSocket 长连接（默认）
- webhook: 使用 Webhook 回调
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from ..gateway import register_channel
from ..protocol import (
    ChannelCapability,
    ChannelType,
    ConnectionConfig,
    ConnectionMode,
    GatewayMessage,
)
from .base import BaseChannelAdapter
from middleware.im.im_platform.feishu.platform import FeishuPlatform
from middleware.im.im_platform.feishu.receiver import FeishuReceiver

logger = logging.getLogger(__name__)

# 抑制 lark SDK 的正常关闭错误日志（ConnectionClosedOK 是正常关闭，不应作为错误）
logging.getLogger("lark").setLevel(logging.CRITICAL)


@register_channel(ChannelType.FEISHU)
class FeishuAdapter(BaseChannelAdapter):
    """
    飞书渠道适配器。

    复用 im_platform/feishu 的实现：
    - Platform: API 主动调用
    - Receiver: WebSocket 长连接接收
    """

    channel_type = ChannelType.FEISHU
    capabilities = [
        ChannelCapability.TEXT,
        ChannelCapability.RICH_TEXT,
        ChannelCapability.IMAGE,
        ChannelCapability.VIDEO,
        ChannelCapability.AUDIO,
        ChannelCapability.FILE,
        ChannelCapability.INTERACTIVE,
        ChannelCapability.REPLY,
        ChannelCapability.EDIT,
        ChannelCapability.DELETE,
    ]

    supported_modes = [
        ConnectionMode.WEBSOCKET,  # 默认使用 WebSocket 长连接
        ConnectionMode.WEBHOOK,
        ConnectionMode.HYBRID,
    ]

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        encrypt_key: str = "",
        verification_token: str = "",
    ):
        super().__init__()
        self.app_id = app_id
        self.app_secret = app_secret
        self.encrypt_key = encrypt_key
        self.verification_token = verification_token

        # im_platform 组件
        self._platform: Any = None  # FeishuPlatform
        self._receiver: Any = None  # FeishuReceiver

        # 消息回调
        self._on_message_callback: Callable[[dict], None] | None = None

        # 停止标志：用于在停止过程中阻止消息处理
        self._stopping = False

        # 启动时间戳：用于过滤积压的旧消息
        self._start_time: float = 0

    # ---- 生命周期 ----

    async def _do_initialize(
        self,
        config: ConnectionConfig,
        mode: ConnectionMode,
    ) -> None:
        """初始化飞书适配器。"""
        logger.info(
            "[FeishuAdapter] _do_initialize called, app_id=%s, mode=%s",
            self.app_id,
            mode.value,
        )
        # 导入 im_platform 组件
        # 创建 Platform 实例（从配置文件读取凭证）
        self._platform = FeishuPlatform()

        logger.info(
            "[FeishuAdapter] ✓ _do_initialize completed: app_id=%s, mode=%s",
            self.app_id,
            mode.value,
        )

    async def _do_start(self) -> None:
        """启动长连接。"""
        import time
        self._stopping = False
        self._start_time = time.time()  # 记录启动时间
        logger.info("[FeishuAdapter] _do_start called, mode=%s", self._mode)

        if self._mode in (ConnectionMode.WEBSOCKET, ConnectionMode.HYBRID):
            # 如果已有接收器，先停止它并等待完全停止
            if self._receiver:
                logger.warning(
                    "[FeishuAdapter] Stopping existing receiver before restart"
                )
                self._receiver.stop()
                self._receiver = None
                import asyncio

                # 等待旧连接完全断开
                await asyncio.sleep(2.0)

            await self._start_websocket_receiver()

        logger.info("[FeishuAdapter] _do_start completed")

    async def _do_stop(self) -> None:
        """停止长连接。"""
        self._stopping = True
        logger.info("[FeishuAdapter] _do_stop called, stopping receiver")

        if self._receiver:
            self._receiver.stop()
            self._receiver = None
            import asyncio

            # 等待连接完全断开，防止消息泄漏
            await asyncio.sleep(2.0)

        logger.info("[FeishuAdapter] _do_stop completed, _stopping=%s", self._stopping)

    async def health_check(self) -> bool:
        """健康检查。"""
        if not self._platform:
            return False
        try:
            # 尝试获取 token 验证连接
            await self._platform._get_token()
            return True
        except Exception:
            return False

    # ---- WebSocket 长连接 ----

    async def _start_websocket_receiver(self) -> None:
        """启动 WebSocket 长连接接收器。"""
        import time

        # 保存当前 receiver 的 ID 用于验证
        current_receiver_id = id(self._receiver) if self._receiver else 0

        # 创建消息回调
        adapter_self = self
        adapter_start_time = self._start_time  # 捕获启动时间
        adapter_instance_id = id(self)  # 捕获 adapter 实例 ID

        # 预先定义 receiver_ref，稍后赋值
        receiver_ref = [None]  # 使用列表以便在闭包中修改

        def on_message(msg_dict: dict) -> None:
            """处理接收到的消息。
            
            注意：飞书长连接有超时重推机制，需要在 3 秒内处理完成且不抛出异常，
            否则会触发重推。因此即使我们不处理消息，也必须正常返回（不抛出异常）。
            """
            try:
                logger.info(
                    "[FeishuAdapter] on_message called: adapter_instance=%d, msg_id=%s",
                    adapter_instance_id,
                    msg_dict.get("id", "unknown"),
                )

                # 检查是否正在停止
                if adapter_self._stopping:
                    logger.info(
                        "[FeishuAdapter] Message ignored (adapter stopping): msg_id=%s",
                        msg_dict.get("id", "unknown"),
                    )
                    return  # 正常返回，不抛出异常，避免触发重推

                # 检查 receiver 是否是最新的
                current_receiver = adapter_self._receiver
                saved_receiver = receiver_ref[0]
                if saved_receiver is None:
                    logger.info(
                        "[FeishuAdapter] Message ignored (no receiver): msg_id=%s",
                        msg_dict.get("id", "unknown"),
                    )
                    return
                if current_receiver is None or current_receiver is not saved_receiver:
                    logger.info(
                        "[FeishuAdapter] Message ignored (receiver mismatch): msg_id=%s",
                        msg_dict.get("id", "unknown"),
                    )
                    return

                # 检查消息时间戳
                msg_time_str = msg_dict.get("create_time", "")
                msg_time = 0
                if msg_time_str:
                    try:
                        msg_time = int(msg_time_str)
                        if msg_time > 10000000000:  # 毫秒时间戳
                            msg_time = msg_time / 1000
                    except (ValueError, TypeError):
                        pass

                current_time = time.time()
                logger.info(
                    "[FeishuAdapter] Message timing: msg_time=%.2f, start_time=%.2f, diff=%.2f",
                    msg_time, adapter_start_time,
                    current_time - msg_time if msg_time > 0 else 0
                )

                if msg_time > 0 and adapter_start_time > 0:
                    if msg_time < adapter_start_time - 5:  # 允许5秒时钟偏差
                        logger.info(
                            "[FeishuAdapter] Message ignored (too old): msg_time=%.2f, start_time=%.2f",
                            msg_time, adapter_start_time
                        )
                        return

                gateway_msg = adapter_self._convert_to_gateway_message(msg_dict)
                logger.info(
                    "[FeishuAdapter] Emitting message: sender_id=%s, chat_id=%s",
                    gateway_msg.sender_id,
                    gateway_msg.chat_id,
                )
                adapter_self._emit_message(gateway_msg)

            except Exception as e:
                # 记录错误但不要抛出异常，避免触发飞书的重推机制
                logger.error(
                    "[FeishuAdapter] Error processing message (not re-raising to avoid retry): %s",
                    e
                )

        # 创建 Receiver（显式传递凭证）
        self._receiver = FeishuReceiver(
            on_message=on_message,
            app_id=self.app_id,
            app_secret=self.app_secret,
        )

        # 保存 receiver 引用用于回调验证
        receiver_ref[0] = self._receiver

        # 在后台线程启动（飞书 SDK 使用阻塞模式）
        self._receiver.start_in_background()

        logger.info(
            "[FeishuAdapter] WebSocket receiver started: app_id=%s, adapter_instance=%d, receiver_id=%d, start_time=%.2f",
            self.app_id, adapter_instance_id, id(self._receiver), adapter_start_time
        )

    # ---- 消息发送 ----

    async def send_message(
        self,
        chat_id: str,
        content: str,
        msg_type: str = "text",
        **kwargs,
    ) -> str:
        """发送消息。"""
        receive_id_type = kwargs.get("receive_id_type", "chat_id")

        result = await self._platform.send_message(
            receive_id=chat_id,
            content=content,
            msg_type=msg_type,
            receive_id_type=receive_id_type,
        )

        # IMMessage 对象有 id 属性
        return result.id if hasattr(result, "id") else ""

    async def reply_message(
        self,
        message_id: str,
        content: str,
        chat_id: str = "",
        **kwargs,
    ) -> str:
        """回复消息。"""
        result = await self._platform.reply_message(
            message_id=message_id,
            content=content,
            msg_type=kwargs.get("msg_type", "text"),
        )

        # IMMessage 对象有 id 属性
        return result.id if hasattr(result, "id") else ""

    async def edit_message(
        self,
        message_id: str,
        content: str,
        chat_id: str = "",
    ) -> bool:
        """编辑消息。"""
        try:
            await self._platform.update_message(
                message_id=message_id,
                content=content,
            )
            return True
        except Exception as e:
            logger.error("Edit message error: %s", e)
            return False

    # ---- Webhook 支持 ----

    async def parse_webhook(self, payload: dict) -> list[GatewayMessage]:
        """解析飞书 Webhook 事件。"""
        messages = []

        # 处理 URL 验证
        if payload.get("type") == "url_verification":
            return []

        # 处理消息事件
        event = payload.get("event", {})
        if event:
            msg_dict = self._parse_event_to_dict(event)
            if msg_dict:
                messages.append(self._convert_to_gateway_message(msg_dict))

        return messages

    async def verify_webhook(
        self,
        signature: str,
        body: bytes,
        headers: dict | None = None,
    ) -> bool:
        """验证飞书 Webhook 签名。"""
        if not self.encrypt_key:
            return True

        try:
            timestamp = headers.get("X-Lark-Request-Timestamp", "") if headers else ""
            nonce = headers.get("X-Lark-Request-Nonce", "") if headers else ""

            if not timestamp or not nonce:
                return True

            import hashlib
            import time

            # 检查时间戳
            current_time = int(time.time())
            if abs(current_time - int(timestamp)) > 300:
                return False

            # 计算签名
            sign_base = timestamp + nonce + self.encrypt_key + body.decode("utf-8")
            computed = hashlib.sha256(sign_base.encode()).hexdigest()

            expected = headers.get("X-Lark-Signature", "") if headers else ""
            if expected.startswith("sha256="):
                expected = expected[7:]

            import hmac

            return hmac.compare_digest(computed, expected)

        except Exception as e:
            logger.error("Webhook verification error: %s", e)
            return False

    # ---- 消息转换 ----

    def _convert_to_gateway_message(self, msg_dict: dict) -> GatewayMessage:
        """将 im_platform 消息格式转换为 GatewayMessage。"""
        return self._create_gateway_message(
            chat_id=msg_dict.get("chat_id", ""),
            sender_id=msg_dict.get("sender_id", ""),
            content=msg_dict.get("content", ""),
            msg_type=msg_dict.get("msg_type", "text"),
            reply_to=msg_dict.get("parent_id", ""),
            thread_id=msg_dict.get("root_id", ""),
            metadata={
                "message_id": msg_dict.get("id", ""),
                "create_time": msg_dict.get("create_time", ""),
                "raw": msg_dict.get("raw", {}),
            },
        )

    def _parse_event_to_dict(self, event: dict) -> dict | None:
        """解析飞书事件为消息字典。"""
        message = event.get("message", {})
        if not message:
            return None

        sender = event.get("sender", {})

        return {
            "id": message.get("message_id", ""),
            "chat_id": message.get("chat_id", ""),
            "sender_id": sender.get("sender_id", {}).get("open_id", ""),
            "content": message.get("content", ""),
            "msg_type": message.get("message_type", "text"),
            "create_time": str(message.get("create_time", "")),
            "parent_id": message.get("parent_id", ""),
            "root_id": message.get("root_id", ""),
            "raw": event,
        }
