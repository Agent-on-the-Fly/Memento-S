"""
钉钉渠道适配器。

复用现有的 im_platform/dingtalk 长连接实现：
- platform.py: API 主动调用
- receiver.py: Stream 长连接接收消息

支持连接模式：
- websocket: 使用 dingtalk-stream SDK 的 Stream 模式（默认）
- webhook: 使用 Webhook 回调
"""

from __future__ import annotations

import asyncio
import json
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
from middleware.im.im_platform.dingtalk.platform import DingTalkPlatform
from middleware.im.im_platform.dingtalk.receiver import DingTalkReceiver

logger = logging.getLogger(__name__)


@register_channel(ChannelType.DINGTALK)
class DingTalkAdapter(BaseChannelAdapter):
    """
    钉钉渠道适配器。

    复用 im_platform/dingtalk 的实现：
    - Platform: API 主动调用
    - Receiver: Stream 长连接接收
    """

    channel_type = ChannelType.DINGTALK
    capabilities = [
        ChannelCapability.TEXT,
        ChannelCapability.RICH_TEXT,
        ChannelCapability.IMAGE,
        ChannelCapability.FILE,
        ChannelCapability.INTERACTIVE,
        ChannelCapability.REPLY,
    ]

    supported_modes = [
        ConnectionMode.WEBSOCKET,  # 使用 Stream 模式
        ConnectionMode.WEBHOOK,
        ConnectionMode.HYBRID,
    ]

    def __init__(self, **kwargs):
        """初始化钉钉适配器。

        注意：凭证参数在 kwargs 中传入，但本适配器从配置文件读取凭证，
        因此忽略 kwargs 中的凭证参数。
        """
        super().__init__()

        # im_platform 组件
        self._platform: Any = None  # DingTalkPlatform
        self._receiver: Any = None  # DingTalkReceiver
        self._client: Any = None  # Stream Client

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
        """初始化钉钉适配器。"""
        # 导入 im_platform 组件
        # 创建 Platform 实例（从配置文件读取凭证）
        self._platform = DingTalkPlatform()

        logger.info(
            "DingTalk adapter initialized: mode=%s",
            mode.value,
        )

    async def _do_start(self) -> None:
        """启动长连接。"""
        import time
        self._stopping = False
        self._start_time = time.time()  # 记录启动时间
        logger.info("[DingTalkAdapter] _do_start called, mode=%s", self._mode)

        if self._mode in (ConnectionMode.WEBSOCKET, ConnectionMode.HYBRID):
            await self._start_stream_receiver()

        logger.info("[DingTalkAdapter] _do_start completed")

    async def _do_stop(self) -> None:
        """停止长连接。"""
        self._stopping = True
        logger.info("[DingTalkAdapter] _do_stop called, stopping receiver")

        if self._client:
            try:
                await self._client.stop()
            except Exception:
                pass
        self._receiver = None
        self._client = None

        logger.info("[DingTalkAdapter] _do_stop completed, _stopping=%s", self._stopping)

    async def health_check(self) -> bool:
        """健康检查。"""
        if not self._platform:
            return False
        try:
            await self._platform._get_token()
            return True
        except Exception:
            return False

    # ---- Stream 长连接 ----

    async def _start_stream_receiver(self) -> None:
        """启动 Stream 长连接接收器。"""
        import time

        # 创建消息回调
        adapter_self = self  # 闭包捕获

        def on_message(msg_dict: dict):
            """处理接收到的消息。"""
            # 检查是否正在停止
            if adapter_self._stopping:
                logger.debug("[DingTalkAdapter] Message ignored: adapter is stopping")
                return

            # 检查消息是否是积压的旧消息（在启动前收到的）
            # 钉钉消息有 createTime 字段（毫秒时间戳）
            msg_time = msg_dict.get("create_time", 0)
            if isinstance(msg_time, str):
                try:
                    msg_time = int(msg_time)
                except ValueError:
                    msg_time = 0

            # 将毫秒转换为秒
            if msg_time > 10000000000:  # 毫秒时间戳
                msg_time = msg_time / 1000

            if msg_time > 0 and adapter_self._start_time > 0:
                if msg_time < adapter_self._start_time - 5:  # 允许5秒时钟偏差
                    logger.debug(
                        "[DingTalkAdapter] Message ignored: too old (msg_time=%.2f, start_time=%.2f)",
                        msg_time, adapter_self._start_time
                    )
                    return

            gateway_msg = adapter_self._convert_to_gateway_message(msg_dict)
            adapter_self._emit_message(gateway_msg)

        # 创建 Receiver（从配置文件读取凭证）
        self._receiver = DingTalkReceiver(
            on_message=on_message,
        )

        # 构建并启动 Stream Client
        self._client = self._receiver._build_client()

        # 在后台启动（异步方式）
        asyncio.create_task(self._run_stream_client())

        logger.info("DingTalk Stream receiver started")

    async def _run_stream_client(self) -> None:
        """运行 Stream 客户端。"""
        try:
            await self._client.start()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Stream client error: %s", e)

    # ---- 消息发送 ----

    async def send_message(
        self,
        chat_id: str,
        content: str,
        msg_type: str = "text",
        **kwargs,
    ) -> str:
        """发送消息。

        Args:
            chat_id: 会话 ID (conversation_id)
            content: 消息内容
            msg_type: 消息类型
            **kwargs:
                - sender_id: 发送者 ID (单聊时使用 staffId)
                - chat_type: 会话类型 ("p2p" 或 "group")
        """
        sender_id = kwargs.get("sender_id", "")
        chat_type = kwargs.get("chat_type", "")

        # 判断是单聊还是群聊
        if chat_type == "p2p" and sender_id:
            # 单聊: 使用 sender_staff_id
            result = await self._platform.send_message(
                receive_id=sender_id,
                content=content,
                msg_type=msg_type,
                receive_id_type="staffId",
            )
        else:
            # 群聊: 使用 conversation_id
            result = await self._platform.send_message(
                receive_id=chat_id,
                content=content,
                msg_type=msg_type,
                receive_id_type="openConversationId",
            )

        return result.id if hasattr(result, 'id') else ""

    async def send_to_chat(
        self,
        chat_id: str,
        content: str,
        msg_type: str = "text",
        **kwargs,
    ) -> str:
        """发送消息到群聊。"""
        result = await self._platform.send_message(
            receive_id=chat_id,
            content=content,
            msg_type=msg_type,
            receive_id_type="conversationId",
        )

        return result.get("id", "")

    async def reply_message(
        self,
        message_id: str,
        content: str,
        chat_id: str = "",
        **kwargs,
    ) -> str:
        """回复消息。"""
        sender_id = kwargs.get("sender_id", chat_id)
        return await self.send_message(
            sender_id,
            content,
            kwargs.get("msg_type", "text"),
            **kwargs,
        )

    # ---- Webhook 支持 ----

    async def parse_webhook(self, payload: dict) -> list[GatewayMessage]:
        """解析钉钉 Webhook 事件。"""
        messages = []

        msg_type = payload.get("msgtype", "")

        if msg_type:
            # 企业内部机器人消息
            msg_dict = self._parse_robot_message(payload)
            messages.append(self._convert_to_gateway_message(msg_dict))
        elif "conversationId" in payload:
            # Stream 消息格式
            msg_dict = self._parse_stream_message(payload)
            messages.append(self._convert_to_gateway_message(msg_dict))

        return messages

    async def verify_webhook(
        self,
        signature: str,
        body: bytes,
        headers: dict | None = None,
    ) -> bool:
        """验证钉钉 Webhook 签名。"""
        if not self.token:
            return True

        try:
            import hashlib
            import hmac
            import time

            timestamp = headers.get("timestamp", "") if headers else ""
            nonce = headers.get("nonce", "") if headers else ""

            if not timestamp or not nonce:
                return True

            # 检查时间戳
            current_time = int(time.time() * 1000)
            if abs(current_time - int(timestamp)) > 300000:
                return False

            # 计算签名
            sign_base = timestamp + nonce + self.token + body.decode("utf-8")
            computed = hmac.new(
                self.token.encode(),
                sign_base.encode(),
                hashlib.sha256,
            ).hexdigest()

            return hmac.compare_digest(computed, signature)

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
            metadata={
                "message_id": msg_dict.get("id", ""),
                "create_time": msg_dict.get("create_time", ""),
                "sender_nick": msg_dict.get("sender_nick", ""),
                "chat_type": msg_dict.get("chat_type", ""),
                "at_users": msg_dict.get("at_users", []),
                "raw": msg_dict.get("raw", {}),
            },
        )

    def _parse_robot_message(self, payload: dict) -> dict:
        """解析机器人消息。"""
        msg_type = payload.get("msgtype", "text")

        # 提取内容
        if msg_type == "text":
            content = payload.get("text", {}).get("content", "")
        elif msg_type == "markdown":
            content = payload.get("markdown", {}).get("text", "")
        else:
            content = json.dumps(payload.get(msg_type, {}))

        return {
            "id": payload.get("msgId", ""),
            "chat_id": payload.get("conversationId", ""),
            "sender_id": payload.get("senderStaffId", "") or payload.get("senderId", ""),
            "content": content,
            "msg_type": msg_type,
            "create_time": payload.get("createAt", ""),
            "sender_nick": payload.get("senderNick", ""),
            "chat_type": payload.get("conversationType", ""),
            "at_users": payload.get("atUsers", []),
            "raw": payload,
        }

    def _parse_stream_message(self, payload: dict) -> dict:
        """解析 Stream 消息。"""
        content = payload.get("content", {})
        if isinstance(content, dict):
            text_content = content.get("content", "")
        else:
            text_content = str(content)

        return {
            "id": payload.get("msgId", ""),
            "chat_id": payload.get("conversationId", ""),
            "sender_id": payload.get("senderId", ""),
            "content": text_content,
            "msg_type": payload.get("msgtype", "text"),
            "create_time": payload.get("createTime", ""),
            "raw": payload,
        }
