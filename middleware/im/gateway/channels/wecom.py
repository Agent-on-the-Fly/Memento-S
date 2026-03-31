"""
企业微信渠道适配器。

复用现有的 im_platform/wecom 长连接实现：
- platform.py: API 主动调用
- receiver.py: WebSocket 智能机器人接收

支持连接模式：
- websocket: 使用 aiohttp WebSocket 长连接（默认）
- webhook: 使用 Webhook 回调
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable
from middleware.im.im_platform.wecom.platform import WecomPlatform
from middleware.im.im_platform.wecom.receiver import WecomReceiver
from ..gateway import register_channel
from ..protocol import (
    ChannelCapability,
    ChannelType,
    ConnectionConfig,
    ConnectionMode,
    GatewayMessage,
)
from .base import BaseChannelAdapter

logger = logging.getLogger(__name__)


@register_channel(ChannelType.WECOM)
class WecomAdapter(BaseChannelAdapter):
    """
    企业微信渠道适配器。

    复用 im_platform/wecom 的实现：
    - Platform: API 主动调用
    - Receiver: WebSocket 智能机器人长连接
    """

    channel_type = ChannelType.WECOM
    capabilities = [
        ChannelCapability.TEXT,
        ChannelCapability.RICH_TEXT,
        ChannelCapability.IMAGE,
        ChannelCapability.VIDEO,
        ChannelCapability.AUDIO,
        ChannelCapability.FILE,
        ChannelCapability.INTERACTIVE,
        ChannelCapability.REPLY,
        ChannelCapability.LOCATION,
    ]

    supported_modes = [
        ConnectionMode.WEBSOCKET,  # 默认使用 WebSocket 长连接
        ConnectionMode.WEBHOOK,
        ConnectionMode.HYBRID,
    ]

    def __init__(self, **kwargs):
        """初始化企业微信适配器。

        注意：凭证参数（corp_id, agent_id, secret 等）在 kwargs 中传入，
        但本适配器从配置文件读取凭证，因此忽略 kwargs 中的凭证参数。
        """
        super().__init__()

        # im_platform 组件
        self._platform: Any = None  # WecomPlatform（企业自建应用模式）
        self._receiver: Any = None  # WecomReceiver（智能机器人模式）

        # 智能机器人模式凭证（bot_id + secret）
        self._bot_id: str = ""
        self._bot_secret: str = ""

        # 缓存 reply_func：key = sender_id（p2p）或 chat_id（群聊）
        # 企业微信单聊必须用 aibot_respond_msg（绑定原消息 req_id）才能回到对话上下文
        self._reply_funcs: dict[str, Any] = {}

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
        """初始化企业微信适配器。

        自动检测接入模式：
        - 智能机器人模式：配置了 bot_id + secret，使用 WecomReceiver WebSocket 长连接
        - 企业自建应用模式：配置了 corp_id + secret + agent_id，使用 WecomPlatform REST API
        """
        import json
        import os
        from pathlib import Path

        cfg_path = Path.home() / "memento_s" / "config.json"
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f).get("im", {}).get("wecom", {})
        except Exception:
            cfg = {}

        bot_id = cfg.get("bot_id") or os.environ.get("WECOM_BOT_ID", "")
        corp_id = cfg.get("corp_id") or os.environ.get("WECOM_CORP_ID", "")

        if bot_id:
            # 智能机器人模式：仅需 bot_id + secret，无需 WecomPlatform
            self._bot_id = bot_id
            self._bot_secret = cfg.get("secret") or os.environ.get("WECOM_SECRET", "")
            logger.info("Wecom adapter initialized in smart robot mode (bot_id=%s)", bot_id)
        elif corp_id:
            # 企业自建应用模式
            self._platform = WecomPlatform()
            logger.info("Wecom adapter initialized in enterprise app mode (corp_id=%s)", corp_id)
        else:
            raise ValueError(
                f"企业微信配置缺失：请在 {cfg_path} 的 im.wecom 节填写 "
                "bot_id + secret（智能机器人模式）或 "
                "corp_id + secret + agent_id（企业自建应用模式）"
            )

        logger.info("Wecom adapter initialized: mode=%s", mode.value)

    async def _do_start(self) -> None:
        """启动长连接。"""
        import time
        self._stopping = False
        self._start_time = time.time()  # 记录启动时间
        logger.info("[WecomAdapter] _do_start called, mode=%s", self._mode)

        if self._mode in (ConnectionMode.WEBSOCKET, ConnectionMode.HYBRID):
            await self._start_websocket_receiver()

        logger.info("[WecomAdapter] _do_start completed")

    async def _do_stop(self) -> None:
        """停止长连接。"""
        self._stopping = True
        logger.info("[WecomAdapter] _do_stop called, stopping receiver")

        if self._receiver:
            await self._receiver.stop()
        self._receiver = None

        logger.info("[WecomAdapter] _do_stop completed, _stopping=%s", self._stopping)

    async def health_check(self) -> bool:
        """健康检查。"""
        if self._receiver:
            # 智能机器人模式：检查 WebSocket 连接是否活跃
            return self._receiver._ws is not None and not self._receiver._ws.closed
        if self._platform:
            try:
                await self._platform._get_token()
                return True
            except Exception:
                return False
        return False

    # ---- WebSocket 长连接 ----

    async def _start_websocket_receiver(self) -> None:
        """启动 WebSocket 智能机器人接收器。"""
        import time

        if not self._bot_id or not self._bot_secret:
            logger.warning(
                "Wecom WebSocket requires bot_id and bot_secret, "
                "falling back to webhook mode"
            )
            return

        # 创建消息回调
        adapter_self = self  # 闭包捕获

        async def on_message(msg_dict: dict):
            """处理接收到的消息。"""
            # 检查是否正在停止
            if adapter_self._stopping:
                logger.debug("[WecomAdapter] Message ignored: adapter is stopping")
                return

            # 检查消息是否是积压的旧消息（在启动前收到的）
            msg_time = msg_dict.get("create_time", 0)
            if isinstance(msg_time, str):
                try:
                    msg_time = int(msg_time)
                except ValueError:
                    msg_time = 0

            if msg_time > 10000000000:  # 毫秒时间戳
                msg_time = msg_time / 1000

            if msg_time > 0 and adapter_self._start_time > 0:
                if msg_time < adapter_self._start_time - 5:  # 允许5秒时钟偏差
                    logger.debug(
                        "[WecomAdapter] Message ignored: too old (msg_time=%.2f, start_time=%.2f)",
                        msg_time, adapter_self._start_time
                    )
                    return

            # 缓存 reply_func：不能放进 metadata（JSON 不可序列化），
            # 直接存在 adapter 层，key 为 sender_id（单聊）或 chat_id（群聊）
            reply_func = msg_dict.get("reply")
            if reply_func:
                chat_type = msg_dict.get("chat_type", "")
                cache_key = (
                    msg_dict.get("sender_id", "")
                    if chat_type == "p2p"
                    else msg_dict.get("chat_id", "")
                )
                if cache_key:
                    adapter_self._reply_funcs[cache_key] = reply_func

            gateway_msg = adapter_self._convert_to_gateway_message(msg_dict)
            adapter_self._emit_message(gateway_msg)

        # 创建 Receiver
        self._receiver = WecomReceiver(
            on_message=on_message,
            bot_id=self._bot_id,
            secret=self._bot_secret,
        )

        # 启动接收器（异步运行）
        asyncio.create_task(self._receiver._run_forever())

        logger.info("Wecom WebSocket receiver started")

    # ---- 消息发送 ----

    async def send_message(
        self,
        chat_id: str,
        content: str,
        msg_type: str = "text",
        **kwargs,
    ) -> str:
        """发送消息。

        智能机器人模式：通过 WecomReceiver WebSocket 发送（aibot_send_msg）。
        企业自建应用模式：通过 WecomPlatform REST API 发送。
        """
        if self._receiver:
            # 智能机器人模式
            sender_id = kwargs.get("sender_id", "")
            chat_type = kwargs.get("chat_type", "")

            # 单聊用 sender_id 作 key，群聊用 chat_id 作 key
            cache_key = sender_id if (chat_type == "p2p" and sender_id) else chat_id
            reply_func = self._reply_funcs.pop(cache_key, None)

            if reply_func:
                # 优先走 aibot_respond_msg（绑定原消息 req_id），单聊必须走这条路
                await reply_func(content)
            elif chat_type == "p2p" and sender_id:
                await self._receiver.send_text(chat_id="", text=content, to_user_id=sender_id)
            else:
                await self._receiver.send_text(chat_id=chat_id, text=content)
            return ""

        # 企业自建应用模式
        result = await self._platform.send_message(
            receive_id=chat_id,
            content=content,
            msg_type=msg_type,
            receive_id_type=kwargs.get("receive_id_type", "touser"),
        )
        return result.id if hasattr(result, "id") else ""

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
            receive_id_type="chat",
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
        # 如果有原消息的 reply 函数（流式回复场景）
        if "reply_func" in kwargs:
            reply_func = kwargs["reply_func"]
            await reply_func(content)
            return message_id

        sender_id = kwargs.get("sender_id", chat_id)
        return await self.send_message(
            sender_id,
            content,
            kwargs.get("msg_type", "text"),
            **kwargs,
        )

    # ---- Webhook 支持 ----

    async def parse_webhook(self, payload: dict) -> list[GatewayMessage]:
        """解析企业微信回调消息。"""
        messages = []

        import xml.etree.ElementTree as ET

        # 企业微信回调可能是 XML 或 JSON 格式
        if isinstance(payload, str):
            try:
                root = ET.fromstring(payload)
                payload = self._xml_to_dict(root)
            except ET.ParseError:
                payload = json.loads(payload)

        msg_type = payload.get("MsgType", payload.get("msgtype", ""))

        if msg_type == "event":
            event_msg = self._parse_event(payload)
            if event_msg:
                messages.append(self._convert_to_gateway_message(event_msg))
        elif msg_type:
            messages.append(
                self._convert_to_gateway_message(
                    self._parse_message(payload)
                )
            )

        return messages

    async def verify_webhook(
        self,
        signature: str,
        body: bytes,
        headers: dict | None = None,
    ) -> bool:
        """验证企业微信回调签名。"""
        if not self.token:
            return True

        try:
            import hashlib
            import hmac

            timestamp = headers.get("timestamp", "") if headers else ""
            nonce = headers.get("nonce", "") if headers else ""
            msg_signature = headers.get("msg_signature", signature) if headers else signature

            if not timestamp or not nonce:
                return True

            # 计算签名
            sign_list = [self.token, timestamp, nonce]
            sign_list.sort()
            sign_str = "".join(sign_list)
            computed = hashlib.sha1(sign_str.encode()).hexdigest()

            return hmac.compare_digest(computed, msg_signature)

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
                "agent_id": self._bot_id,
                "create_time": msg_dict.get("create_time", ""),
                "event_type": msg_dict.get("event_type", ""),
                "chat_type": msg_dict.get("chat_type", ""),
                "raw": msg_dict.get("raw", {}),
            },
        )

    def _parse_message(self, payload: dict) -> dict:
        """解析普通消息。"""
        msg_type = payload.get("MsgType", payload.get("msgtype", "text"))

        # 提取内容
        if msg_type == "text":
            content = payload.get("Content", payload.get("text", {}).get("content", ""))
        elif msg_type == "image":
            content = payload.get("PicUrl", "")
        elif msg_type == "video":
            content = payload.get("ThumbMediaId", "")
        elif msg_type == "voice":
            content = payload.get("MediaId", "")
        elif msg_type == "location":
            label = payload.get("Label", "")
            content = f"位置: {label}"
        elif msg_type == "markdown":
            content = payload.get("markdown", {}).get("content", "")
        else:
            content = json.dumps(payload.get(msg_type, {}))

        from_user = payload.get("FromUserName", payload.get("userid", ""))
        chat_id = payload.get("ChatId", from_user)

        return {
            "id": payload.get("MsgId", ""),
            "chat_id": chat_id,
            "sender_id": from_user,
            "content": content,
            "msg_type": msg_type,
            "create_time": payload.get("CreateTime", ""),
            "raw": payload,
        }

    def _parse_event(self, payload: dict) -> dict | None:
        """解析事件消息。"""
        event = payload.get("Event", "")
        event_key = payload.get("EventKey", "")
        from_user = payload.get("FromUserName", payload.get("UserID", ""))

        event_contents = {
            "subscribe": f"用户 {from_user} 关注了应用",
            "unsubscribe": f"用户 {from_user} 取消关注了应用",
            "enter_chat": f"用户 {from_user} 进入应用",
            "click": f"用户 {from_user} 点击了菜单: {event_key}",
            "view": f"用户 {from_user} 跳转到: {event_key}",
        }

        return {
            "id": "",
            "chat_id": from_user,
            "sender_id": from_user,
            "content": event_contents.get(event, f"事件: {event}"),
            "msg_type": "event",
            "create_time": payload.get("CreateTime", ""),
            "event_type": event,
            "raw": payload,
        }

    # ---- 工具方法 ----

    def _xml_to_dict(self, root) -> dict:
        """将 XML 转换为字典。"""
        result = {}
        for child in root:
            if len(child) > 0:
                result[child.tag] = self._xml_to_dict(child)
            else:
                result[child.tag] = child.text
        return result
