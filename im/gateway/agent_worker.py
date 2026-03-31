"""Gateway 模式 Agent Worker

通过 WebSocket 连接到 Gateway，接收来自 IM 渠道的消息，
使用本地 MementoSAgent 处理，并将响应发送回 Gateway。

架构:
  IM Channel → Gateway → Agent Worker → MementoSAgent
                                ↓
            IM Channel ← Gateway ← Agent Worker

与 middleware/im/gateway/agent_worker.py 的区别:
  - middleware/im/gateway/agent_worker.py: 底层实现（保留）
  - im/gateway/agent_worker.py: 顶层调用入口（本文件）
"""

from __future__ import annotations

import asyncio
from typing import Any

import websockets

from im.bridge_base import IMBridgeBase
from middleware.config import g_config
from middleware.im.gateway.protocol import GatewayMessage, MessageType
from core.protocol import AGUIEventType
from utils.logger import get_logger

logger = get_logger(__name__)


class GatewayAgentWorker(IMBridgeBase):
    """Gateway 模式 Agent Worker

    连接到 Gateway 服务，接收多渠道消息并处理。
    继承自 IMBridgeBase，复用 Agent 调用逻辑。
    """

    def __init__(
        self,
        gateway_url: str = "ws://127.0.0.1:8765",
        agent_id: str = "agent_main",
    ):
        super().__init__(platform="gateway")
        self.gateway_url = gateway_url
        self.agent_id = agent_id
        self._ws: Any = None

    async def start(self) -> None:
        """启动 Agent Worker"""
        if self._running:
            return

        self._logger.info(
            f"[GatewayWorker] Starting, connecting to {self.gateway_url}..."
        )

        # 1. 初始化 Agent（复用基类）
        await self.initialize()

        # 2. 连接 Gateway
        await self._connect_gateway()

        self._running = True
        self._logger.info("[GatewayWorker] Started successfully")

    async def _connect_gateway(self) -> None:
        """连接到 Gateway 并注册"""
        try:
            self._ws = await websockets.connect(self.gateway_url)

            # 发送 CONNECT 消息注册为 Agent
            connect_msg = GatewayMessage(
                type=MessageType.CONNECT,
                source=self.agent_id,
                source_type="agent",
                metadata={
                    "capabilities": ["chat", "skill_execution"],
                },
            )
            await self._ws.send(connect_msg.to_json())

            # 等待确认
            response = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
            ack_msg = GatewayMessage.from_json(response)

            if ack_msg.type != MessageType.CONNECT_ACK:
                raise RuntimeError(
                    f"Gateway did not acknowledge connection: {ack_msg.type}"
                )

            self._logger.info(
                f"[GatewayWorker] Connected to Gateway as {self.agent_id}"
            )

            # 启动消息处理循环
            asyncio.create_task(self._message_loop())

        except Exception as e:
            self._logger.error(f"[GatewayWorker] Failed to connect to Gateway: {e}")
            raise

    async def _message_loop(self) -> None:
        """消息处理循环"""
        try:
            async for data in self._ws:
                try:
                    msg = GatewayMessage.from_json(data)

                    if msg.type == MessageType.CHANNEL_MESSAGE:
                        # 处理来自 IM 渠道的消息
                        asyncio.create_task(self._handle_channel_message(msg))

                    elif msg.type == MessageType.PING:
                        # 响应心跳
                        pong = GatewayMessage(
                            type=MessageType.PONG,
                            source=self.agent_id,
                            correlation_id=msg.id,
                        )
                        await self._ws.send(pong.to_json())

                except Exception as e:
                    self._logger.error(f"[GatewayWorker] Error processing message: {e}")

        except websockets.ConnectionClosed:
            self._logger.warning("[GatewayWorker] Gateway connection closed")
            self._running = False
        except Exception as e:
            self._logger.error(f"[GatewayWorker] Message loop error: {e}")
            self._running = False

    async def _handle_channel_message(self, msg: GatewayMessage) -> None:
        """处理来自 IM 渠道的消息"""
        try:
            # 从消息中获取发送者信息
            sender_id = msg.sender_id or msg.metadata.get("sender_id", "")
            content = msg.content or ""
            channel = (
                msg.channel_type.value
                if msg.channel_type
                else msg.metadata.get("channel", "")
            )
            account_id = msg.channel_account or msg.metadata.get("account_id", "")
            original_connection_id = msg.metadata.get("original_connection_id", "")

            self._logger.info(
                f"[GatewayWorker] Received message from {channel}/{sender_id}: "
                f"{content[:50]}..."
            )

            if not sender_id or not content:
                self._logger.warning(
                    f"[GatewayWorker] Empty sender_id or content, skipping"
                )
                return

            # 检查渠道是否已启用
            if not self._is_channel_enabled(channel):
                self._logger.warning(
                    f"[GatewayWorker] Channel {channel} is disabled, skipping"
                )
                return

            # 事件回调
            def on_event(event: dict) -> None:
                event_type = event.get("type")
                if event_type == AGUIEventType.TOOL_CALL_START:
                    self._logger.info(
                        f"[GatewayWorker] Tool call: {event.get('toolName')}"
                    )

            # 调用 Agent 处理（复用基类方法）
            final_text = await self.process_with_agent(
                sender_id=sender_id,
                content=content,
                on_event=on_event,
                metadata={
                    "channel": channel,
                    "account_id": account_id,
                },
            )

            # 发送响应回 Gateway
            if final_text:
                response_msg = GatewayMessage(
                    type=MessageType.AGENT_RESPONSE,
                    source=self.agent_id,
                    target=msg.source,
                    content=final_text,
                    session_id=self.get_session_id(sender_id) or "",
                    channel_type=msg.channel_type,
                    channel_account=msg.channel_account,
                    chat_id=msg.chat_id,
                    correlation_id=msg.id,
                    metadata={
                        "sender_id": sender_id,
                        "channel": channel,
                        "account_id": account_id,
                        "original_connection_id": original_connection_id,
                        "chat_type": msg.metadata.get("chat_type", ""),
                    },
                )
                await self._ws.send(response_msg.to_json())
                self._logger.info(f"[GatewayWorker] Sent response to {sender_id}")

        except Exception as e:
            self._logger.error(f"[GatewayWorker] Error handling channel message: {e}")

    def _is_channel_enabled(self, channel: str) -> bool:
        """检查渠道是否已启用"""
        try:
            # 强制重新加载配置，避免缓存不一致
            config = g_config.load()

            # 检查 Gateway 总开关
            gateway_enabled = (
                getattr(config.gateway, "enabled", False)
                if hasattr(config, "gateway")
                else False
            )
            self._logger.debug(f"[GatewayWorker] Channel {channel} check: gateway_enabled={gateway_enabled}")
            if not gateway_enabled:
                return False

            im_config = getattr(config, "im", None)
            if not im_config:
                return True

            # 标准化渠道名称
            channel_lower = channel.lower()
            if channel_lower in ("feishu", "lark"):
                platform_cfg = getattr(im_config, "feishu", None)
            elif channel_lower in ("dingtalk", "dingding"):
                platform_cfg = getattr(im_config, "dingtalk", None)
            elif channel_lower in ("wecom", "wechatwork", "企业微信"):
                platform_cfg = getattr(im_config, "wecom", None)
            elif channel_lower == "wechat":
                platform_cfg = getattr(im_config, "wechat", None)
            else:
                return True

            if platform_cfg is None:
                return True

            return getattr(platform_cfg, "enabled", True)

        except Exception as e:
            self._logger.error(f"[GatewayWorker] Error checking channel status: {e}")
            return True

    async def on_message(self, sender_id: str, content: str, **kwargs) -> None:
        """处理消息（由 _handle_channel_message 调用）"""
        # 此方法由 _handle_channel_message 处理，不需要单独实现
        pass

    async def send_reply(self, sender_id: str, content: str, **kwargs) -> None:
        """发送回复（由 _handle_channel_message 处理）"""
        # 此方法由 _handle_channel_message 处理，不需要单独实现
        pass

    async def stop(self) -> None:
        """停止 Agent Worker"""
        if not self._running:
            return

        self._logger.info("[GatewayWorker] Stopping...")
        self._running = False

        if self._ws:
            await self._ws.close()
            self._ws = None

        self._logger.info("[GatewayWorker] Stopped")


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

_gateway_worker: GatewayAgentWorker | None = None


def get_gateway_worker() -> GatewayAgentWorker | None:
    """获取全局 Gateway Worker 实例"""
    return _gateway_worker


async def start_gateway_worker(
    gateway_url: str = "ws://127.0.0.1:8765",
    agent_id: str = "agent_main",
) -> GatewayAgentWorker:
    """启动全局 Gateway Worker"""
    global _gateway_worker
    if _gateway_worker is None:
        _gateway_worker = GatewayAgentWorker(
            gateway_url=gateway_url,
            agent_id=agent_id,
        )
        await _gateway_worker.start()
    return _gateway_worker


async def stop_gateway_worker() -> None:
    """停止全局 Gateway Worker"""
    global _gateway_worker
    if _gateway_worker:
        await _gateway_worker.stop()
        _gateway_worker = None
