"""
渠道适配器基类。

提供默认实现，子类只需重写必要方法。
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Callable

from ..protocol import (
    ChannelCapability,
    ChannelType,
    ConnectionConfig,
    ConnectionMode,
    GatewayMessage,
    MessageType,
    PROTOCOL_VERSION,
)

logger = logging.getLogger(__name__)


class BaseChannelAdapter:
    """
    渠道适配器基类。

    提供默认实现，子类只需重写必要方法。
    支持多种连接模式：polling / webhook / websocket / hybrid。
    """

    # 子类必须定义
    channel_type: ChannelType = None
    capabilities: list[str | ChannelCapability] = []

    # 支持的连接模式
    supported_modes: list[ConnectionMode] = [
        ConnectionMode.POLLING,
        ConnectionMode.WEBHOOK,
    ]

    # 协议版本
    protocol_version: int = PROTOCOL_VERSION

    def __init__(self, **kwargs):
        self._mode: ConnectionMode = ConnectionMode.POLLING
        self._config: ConnectionConfig | None = None
        self._message_callbacks: list[Callable[[GatewayMessage], None]] = []
        self._event_callbacks: list[Callable[[dict], None]] = []
        self._running = False
        self._event_loop: asyncio.AbstractEventLoop | None = None  # 保存事件循环引用

    # ---- 属性 ----

    @property
    def mode(self) -> ConnectionMode:
        """当前连接模式。"""
        return self._mode

    @property
    def is_running(self) -> bool:
        """是否正在运行。"""
        return self._running

    # ---- 生命周期 ----

    async def initialize(
        self,
        config: ConnectionConfig,
        mode: ConnectionMode = ConnectionMode.POLLING,
    ) -> None:
        """
        初始化适配器。

        Args:
            config: 连接配置
            mode: 连接模式
        """
        logger.info(
            f"[BaseChannelAdapter] initialize called for {self.channel_type}, mode={mode}"
        )
        try:
            self._config = config
            self._mode = mode
            self._running = False

            # 保存当前事件循环引用，用于后台线程回调
            try:
                self._event_loop = asyncio.get_running_loop()
                logger.debug(
                    f"[BaseChannelAdapter] Saved event loop: {self._event_loop}"
                )
            except RuntimeError:
                self._event_loop = None
                logger.warning("[BaseChannelAdapter] No event loop in initialize")

            # 子类可以重写 _do_initialize 进行额外初始化
            await self._do_initialize(config, mode)

            logger.info(
                "Channel adapter initialized: %s (mode=%s)",
                self.channel_type.value if self.channel_type else "unknown",
                mode.value,
            )
        except Exception as e:
            logger.error(f"[BaseChannelAdapter] initialize failed: {e}", exc_info=True)
            raise

    async def _do_initialize(
        self,
        config: ConnectionConfig,
        mode: ConnectionMode,
    ) -> None:
        """子类实现的初始化逻辑。"""
        pass

    async def start(self) -> None:
        """
        启动渠道服务。

        根据连接模式执行不同操作：
        - POLLING: 启动轮询循环
        - WEBHOOK: 无需启动连接
        - WEBSOCKET: 建立 WebSocket 连接
        - HYBRID: 同时启动 polling 和 webhook 处理
        """
        if self._running:
            return

        await self._do_start()
        self._running = True

        logger.info(
            "Channel adapter started: %s (mode=%s)",
            self.channel_type.value if self.channel_type else "unknown",
            self._mode.value,
        )

    async def _do_start(self) -> None:
        """子类实现的启动逻辑。"""
        pass

    async def stop(self) -> None:
        """停止渠道服务。"""
        if not self._running:
            return

        await self._do_stop()
        self._running = False

        logger.info(
            "Channel adapter stopped: %s",
            self.channel_type.value if self.channel_type else "unknown",
        )

    async def _do_stop(self) -> None:
        """子类实现的停止逻辑。"""
        pass

    async def health_check(self) -> bool:
        """健康检查。"""
        return self._running

    # ---- 消息发送 ----

    async def send_message(
        self,
        chat_id: str,
        content: str,
        msg_type: str = "text",
        **kwargs,
    ) -> str:
        """
        发送消息到渠道。

        Args:
            chat_id: 会话/群组ID
            content: 消息内容
            msg_type: 消息类型

        Returns:
            message_id: 发送的消息ID
        """
        raise NotImplementedError("Subclass must implement send_message()")

    async def reply_message(
        self,
        message_id: str,
        content: str,
        chat_id: str = "",
        **kwargs,
    ) -> str:
        """
        回复消息。

        默认实现调用 send_message。
        """
        return await self.send_message(chat_id, content, **kwargs)

    async def edit_message(
        self,
        message_id: str,
        content: str,
        chat_id: str = "",
    ) -> bool:
        """
        编辑消息。

        不支持编辑的渠道可以不重写此方法。
        """
        raise NotImplementedError(
            f"Message editing not supported on {self.channel_type}"
        )

    async def delete_message(
        self,
        message_id: str,
        chat_id: str = "",
    ) -> bool:
        """
        删除消息。

        不支持删除的渠道可以不重写此方法。
        """
        raise NotImplementedError(
            f"Message deletion not supported on {self.channel_type}"
        )

    # ---- Webhook 支持 ----

    async def parse_webhook(self, payload: dict) -> list[GatewayMessage]:
        """
        解析 Webhook 回调。

        将平台回调转换为 GatewayMessage 列表。
        Webhook 模式必须实现此方法。

        Args:
            payload: Webhook 请求体

        Returns:
            消息列表
        """
        # 默认实现：返回空列表
        logger.debug(
            "parse_webhook not implemented for %s, returning empty list",
            self.channel_type.value if self.channel_type else "unknown",
        )
        return []

    async def verify_webhook(
        self,
        signature: str,
        body: bytes,
        headers: dict | None = None,
    ) -> bool:
        """
        验证 Webhook 签名。

        Webhook 模式建议实现此方法以确保安全。

        Args:
            signature: 签名字符串
            body: 请求体
            headers: 请求头

        Returns:
            是否验证通过
        """
        # 默认实现：不验证
        logger.warning(
            "verify_webhook not implemented for %s, skipping verification",
            self.channel_type.value if self.channel_type else "unknown",
        )
        return True

    # ---- 事件回调 ----

    def on_message(self, callback: Callable[[GatewayMessage], None]) -> None:
        """注册消息回调。"""
        self._message_callbacks.append(callback)

    def on_event(self, callback: Callable[[dict], None]) -> None:
        """注册事件回调。"""
        self._event_callbacks.append(callback)

    def _emit_message(self, message: GatewayMessage) -> None:
        """触发消息回调。

        注意：此方法可能从后台线程调用（FeishuReceiver），必须安全处理。
        """
        logger.debug(
            f"[_emit_message] Emitting message, callbacks={len(self._message_callbacks)}, "
            f"event_loop={self._event_loop}, thread={threading.current_thread().name}"
        )
        for callback in self._message_callbacks:
            try:
                result = callback(message)
                if asyncio.iscoroutine(result):
                    # 回调返回协程，需要在事件循环中调度
                    # 优先使用保存的事件循环（主线程的），其次尝试获取当前线程的
                    loop = self._event_loop
                    if loop is None or loop.is_closed():
                        try:
                            loop = asyncio.get_running_loop()
                        except RuntimeError:
                            loop = None

                    if loop and not loop.is_closed():
                        logger.debug(
                            f"[_emit_message] Scheduling coroutine in loop: {loop}"
                        )
                        # 使用 run_coroutine_threadsafe 将协程调度到目标事件循环
                        asyncio.run_coroutine_threadsafe(result, loop)
                    else:
                        logger.error(
                            f"[_emit_message] No valid event loop available, dropping message"
                        )
            except RuntimeError as e:
                # 可能没有运行中的事件循环（如从非 async 上下文调用）
                logger.warning(f"[_emit_message] RuntimeError: {e}")
            except Exception as e:
                logger.error("[_emit_message] Message callback error: %s", e)

    def _emit_event(self, event: dict) -> None:
        """触发事件回调。"""
        for callback in self._event_callbacks:
            try:
                result = callback(event)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as e:
                logger.error("Event callback error: %s", e)

    # ---- 工具方法 ----

    def _create_gateway_message(
        self,
        chat_id: str,
        sender_id: str,
        content: str,
        msg_type: str = "text",
        metadata: dict | None = None,
        reply_to: str = "",
        thread_id: str = "",
        **kwargs,
    ) -> GatewayMessage:
        """创建 GatewayMessage 实例。"""
        return GatewayMessage(
            type=MessageType.CHANNEL_MESSAGE,
            channel_type=self.channel_type,
            channel_account=self._config.channel_account if self._config else "",
            chat_id=chat_id,
            sender_id=sender_id,
            content=content,
            msg_type=msg_type,
            reply_to=reply_to,
            thread_id=thread_id,
            metadata={**(metadata or {}), **kwargs},
        )
