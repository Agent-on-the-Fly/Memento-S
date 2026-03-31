"""
WeChat (Personal) Integration via OpenClaw Weixin Python SDK

基于 openclaw-weixin-python SDK 的微信个人号适配器
使用微信 ilinkai 官方 API (https://ilinkai.weixin.qq.com)

Architecture:
- Polling mode (matches Gateway architecture)
- Long-polling message receive (35s timeout)
- Automatic session management via context_token
- Supports text/image/video/file/voice messages

Usage:
    1. Install openclaw-weixin-python SDK
    2. Configure token in config.json
    3. Start gateway: memento gateway-worker

Configuration:
    {
        "im": {
            "gateway": {
                "channels": {
                    "wechat": {
                        "enabled": true,
                        "accounts": [
                            {
                                "account_id": "personal",
                                "type": "ilinkai",
                                "base_url": "https://ilinkai.weixin.qq.com",
                                "token": "your-bot-token",
                                "mode": "polling"
                            }
                        ]
                    }
                }
            }
        }
    }
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Instance counter for debugging
_instance_counter = 0

# Add 3rd party SDK to path before importing
_3RD_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "3rd"
if str(_3RD_DIR) not in sys.path:
    sys.path.insert(0, str(_3RD_DIR))

from .base import BaseChannelAdapter
from ..protocol import (
    ChannelCapability,
    ChannelType,
    ConnectionConfig,
    ConnectionMode,
    GatewayMessage,
    MessageType as GatewayMsgType,
)
from ..gateway import register_channel
from utils.logger import get_logger

logger = get_logger(__name__)

# Try to import openclaw-weixin-python SDK
# First try local 3rd party version, then system installed
try:
    from weixin_sdk import (
        WeixinClient,
        WeixinMessage,
        MessageType,
        MessageItemType,
        WeixinSessionExpiredError,
    )

    WEIXIN_SDK_AVAILABLE = True
    logger.info("[WechatIlinkaiAdapter] Using weixin_sdk from local 3rd/ directory")
except ImportError as e:
    WEIXIN_SDK_AVAILABLE = False
    logger.warning(
        f"[WechatIlinkaiAdapter] weixin_sdk not available: {e}. "
        "Make sure 3rd/weixin_sdk exists or install with: "
        "uv pip install -e /path/to/openclaw-weixin-python"
    )


@register_channel(ChannelType.WECHAT)
class WechatIlinkaiAdapter(BaseChannelAdapter):
    """
    WeChat Personal Adapter via OpenClaw Weixin Python SDK.

    Follows the same architecture pattern as Feishu/DingTalk/WeCom adapters:
    - Uses BaseChannelAdapter as base class
    - Implements Polling connection mode
    - Converts WeixinMessage to GatewayMessage
    """

    channel_type = ChannelType.WECHAT
    capabilities = [
        ChannelCapability.TEXT,
        ChannelCapability.IMAGE,
        ChannelCapability.VIDEO,
        ChannelCapability.FILE,
        ChannelCapability.AUDIO,
        ChannelCapability.REPLY,
    ]
    supported_modes = [
        ConnectionMode.POLLING,
    ]

    def __init__(
        self,
        account_id: str = "default",
        base_url: str = "https://ilinkai.weixin.qq.com",
        token: Optional[str] = None,
    ):
        super().__init__()

        # Instance tracking for debugging
        global _instance_counter
        _instance_counter += 1
        self._instance_id = _instance_counter

        self.account_id = account_id
        self.base_url = base_url
        self._token = token

        # SDK client
        self._client: Optional[Any] = None
        # Per-user typing tickets (user_id -> typing_ticket)
        self._user_typing_tickets: Dict[str, str] = {}
        # Per-user active typing tasks (user_id -> (task, stop_event))
        self._active_typing_tasks: Dict[str, tuple] = {}

        # Runtime state
        self._polling_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info(
            f"[WechatIlinkaiAdapter#{self._instance_id}] Instance #{self._instance_id} created for account {account_id}"
        )

    async def _do_initialize(
        self,
        config: ConnectionConfig,
        mode: ConnectionMode,
    ) -> None:
        """Initialize WeChat adapter."""
        if not WEIXIN_SDK_AVAILABLE:
            raise RuntimeError(
                "openclaw-weixin-python SDK not installed. "
                "Install with: uv pip install -e /path/to/openclaw-weixin-python"
            )

        logger.info(
            f"[WechatIlinkaiAdapter#{self._instance_id}] Initializing account: {self.account_id}"
        )

        # Override with config values from metadata (passed via credentials in gateway_starter.py)
        cfg = getattr(config, "metadata", {}) or {}
        self.base_url = cfg.get("base_url", self.base_url)
        self._token = cfg.get("token") or self._token

        if not self._token:
            raise ValueError(
                f"[WechatIlinkaiAdapter#{self._instance_id}] Token required for account {self.account_id}. "
                "Please configure token or run 'memento wechat-login' to obtain one."
            )

        # Initialize SDK client
        logger.info(
            f"[WechatIlinkaiAdapter#{self._instance_id}] Creating WeixinClient with base_url={self.base_url}"
        )
        self._client = WeixinClient(
            base_url=self.base_url,
            token=self._token,
        )
        logger.info(
            f"[WechatIlinkaiAdapter#{self._instance_id}] WeixinClient created successfully: {self._client is not None}"
        )

        logger.info(
            f"[WechatIlinkaiAdapter#{self._instance_id}] Initialized with base_url={self.base_url}"
        )

    async def _do_start(self) -> None:
        """Start polling loop."""
        if not self._client:
            raise RuntimeError(
                "[WechatIlinkaiAdapter#{self._instance_id}] Not initialized"
            )

        self._running = True

        # Start polling task
        self._polling_task = asyncio.create_task(
            self._polling_loop(), name=f"wechat-polling-{self.account_id}"
        )

        logger.info(
            f"[WechatIlinkaiAdapter#{self._instance_id}] Polling started for account {self.account_id}"
        )

    async def _do_stop(self) -> None:
        """Stop polling."""
        logger.info(
            f"[WechatIlinkaiAdapter#{self._instance_id}] Stopping account {self.account_id}"
        )

        self._running = False

        # 先关闭客户端连接，强制中断正在进行的 HTTP 长轮询请求
        # 这样 poll_messages() 会立即抛出异常，polling task 才能快速退出
        if self._client:
            logger.info(
                f"[WechatIlinkaiAdapter#{self._instance_id}] Closing client to interrupt polling"
            )
            await self._client.close()
            self._client = None
            logger.info(
                f"[WechatIlinkaiAdapter#{self._instance_id}] Client closed, polling interrupted"
            )

        # 现在取消 polling task（它应该已经因为 client 关闭而退出）
        if self._polling_task and not self._polling_task.done():
            self._polling_task.cancel()
            try:
                await asyncio.wait_for(self._polling_task, timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning(
                    f"[WechatIlinkaiAdapter#{self._instance_id}] Polling task did not stop in time"
                )
            except asyncio.CancelledError:
                pass

        # 清理打字指示器任务
        for user_id, (task, stop_event) in list(self._active_typing_tasks.items()):
            stop_event.set()
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
        self._active_typing_tasks.clear()
        self._user_typing_tickets.clear()

        logger.info(
            f"[WechatIlinkaiAdapter#{self._instance_id}] Stopped account {self.account_id}"
        )

    async def _polling_loop(self) -> None:
        """
        Main polling loop using SDK's poll_messages() generator.

        This mirrors the architecture of openclaw-weixin TypeScript version:
        - Uses long-polling (35s timeout)
        - Automatically handles cursor persistence
        - Handles session expiration gracefully
        """
        logger.info(
            f"[WechatIlinkaiAdapter#{self._instance_id}] _polling_loop started, client={self._client is not None}"
        )
        if not self._client:
            logger.error(
                "[WechatIlinkaiAdapter#{self._instance_id}] _client is None, cannot start polling"
            )
            return

        retry_delay = 5.0

        while self._running:
            try:
                # poll_messages() is an async generator that yields WeixinMessage
                async for msg in self._client.poll_messages():
                    if not self._running:
                        break
                    if not self._running:
                        break

                    # Get typing ticket and start typing indicator immediately
                    # This provides instant feedback to user that bot is processing
                    if msg.from_user_id and self._client:
                        # First ensure we have the typing ticket
                        if msg.from_user_id not in self._user_typing_tickets:
                            await self._prefetch_typing_ticket(msg.from_user_id)

                        # Start typing indicator if we have ticket and not already typing
                        if (
                            msg.from_user_id in self._user_typing_tickets
                            and msg.from_user_id not in self._active_typing_tasks
                        ):
                            typing_ticket = self._user_typing_tickets[msg.from_user_id]
                            stop_event = asyncio.Event()
                            typing_task = asyncio.create_task(
                                self._typing_keepalive(
                                    msg.from_user_id, stop_event, typing_ticket
                                )
                            )
                            self._active_typing_tasks[msg.from_user_id] = (
                                typing_task,
                                stop_event,
                            )
                            logger.info(
                                f"[WechatIlinkaiAdapter#{self._instance_id}] Started typing indicator immediately for {msg.from_user_id}"
                            )

                    # Convert and emit message to gateway for processing
                    gateway_msg = self._convert_to_gateway_message(msg)
                    self._emit_message(gateway_msg)

                # If we get here, poll_messages ended (shouldn't happen normally)
                logger.warning(
                    "[WechatIlinkaiAdapter#{self._instance_id}] poll_messages generator ended unexpectedly"
                )

            except WeixinSessionExpiredError:
                # 检查适配器是否仍在运行（避免已停止的适配器发送事件）
                if not self._running:
                    logger.info(
                        f"[WechatIlinkaiAdapter#{self._instance_id}] Session expired but adapter already stopped, ignoring"
                    )
                    break

                logger.error(
                    f"[WechatIlinkaiAdapter#{self._instance_id}] Session expired, login required"
                )
                self._emit_event(
                    {
                        "type": "session_expired",
                        "account_id": self.account_id,
                        "message": "WeChat session expired. Please run 'memento wechat-login' to re-authenticate.",
                    }
                )
                # Stop polling, wait for manual intervention
                self._running = False
                break

            except asyncio.CancelledError:
                logger.info(
                    "[WechatIlinkaiAdapter#{self._instance_id}] Polling task cancelled"
                )
                raise

            except Exception as e:
                logger.error(
                    f"[WechatIlinkaiAdapter#{self._instance_id}] Polling error: {e}"
                )
                await asyncio.sleep(retry_delay)
                # Exponential backoff with max 60s
                retry_delay = min(retry_delay * 1.5, 60.0)

    async def _typing_keepalive(
        self,
        user_id: str,
        stop_event: asyncio.Event,
        typing_ticket: str,
    ) -> None:
        """Send typing indicator every 3 seconds until stopped.

        WeChat typing indicator expires after ~5-10 seconds, so we need to
        refresh it periodically while the Agent is processing.
        """
        if not self._client or not typing_ticket:
            logger.warning(
                f"[WechatIlinkaiAdapter#{self._instance_id}] Cannot start typing: client={self._client is not None}, ticket={typing_ticket is not None}"
            )
            return

        logger.info(
            f"[WechatIlinkaiAdapter#{self._instance_id}] Starting typing keepalive for {user_id}"
        )

        try:
            iteration = 0
            last_send_time = 0
            while not stop_event.is_set():
                iteration += 1
                current_time = time.time()

                try:
                    logger.info(
                        f"[WechatIlinkaiAdapter#{self._instance_id}] Sending typing indicator #{iteration} to {user_id}"
                    )
                    await self._client.send_typing(
                        ilink_user_id=user_id,
                        typing_ticket=typing_ticket,
                        status=1,  # 1 = typing
                    )
                    last_send_time = time.time()
                    logger.info(
                        f"[WechatIlinkaiAdapter#{self._instance_id}] Successfully sent typing indicator #{iteration} to {user_id}"
                    )
                except Exception as e:
                    logger.error(
                        f"[WechatIlinkaiAdapter#{self._instance_id}] Typing indicator error: {e}"
                    )

                # Wait 3 seconds or until stopped (more frequent updates)
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=3.0)
                    # Stop event was set - ensure minimum 2 seconds of typing display
                    elapsed = time.time() - last_send_time
                    if elapsed < 2.0 and iteration == 1:
                        # If we only sent once and it's been less than 2 seconds,
                        # wait a bit more so user can see the indicator
                        remaining = 2.0 - elapsed
                        logger.info(
                            f"[WechatIlinkaiAdapter#{self._instance_id}] Ensuring minimum typing display time ({remaining:.1f}s)"
                        )
                        try:
                            await asyncio.wait_for(stop_event.wait(), timeout=remaining)
                        except asyncio.TimeoutError:
                            pass
                    logger.info(
                        f"[WechatIlinkaiAdapter#{self._instance_id}] Typing stopped for {user_id}"
                    )
                    break
                except asyncio.TimeoutError:
                    continue  # Send next keepalive

        except asyncio.CancelledError:
            logger.info(
                f"[WechatIlinkaiAdapter#{self._instance_id}] Typing keepalive cancelled for {user_id}"
            )
            pass

    async def _prefetch_typing_ticket(self, user_id: str) -> None:
        """Pre-fetch typing ticket for a user to reduce latency in send_message.

        This is called when a message is received, so the ticket is ready
        when we need to send a response with typing indicator.
        """
        if not self._client:
            return

        # Skip if already cached
        if user_id in self._user_typing_tickets:
            return

        try:
            start_time = time.time()
            config_resp = await self._client.get_config(user_id=user_id)
            elapsed = (time.time() - start_time) * 1000

            if config_resp and config_resp.typing_ticket:
                self._user_typing_tickets[user_id] = config_resp.typing_ticket
                logger.info(
                    f"[WechatIlinkaiAdapter#{self._instance_id}] Pre-fetched typing ticket for {user_id} "
                    f"in {elapsed:.1f}ms"
                )
            else:
                logger.debug(
                    f"[WechatIlinkaiAdapter#{self._instance_id}] No typing ticket in pre-fetch for {user_id} "
                    f"({elapsed:.1f}ms)"
                )
        except Exception as e:
            logger.debug(
                f"[WechatIlinkaiAdapter#{self._instance_id}] Pre-fetch typing ticket failed for {user_id}: {e}"
            )

    def _convert_to_gateway_message(self, msg: WeixinMessage) -> GatewayMessage:
        """
        Convert WeixinMessage to GatewayMessage.

        Maps SDK message format to Gateway unified format.
        """
        # Extract content and type from first item
        content = ""
        msg_type = "text"

        if msg.item_list:
            first_item = msg.item_list[0]

            if first_item.type == MessageItemType.TEXT:
                content = first_item.text_item.text if first_item.text_item else ""
                msg_type = "text"
            elif first_item.type == MessageItemType.IMAGE:
                content = "[图片]"
                msg_type = "image"
            elif first_item.type == MessageItemType.VIDEO:
                content = "[视频]"
                msg_type = "video"
            elif first_item.type == MessageItemType.FILE:
                content = "[文件]"
                msg_type = "file"
            elif first_item.type == MessageItemType.VOICE:
                content = "[语音]"
                msg_type = "voice"

        # Determine chat type - WeixinMessage has group_id field for group chats
        chat_type = "group" if msg.group_id else "direct"

        return GatewayMessage(
            type=GatewayMsgType.CHANNEL_MESSAGE,
            channel_type=self.channel_type,
            channel_account=self.account_id,
            chat_id=msg.from_user_id or "",
            sender_id=msg.from_user_id or "",
            content=content,
            msg_type=msg_type,
            timestamp=time.time(),
            metadata={
                "weixin_message_id": msg.message_id,
                "context_token": msg.context_token,  # Key for session continuity
                "chat_type": chat_type,
                "sender_name": "",  # WeChat doesn't provide nickname in API
                "raw_message": msg.model_dump()
                if hasattr(msg, "model_dump")
                else str(msg),
            },
        )

    async def send_message(
        self, chat_id: str, content: str, msg_type: str = "text", **kwargs
    ) -> str:
        """Send message to WeChat user with typing indicator (if available)."""
        start_time = time.time()

        if not self._client:
            raise RuntimeError(
                "[WechatIlinkaiAdapter#{self._instance_id}] Not initialized"
            )

        context_token = kwargs.get("metadata", {}).get("context_token")

        # Check if we already have an active typing task from _polling_loop
        typing_task = None
        stop_typing = None
        ticket_fetch_time = 0

        if chat_id in self._active_typing_tasks:
            # Use the existing typing task started when message was received
            typing_task, stop_typing = self._active_typing_tasks[chat_id]
            logger.info(
                f"[WechatIlinkaiAdapter#{self._instance_id}] Using existing typing indicator for {chat_id} "
                f"(started when message received)"
            )
        elif self._client and chat_id:
            # No existing typing, try to start one (fallback)
            ticket_start = time.time()

            # Check cache first
            if chat_id in self._user_typing_tickets:
                typing_ticket = self._user_typing_tickets[chat_id]
                ticket_fetch_time = (time.time() - ticket_start) * 1000
                logger.info(
                    f"[WechatIlinkaiAdapter#{self._instance_id}] Using cached typing ticket for {chat_id} "
                    f"({ticket_fetch_time:.1f}ms)"
                )
            else:
                # Fetch typing ticket for this user
                try:
                    logger.info(
                        f"[WechatIlinkaiAdapter#{self._instance_id}] Fetching typing ticket for user {chat_id}..."
                    )
                    config_resp = await self._client.get_config(user_id=chat_id)
                    ticket_fetch_time = (time.time() - ticket_start) * 1000

                    if config_resp and config_resp.typing_ticket:
                        typing_ticket = config_resp.typing_ticket
                        self._user_typing_tickets[chat_id] = typing_ticket
                        logger.info(
                            f"[WechatIlinkaiAdapter#{self._instance_id}] Got typing ticket for user {chat_id} "
                            f"in {ticket_fetch_time:.1f}ms: {typing_ticket[:30]}..."
                        )
                    else:
                        logger.warning(
                            f"[WechatIlinkaiAdapter#{self._instance_id}] No typing ticket for user {chat_id} "
                            f"({ticket_fetch_time:.1f}ms)"
                        )
                except Exception as e:
                    ticket_fetch_time = (time.time() - ticket_start) * 1000
                    logger.warning(
                        f"[WechatIlinkaiAdapter#{self._instance_id}] Failed to get typing ticket for {chat_id} "
                        f"({ticket_fetch_time:.1f}ms): {e}"
                    )

            # Start typing indicator if we have ticket
            if typing_ticket:
                stop_typing = asyncio.Event()
                typing_task = asyncio.create_task(
                    self._typing_keepalive(chat_id, stop_typing, typing_ticket)
                )
                self._active_typing_tasks[chat_id] = (typing_task, stop_typing)
                logger.info(
                    f"[WechatIlinkaiAdapter#{self._instance_id}] Started typing indicator for {chat_id}"
                )

        try:
            send_start = time.time()
            if msg_type == "text":
                message_id = await self._client.send_text(
                    to_user_id=chat_id,
                    text=content,
                    context_token=context_token,
                )
            elif msg_type == "image":
                file_path = kwargs.get("file_path")
                if not file_path:
                    raise ValueError("file_path required for image messages")
                message_id = await self._client.send_image(
                    to_user_id=chat_id,
                    image_path=file_path,
                    context_token=context_token,
                )
            elif msg_type == "file":
                file_path = kwargs.get("file_path")
                if not file_path:
                    raise ValueError("file_path required for file messages")
                message_id = await self._client.send_file(
                    to_user_id=chat_id,
                    file_path=file_path,
                    context_token=context_token,
                )
            elif msg_type == "video":
                file_path = kwargs.get("file_path")
                if not file_path:
                    raise ValueError("file_path required for video messages")
                message_id = await self._client.send_video(
                    to_user_id=chat_id,
                    video_path=file_path,
                    context_token=context_token,
                )
            else:
                message_id = await self._client.send_message(
                    to_user_id=chat_id,
                    content=content,
                    msg_type=msg_type,
                    context_token=context_token,
                )

            send_time = (time.time() - send_start) * 1000
            total_time = (time.time() - start_time) * 1000

            logger.info(
                f"[WechatIlinkaiAdapter#{self._instance_id}] Message sent to {chat_id} "
                f"in {send_time:.1f}ms (total: {total_time:.1f}ms, ticket: {ticket_fetch_time:.1f}ms)"
            )
            return str(message_id)

        finally:
            # Stop typing indicator after message is sent
            if stop_typing:
                stop_typing.set()
                if typing_task:
                    try:
                        await asyncio.wait_for(typing_task, timeout=2.0)
                    except asyncio.TimeoutError:
                        typing_task.cancel()
                        try:
                            await typing_task
                        except asyncio.CancelledError:
                            pass
                    logger.info(
                        f"[WechatIlinkaiAdapter#{self._instance_id}] Stopped typing indicator for {chat_id}"
                    )
                # Remove from active tasks
                if chat_id in self._active_typing_tasks:
                    del self._active_typing_tasks[chat_id]

    async def reply_message(
        self, message_id: str, content: str, chat_id: str = "", **kwargs
    ) -> str:
        """Reply to message (WeChat personal doesn't distinguish reply from normal send)."""
        return await self.send_message(chat_id, content, **kwargs)

    async def health_check(self) -> bool:
        """Check if adapter is healthy."""
        if not self._client:
            return False
        return (
            self._running
            and self._polling_task is not None
            and not self._polling_task.done()
        )

    def simulate_session_expired(self) -> None:
        """模拟会话过期（用于测试自动登录功能）。

        调用后会触发 WeixinSessionExpiredError，导致适配器停止并发送 session_expired 事件。
        这会触发 GUI 自动弹出登录对话框。
        """
        logger.warning(
            f"[WechatIlinkaiAdapter#{self._instance_id}] Simulating session expiration for testing"
        )

        # 停止轮询循环
        self._running = False

        # 发送 session_expired 事件
        self._emit_event(
            {
                "type": "session_expired",
                "account_id": self.account_id,
                "message": "WeChat session expired (simulated). Please run 'memento wechat-login' to re-authenticate.",
                "simulated": True,
            }
        )

        logger.info(
            f"[WechatIlinkaiAdapter#{self._instance_id}] Session expiration simulation completed"
        )


# 模块级别的测试函数
def simulate_wechat_session_expired(account_id: str | None = None) -> bool:
    """模拟指定微信账户的会话过期。

    这是一个测试工具函数，用于手动触发微信会话过期流程。
    可以在代码中调用此函数来测试自动登录对话框。

    Args:
        account_id: 账户ID，如果为 None 则自动查找运行的微信适配器

    Returns:
        bool: 是否成功触发模拟

    Example:
        # 在任意地方调用（如点击测试按钮时）
        from middleware.im.gateway.channels.wechat_ilinkai import simulate_wechat_session_expired
        simulate_wechat_session_expired()
    """
    # 从注册表中查找适配器实例
    from ..gateway import _global_registry

    # 如果指定了 account_id，直接查找
    if account_id:
        adapter = _global_registry.get_adapter(account_id)
        if adapter and isinstance(adapter, WechatIlinkaiAdapter):
            adapter.simulate_session_expired()
            logger.info(
                f"[simulate_wechat_session_expired] Triggered for account: {account_id}"
            )
            return True

    # 如果没有指定或没找到，尝试常见 account_id
    common_ids = ["wechat_main", "default"]
    for aid in common_ids:
        adapter = _global_registry.get_adapter(aid)
        if adapter and isinstance(adapter, WechatIlinkaiAdapter):
            adapter.simulate_session_expired()
            logger.info(
                f"[simulate_wechat_session_expired] Triggered for account: {aid}"
            )
            return True

    # 最后尝试遍历所有适配器查找微信适配器
    for aid, adapter in _global_registry._adapters.items():
        if isinstance(adapter, WechatIlinkaiAdapter):
            adapter.simulate_session_expired()
            logger.info(
                f"[simulate_wechat_session_expired] Triggered for account: {aid}"
            )
            return True

    logger.warning(
        f"[simulate_wechat_session_expired] WeChat adapter not found in registry"
    )
    return False
