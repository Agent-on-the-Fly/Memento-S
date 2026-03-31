"""企业微信 Agent 桥接

封装企业微信平台的 Agent 调用逻辑，处理企业微信消息并与 Agent 交互。

企业微信智能机器人特点:
  - 通过 WebSocket 长连接接收消息
  - 消息 dict 中包含 "reply" 函数，可直接调用回复
  - 无需公网 IP，只需 bot_id 和 secret
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any, Callable

from im.bridge_base import IMBridgeBase
from middleware.config import g_config
from middleware.im.im_platform.wecom.receiver import WecomReceiver
from middleware.im.im_platform.wecom.platform import WecomPlatform
from core.protocol import AGUIEventType
from utils.logger import get_logger

logger = get_logger(__name__)


class WecomBridge(IMBridgeBase):
    """企业微信平台 Agent 桥接

    接收企业微信智能机器人消息，调用 Agent 处理，并回复结果。

    使用方式:
        bridge = WecomBridge()
        await bridge.start()  # 启动 WebSocket 接收

        # 或在后台运行
        bridge.start_in_background()
    """

    def __init__(self):
        super().__init__(platform="wecom")
        self._receiver: WecomReceiver | None = None
        self._platform: WecomPlatform | None = None
        self._bg_thread: threading.Thread | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._mapping_file: Path | None = None

    async def start(self) -> None:
        """启动企业微信桥接（阻塞模式）"""
        await self.initialize()

        # 加载持久化的 session 映射
        self._load_session_mapping()

        self._logger.info("[wecom] Starting bridge...")

        # 创建事件循环用于消息处理
        self._event_loop = asyncio.get_running_loop()

        # 创建接收器（企业微信的 on_message 必须是 async 函数）
        self._receiver = WecomReceiver(on_message=self._on_raw_message_async)

        # 标记运行中
        self._running = True

        # 启动接收器（阻塞）
        self._receiver.start()

    def start_in_background(self) -> threading.Thread:
        """在后台线程启动企业微信桥接

        Returns:
            后台线程对象
        """
        # 创建新的事件循环
        self._event_loop = asyncio.new_event_loop()

        def _run():
            asyncio.set_event_loop(self._event_loop)

            # 同步初始化
            self._load_session_mapping()

            # 初始化 Agent（在事件循环中）
            self._event_loop.run_until_complete(self.initialize())

            self._logger.info("[wecom] Background bridge starting...")

            # 创建接收器
            self._receiver = WecomReceiver(on_message=self._on_raw_message_async)
            self._running = True

            try:
                self._receiver.start()
            except Exception as e:
                self._logger.error(f"[wecom] Bridge error: {e}")
            finally:
                self._running = False

        self._bg_thread = threading.Thread(target=_run, daemon=True, name="wecom-bridge")
        self._bg_thread.start()

        self._logger.info("[wecom] Bridge started in background")
        return self._bg_thread

    async def stop(self) -> None:
        """停止企业微信桥接"""
        self._running = False

        # 保存 session 映射
        self._save_session_mapping()

        if self._receiver:
            self._receiver.stop()
            self._receiver = None

        self._logger.info("[wecom] Bridge stopped")

    async def on_message(
        self,
        sender_id: str,
        content: str,
        chat_id: str = "",
        message_id: str = "",
        chat_type: str = "p2p",
        reply_func: Callable[[str], None] | None = None,
        **kwargs,
    ) -> None:
        """处理企业微信消息

        Args:
            sender_id: 发送者 userid
            content: 消息内容
            chat_id: 会话 ID
            message_id: 消息 ID
            chat_type: p2p 或 group
            reply_func: 回复函数（从消息 dict 中提取）
        """
        if not content.strip():
            return

        self._logger.info(f"[wecom] Message from {sender_id}: {content[:50]}...")

        # 定义事件回调（用于日志）
        def on_event(event: dict) -> None:
            event_type = event.get("type")
            if event_type == AGUIEventType.TOOL_CALL_START:
                tool_name = event.get("toolName", "unknown")
                self._logger.info(f"[wecom] Tool call: {tool_name}")
            elif event_type == AGUIEventType.RUN_ERROR:
                self._logger.error(f"[wecom] Agent error: {event.get('message')}")

        # 调用 Agent 处理
        final_text = await self.process_with_agent(
            sender_id=sender_id,
            content=content,
            on_event=on_event,
            metadata={
                "chat_id": chat_id,
                "message_id": message_id,
                "chat_type": chat_type,
                "source": "wecom",
            },
        )

        # 发送回复
        if final_text:
            await self.send_reply(
                sender_id=sender_id,
                content=final_text,
                chat_id=chat_id,
                reply_func=reply_func,
            )

    async def send_reply(
        self,
        sender_id: str,
        content: str,
        chat_id: str = "",
        reply_func: Callable[[str], None] | None = None,
        **kwargs,
    ) -> None:
        """发送企业微信回复

        Args:
            sender_id: 接收者 userid
            content: 回复内容
            chat_id: 会话 ID
            reply_func: 回复函数（优先使用）
        """
        try:
            # 优先使用消息自带的 reply 函数
            if reply_func:
                await reply_func(content)
                self._logger.info(f"[wecom] Reply sent to {sender_id} via reply_func")
            elif self._receiver:
                # 使用 receiver 的 send_text 方法
                await self._receiver.send_text(chat_id=chat_id, text=content, to_user_id=sender_id)
                self._logger.info(f"[wecom] Reply sent to {sender_id} via send_text")
            else:
                self._logger.warning("[wecom] No method available to send reply")

        except Exception as e:
            self._logger.error(f"[wecom] Failed to send reply: {e}")

    async def _on_raw_message_async(self, msg: dict) -> None:
        """处理原始企业微信消息（异步回调）"""
        if not self._running:
            return

        sender_id = msg.get("sender_id", "")
        content = msg.get("content", "")
        chat_id = msg.get("chat_id", "")
        message_id = msg.get("id", "")
        chat_type = msg.get("chat_type", "p2p")
        reply_func = msg.get("reply")  # 企业微信特有的回复函数

        await self.on_message(
            sender_id=sender_id,
            content=content,
            chat_id=chat_id,
            message_id=message_id,
            chat_type=chat_type,
            reply_func=reply_func,
        )

    # -----------------------------------------------------------------------
    # Session 映射持久化
    # -----------------------------------------------------------------------

    def _mapping_path(self) -> Path:
        """获取 session 映射文件路径"""
        workspace = Path(g_config.paths.workspace_dir).expanduser().resolve()
        return workspace / "wecom_sessions.json"

    def _load_session_mapping(self) -> None:
        """从文件加载 session 映射"""
        try:
            path = self._mapping_path()
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                mapping = {}
                for sender_id, session_id in data.items():
                    key = sender_id if ":" in sender_id else f"wecom:{sender_id}"
                    mapping[key] = session_id
                self.load_session_mapping(mapping)
                self._logger.info(f"[wecom] Loaded {len(data)} session mappings")
        except Exception as e:
            self._logger.warning(f"[wecom] Failed to load session mapping: {e}")

    def _save_session_mapping(self) -> None:
        """保存 session 映射到文件"""
        try:
            path = self._mapping_path()
            path.parent.mkdir(parents=True, exist_ok=True)

            mapping = {}
            for key, session_id in self._sender_sessions.items():
                sender_id = key.replace("wecom:", "") if key.startswith("wecom:") else key
                mapping[sender_id] = session_id

            path.write_text(
                json.dumps(mapping, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._logger.info(f"[wecom] Saved {len(mapping)} session mappings")
        except Exception as e:
            self._logger.warning(f"[wecom] Failed to save session mapping: {e}")


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

_wecom_bridge: WecomBridge | None = None


def get_wecom_bridge() -> WecomBridge | None:
    """获取全局企业微信桥接实例"""
    return _wecom_bridge


async def start_wecom_bridge() -> WecomBridge:
    """启动全局企业微信桥接（阻塞模式）"""
    global _wecom_bridge
    if _wecom_bridge is None:
        _wecom_bridge = WecomBridge()
        await _wecom_bridge.start()
    return _wecom_bridge


def start_wecom_bridge_background() -> WecomBridge:
    """在后台启动全局企业微信桥接"""
    global _wecom_bridge
    if _wecom_bridge is None:
        _wecom_bridge = WecomBridge()
        _wecom_bridge.start_in_background()
    return _wecom_bridge


async def stop_wecom_bridge() -> None:
    """停止全局企业微信桥接"""
    global _wecom_bridge
    if _wecom_bridge:
        await _wecom_bridge.stop()
        _wecom_bridge = None
