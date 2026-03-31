"""钉钉 Agent 桥接

封装钉钉平台的 Agent 调用逻辑，处理钉钉消息并与 Agent 交互。
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any, Callable

from im.bridge_base import IMBridgeBase
from middleware.config import g_config
from middleware.im.im_platform.dingtalk.receiver import DingTalkReceiver
from middleware.im.im_platform.dingtalk.platform import DingTalkPlatform
from core.protocol import AGUIEventType
from utils.logger import get_logger

logger = get_logger(__name__)


class DingtalkBridge(IMBridgeBase):
    """钉钉平台 Agent 桥接

    接收钉钉消息，调用 Agent 处理，并回复结果。

    使用方式:
        bridge = DingtalkBridge()
        await bridge.start()  # 启动 Stream 接收

        # 或在后台运行
        bridge.start_in_background()
    """

    def __init__(self):
        super().__init__(platform="dingtalk")
        self._receiver: DingTalkReceiver | None = None
        self._platform: DingTalkPlatform | None = None
        self._bg_thread: threading.Thread | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._mapping_file: Path | None = None

    async def start(self) -> None:
        """启动钉钉桥接（阻塞模式）"""
        await self.initialize()

        # 加载持久化的 session 映射
        self._load_session_mapping()

        self._logger.info("[dingtalk] Starting bridge...")

        # 初始化平台 API
        self._platform = DingTalkPlatform()

        # 创建事件循环用于消息处理
        self._event_loop = asyncio.get_running_loop()

        # 创建接收器
        self._receiver = DingTalkReceiver(on_message=self._on_raw_message_async)

        # 标记运行中
        self._running = True

        # 启动接收器（阻塞）
        self._receiver.start()

    def start_in_background(self) -> threading.Thread:
        """在后台线程启动钉钉桥接

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

            self._logger.info("[dingtalk] Background bridge starting...")

            # 初始化平台 API
            try:
                self._platform = DingTalkPlatform()
            except Exception as e:
                self._logger.error(f"[dingtalk] Failed to init platform: {e}")
                return

            # 创建接收器
            self._receiver = DingTalkReceiver(on_message=self._on_raw_message_sync)
            self._running = True

            try:
                self._receiver.start()
            except Exception as e:
                self._logger.error(f"[dingtalk] Bridge error: {e}")
            finally:
                self._running = False

        self._bg_thread = threading.Thread(target=_run, daemon=True, name="dingtalk-bridge")
        self._bg_thread.start()

        self._logger.info("[dingtalk] Bridge started in background")
        return self._bg_thread

    async def stop(self) -> None:
        """停止钉钉桥接"""
        self._running = False

        # 保存 session 映射
        self._save_session_mapping()

        if self._receiver:
            self._receiver.stop()
            self._receiver = None

        self._logger.info("[dingtalk] Bridge stopped")

    async def on_message(
        self,
        sender_id: str,
        content: str,
        chat_id: str = "",
        message_id: str = "",
        chat_type: str = "p2p",
        **kwargs,
    ) -> None:
        """处理钉钉消息

        Args:
            sender_id: 发送者 staff_id
            content: 消息内容
            chat_id: 会话 ID
            message_id: 消息 ID
            chat_type: p2p 或 group
        """
        if not content.strip():
            return

        self._logger.info(f"[dingtalk] Message from {sender_id}: {content[:50]}...")

        # 定义事件回调（用于日志）
        def on_event(event: dict) -> None:
            event_type = event.get("type")
            if event_type == AGUIEventType.TOOL_CALL_START:
                tool_name = event.get("toolName", "unknown")
                self._logger.info(f"[dingtalk] Tool call: {tool_name}")
            elif event_type == AGUIEventType.RUN_ERROR:
                self._logger.error(f"[dingtalk] Agent error: {event.get('message')}")

        # 调用 Agent 处理
        final_text = await self.process_with_agent(
            sender_id=sender_id,
            content=content,
            on_event=on_event,
            metadata={
                "chat_id": chat_id,
                "message_id": message_id,
                "chat_type": chat_type,
                "source": "dingtalk",
            },
        )

        # 发送回复
        if final_text:
            await self.send_reply(
                sender_id=sender_id,
                content=final_text,
                chat_id=chat_id,
                chat_type=chat_type,
            )

    async def send_reply(
        self,
        sender_id: str,
        content: str,
        chat_id: str = "",
        chat_type: str = "p2p",
        **kwargs,
    ) -> None:
        """发送钉钉回复

        Args:
            sender_id: 接收者 staff_id
            content: 回复内容
            chat_id: 会话 ID（群聊时使用）
            chat_type: p2p 或 group
        """
        try:
            if not self._platform:
                self._logger.error("[dingtalk] Platform not initialized")
                return

            # 使用平台 API 发送消息
            if chat_type == "group" and chat_id:
                # 群聊回复
                await self._platform.send_message(
                    receive_id=chat_id,
                    content=content,
                    msg_type="text",
                    receive_id_type="chat_id",
                )
            else:
                # 单聊回复
                await self._platform.send_message(
                    receive_id=sender_id,
                    content=content,
                    msg_type="text",
                    receive_id_type="staff_id",
                )

            self._logger.info(f"[dingtalk] Reply sent to {sender_id}")
        except Exception as e:
            self._logger.error(f"[dingtalk] Failed to send reply: {e}")

    async def _on_raw_message_async(self, msg: dict) -> None:
        """处理原始钉钉消息（异步回调）"""
        if not self._running:
            return

        sender_id = msg.get("sender_id", "")
        content = msg.get("content", "")
        chat_id = msg.get("chat_id", "")
        message_id = msg.get("id", "")
        chat_type = msg.get("chat_type", "p2p")

        await self.on_message(
            sender_id=sender_id,
            content=content,
            chat_id=chat_id,
            message_id=message_id,
            chat_type=chat_type,
        )

    def _on_raw_message_sync(self, msg: dict) -> None:
        """处理原始钉钉消息（同步回调，用于后台线程）"""
        if not self._running:
            return

        sender_id = msg.get("sender_id", "")
        content = msg.get("content", "")
        chat_id = msg.get("chat_id", "")
        message_id = msg.get("id", "")
        chat_type = msg.get("chat_type", "p2p")

        if self._event_loop:
            future = asyncio.run_coroutine_threadsafe(
                self.on_message(
                    sender_id=sender_id,
                    content=content,
                    chat_id=chat_id,
                    message_id=message_id,
                    chat_type=chat_type,
                ),
                self._event_loop,
            )

    # -----------------------------------------------------------------------
    # Session 映射持久化
    # -----------------------------------------------------------------------

    def _mapping_path(self) -> Path:
        """获取 session 映射文件路径"""
        workspace = Path(g_config.paths.workspace_dir).expanduser().resolve()
        return workspace / "dingtalk_sessions.json"

    def _load_session_mapping(self) -> None:
        """从文件加载 session 映射"""
        try:
            path = self._mapping_path()
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                mapping = {}
                for sender_id, session_id in data.items():
                    key = sender_id if ":" in sender_id else f"dingtalk:{sender_id}"
                    mapping[key] = session_id
                self.load_session_mapping(mapping)
                self._logger.info(f"[dingtalk] Loaded {len(data)} session mappings")
        except Exception as e:
            self._logger.warning(f"[dingtalk] Failed to load session mapping: {e}")

    def _save_session_mapping(self) -> None:
        """保存 session 映射到文件"""
        try:
            path = self._mapping_path()
            path.parent.mkdir(parents=True, exist_ok=True)

            mapping = {}
            for key, session_id in self._sender_sessions.items():
                sender_id = key.replace("dingtalk:", "") if key.startswith("dingtalk:") else key
                mapping[sender_id] = session_id

            path.write_text(
                json.dumps(mapping, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._logger.info(f"[dingtalk] Saved {len(mapping)} session mappings")
        except Exception as e:
            self._logger.warning(f"[dingtalk] Failed to save session mapping: {e}")


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

_dingtalk_bridge: DingtalkBridge | None = None


def get_dingtalk_bridge() -> DingtalkBridge | None:
    """获取全局钉钉桥接实例"""
    return _dingtalk_bridge


async def start_dingtalk_bridge() -> DingtalkBridge:
    """启动全局钉钉桥接（阻塞模式）"""
    global _dingtalk_bridge
    if _dingtalk_bridge is None:
        _dingtalk_bridge = DingtalkBridge()
        await _dingtalk_bridge.start()
    return _dingtalk_bridge


def start_dingtalk_bridge_background() -> DingtalkBridge:
    """在后台启动全局钉钉桥接"""
    global _dingtalk_bridge
    if _dingtalk_bridge is None:
        _dingtalk_bridge = DingtalkBridge()
        _dingtalk_bridge.start_in_background()
    return _dingtalk_bridge


async def stop_dingtalk_bridge() -> None:
    """停止全局钉钉桥接"""
    global _dingtalk_bridge
    if _dingtalk_bridge:
        await _dingtalk_bridge.stop()
        _dingtalk_bridge = None
