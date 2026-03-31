"""飞书 Agent 桥接

封装飞书平台的 Agent 调用逻辑，处理飞书消息并与 Agent 交互。
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from im.bridge_base import IMBridgeBase
from middleware.config import g_config
from middleware.im.im_platform.feishu.receiver import FeishuReceiver
from middleware.im.im_platform.messaging import send_text_message
from core.protocol import AGUIEventType
from utils.logger import get_logger

logger = get_logger(__name__)


class FeishuBridge(IMBridgeBase):
    """飞书平台 Agent 桥接

    接收飞书消息，调用 Agent 处理，并回复结果。

    使用方式:
        bridge = FeishuBridge()
        await bridge.start()  # 启动 WebSocket 接收

        # 或在后台运行
        bridge.start_in_background()
    """

    def __init__(self):
        super().__init__(platform="feishu")
        self._receiver: FeishuReceiver | None = None
        self._bg_thread: threading.Thread | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._mapping_file: Path | None = None

    async def start(self) -> None:
        """启动飞书桥接（阻塞模式）"""
        await self.initialize()

        # 加载持久化的 session 映射
        self._load_session_mapping()

        self._logger.info("[feishu] Starting bridge...")

        # 创建事件循环用于消息处理
        self._event_loop = asyncio.get_running_loop()

        # 创建接收器
        self._receiver = FeishuReceiver(on_message=self._on_raw_message)

        # 标记运行中
        self._running = True

        # 启动接收器（阻塞）
        self._receiver.start()

    def start_in_background(self) -> threading.Thread:
        """在后台线程启动飞书桥接

        Returns:
            后台线程对象
        """
        # 初始化（同步部分）
        import asyncio

        # 创建新的事件循环
        self._event_loop = asyncio.new_event_loop()

        def _run():
            asyncio.set_event_loop(self._event_loop)

            # 同步初始化
            self._load_session_mapping()

            # 初始化 Agent（在事件循环中）
            self._event_loop.run_until_complete(self.initialize())

            self._logger.info("[feishu] Background bridge starting...")

            # 创建接收器
            self._receiver = FeishuReceiver(on_message=self._on_raw_message_sync)
            self._running = True

            try:
                self._receiver.start()
            except Exception as e:
                self._logger.error(f"[feishu] Bridge error: {e}")
            finally:
                self._running = False

        self._bg_thread = threading.Thread(target=_run, daemon=True)
        self._bg_thread.start()

        self._logger.info("[feishu] Bridge started in background")
        return self._bg_thread

    async def stop(self) -> None:
        """停止飞书桥接"""
        self._running = False

        # 保存 session 映射
        self._save_session_mapping()

        if self._receiver:
            self._receiver.stop()
            self._receiver = None

        self._logger.info("[feishu] Bridge stopped")

    async def on_message(
        self,
        sender_id: str,
        content: str,
        chat_id: str = "",
        message_id: str = "",
        **kwargs,
    ) -> None:
        """处理飞书消息

        Args:
            sender_id: 发送者 open_id
            content: 消息内容
            chat_id: 聊天 ID
            message_id: 消息 ID
        """
        if not content.strip():
            return

        self._logger.info(f"[feishu] Message from {sender_id}: {content[:50]}...")

        # 定义事件回调（用于日志）
        def on_event(event: dict) -> None:
            event_type = event.get("type")
            if event_type == AGUIEventType.TOOL_CALL_START:
                tool_name = event.get("toolName", "unknown")
                self._logger.info(f"[feishu] Tool call: {tool_name}")
            elif event_type == AGUIEventType.RUN_ERROR:
                self._logger.error(f"[feishu] Agent error: {event.get('message')}")

        # 调用 Agent 处理
        final_text = await self.process_with_agent(
            sender_id=sender_id,
            content=content,
            on_event=on_event,
            metadata={
                "chat_id": chat_id,
                "message_id": message_id,
                "source": "feishu",
            },
        )

        # 发送回复
        if final_text:
            await self.send_reply(sender_id, final_text, chat_id=chat_id)

    async def send_reply(
        self,
        sender_id: str,
        content: str,
        chat_id: str = "",
        **kwargs,
    ) -> None:
        """发送飞书回复

        Args:
            sender_id: 接收者 open_id
            content: 回复内容
            chat_id: 聊天 ID（可选）
        """
        try:
            await send_text_message(
                receive_id=sender_id,
                text=content,
                receive_id_type="open_id",
                platform="feishu",
            )
            self._logger.info(f"[feishu] Reply sent to {sender_id}")
        except Exception as e:
            self._logger.error(f"[feishu] Failed to send reply: {e}")

    def _on_raw_message(self, msg: dict) -> None:
        """处理原始飞书消息（异步回调）"""
        if not self._running:
            return

        sender_id = msg.get("sender_id", "")
        content = msg.get("content", "")
        chat_id = msg.get("chat_id", "")
        message_id = msg.get("id", "")

        # 在事件循环中调度处理
        if self._event_loop and self._event_loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.on_message(
                    sender_id=sender_id,
                    content=content,
                    chat_id=chat_id,
                    message_id=message_id,
                ),
                self._event_loop,
            )

    def _on_raw_message_sync(self, msg: dict) -> None:
        """处理原始飞书消息（同步回调，用于后台线程）"""
        if not self._running:
            return

        sender_id = msg.get("sender_id", "")
        content = msg.get("content", "")
        chat_id = msg.get("chat_id", "")
        message_id = msg.get("id", "")

        if self._event_loop:
            asyncio.run_coroutine_threadsafe(
                self.on_message(
                    sender_id=sender_id,
                    content=content,
                    chat_id=chat_id,
                    message_id=message_id,
                ),
                self._event_loop,
            )
            # 等待完成（可选，不阻塞消息接收）
            # future.result(timeout=60)

    # -----------------------------------------------------------------------
    # Session 映射持久化
    # -----------------------------------------------------------------------

    def _mapping_path(self) -> Path:
        """获取 session 映射文件路径"""
        workspace = Path(g_config.paths.workspace_dir).expanduser().resolve()
        return workspace / "feishu_sessions.json"

    def _load_session_mapping(self) -> None:
        """从文件加载 session 映射"""
        try:
            path = self._mapping_path()
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                # 转换为带平台前缀的格式
                mapping = {}
                for sender_id, session_id in data.items():
                    key = sender_id if ":" in sender_id else f"feishu:{sender_id}"
                    mapping[key] = session_id
                self.load_session_mapping(mapping)
                self._logger.info(f"[feishu] Loaded {len(data)} session mappings")
        except Exception as e:
            self._logger.warning(f"[feishu] Failed to load session mapping: {e}")

    def _save_session_mapping(self) -> None:
        """保存 session 映射到文件"""
        try:
            path = self._mapping_path()
            path.parent.mkdir(parents=True, exist_ok=True)

            # 转换为不带平台前缀的格式（兼容旧版本）
            mapping = {}
            for key, session_id in self._sender_sessions.items():
                # 移除 feishu: 前缀
                sender_id = key.replace("feishu:", "") if key.startswith("feishu:") else key
                mapping[sender_id] = session_id

            path.write_text(
                json.dumps(mapping, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._logger.info(f"[feishu] Saved {len(mapping)} session mappings")
        except Exception as e:
            self._logger.warning(f"[feishu] Failed to save session mapping: {e}")


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

_feishu_bridge: FeishuBridge | None = None


def get_feishu_bridge() -> FeishuBridge | None:
    """获取全局飞书桥接实例"""
    return _feishu_bridge


async def start_feishu_bridge() -> FeishuBridge:
    """启动全局飞书桥接（阻塞模式）"""
    global _feishu_bridge
    if _feishu_bridge is None:
        _feishu_bridge = FeishuBridge()
        await _feishu_bridge.start()
    return _feishu_bridge


def start_feishu_bridge_background() -> FeishuBridge:
    """在后台启动全局飞书桥接"""
    global _feishu_bridge
    if _feishu_bridge is None:
        _feishu_bridge = FeishuBridge()
        _feishu_bridge.start_in_background()
    return _feishu_bridge


async def stop_feishu_bridge() -> None:
    """停止全局飞书桥接"""
    global _feishu_bridge
    if _feishu_bridge:
        await _feishu_bridge.stop()
        _feishu_bridge = None
