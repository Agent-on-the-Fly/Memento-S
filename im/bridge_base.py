"""IM Bridge 基类 - 封装 MementoSAgent 调用逻辑

参照 GUI MessageController 实现，提供统一的 Agent 调用接口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable

from shared.chat.session_manager import SessionManager
from shared.chat.conversation_manager import ConversationManager
from shared.chat.types import SessionInfo
from core.memento_s.agent import MementoSAgent
from core.protocol import (
    AGUIEventPipeline,
    AGUIEventType,
    PersistenceSink,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class IMBridgeBase(ABC):
    """IM 桥接基类，封装 Agent 调用逻辑

    子类需要实现:
        - on_message(): 处理收到的消息
        - send_reply(): 发送回复到 IM 平台

    提供的能力:
        - Agent 初始化和管理
        - Session/Conversation 持久化
        - 消息处理流程编排
    """

    def __init__(self, platform: str = "unknown"):
        self.platform = platform
        self._agent: MementoSAgent | None = None
        self._session_manager: SessionManager | None = None
        self._conversation_manager: ConversationManager | None = None
        self._sender_sessions: dict[str, str] = {}  # sender_id -> session_id
        self._running = False
        self._logger = get_logger(f"{__name__}.{platform}")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def agent(self) -> MementoSAgent | None:
        return self._agent

    async def initialize(self) -> None:
        """初始化 Agent 和管理器"""
        if self._agent is not None:
            return

        self._logger.info(f"[{self.platform}] Initializing bridge...")
        self._agent = MementoSAgent()
        self._session_manager = SessionManager()
        self._conversation_manager = ConversationManager()
        self._logger.info(f"[{self.platform}] Bridge initialized")

    async def start(self) -> None:
        """启动桥接（子类可重写以添加平台特定逻辑）"""
        await self.initialize()
        self._running = True
        self._logger.info(f"[{self.platform}] Bridge started")

    async def stop(self) -> None:
        """停止桥接"""
        self._running = False
        self._logger.info(f"[{self.platform}] Bridge stopped")

    @abstractmethod
    async def on_message(self, sender_id: str, content: str, **kwargs) -> None:
        """处理收到的消息

        Args:
            sender_id: 发送者 ID
            content: 消息内容
            **kwargs: 平台特定参数（如 chat_id, message_id 等）
        """
        pass

    @abstractmethod
    async def send_reply(self, sender_id: str, content: str, **kwargs) -> None:
        """发送回复到 IM 平台

        Args:
            sender_id: 接收者 ID（通常是发送消息的用户）
            content: 回复内容
            **kwargs: 平台特定参数
        """
        pass

    async def process_with_agent(
        self,
        sender_id: str,
        content: str,
        on_event: Callable[[dict], None] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """核心方法：调用 Agent 处理消息

        Args:
            sender_id: 发送者 ID
            content: 消息内容
            on_event: 事件回调（可选）
            metadata: 额外元数据

        Returns:
            Agent 的最终回复文本
        """
        if self._agent is None:
            await self.initialize()

        # 获取或创建 session
        session_id = await self._get_or_create_session(sender_id)

        # 保存用户消息
        user_title = content[:50] + "..." if len(content) > 50 else content
        user_conv = await self._conversation_manager.create(
            session_id=session_id,
            role="user",
            title=user_title,
            content=content,
            meta_info=metadata or {},
        )

        final_text = ""
        start_time = datetime.now()
        step_count = 0

        async def _persist_reply(
            text: str, usage: dict[str, Any] | None = None
        ) -> None:
            nonlocal final_text
            final_text = text
            if not text:
                return

            duration = (datetime.now() - start_time).total_seconds()
            reply_title = text[:50] + "..." if len(text) > 50 else text
            await self._conversation_manager.create(
                session_id=session_id,
                role="assistant",
                title=reply_title,
                content=text,
                meta_info={
                    "reply_to": user_conv.id,
                    "steps": step_count,
                    "duration_seconds": duration,
                    "tokens": usage.get("total_tokens") if usage else None,
                },
            )

        pipeline = AGUIEventPipeline()
        pipeline.add_sink(PersistenceSink(callback=_persist_reply))

        try:
            async for event in self._agent.reply_stream(
                session_id=session_id,
                user_content=content,
            ):
                await pipeline.emit(event)

                # 事件回调
                if on_event:
                    try:
                        on_event(event)
                    except Exception as e:
                        self._logger.warning(
                            f"[{self.platform}] Event callback error: {e}"
                        )

                event_type = event.get("type")

                # 跟踪步骤数
                if event_type == AGUIEventType.STEP_STARTED:
                    step = int(event.get("step", 0))
                    step_count = max(step_count, step)

                # 错误处理
                elif event_type == AGUIEventType.RUN_ERROR:
                    error_msg = event.get("message", "Unknown error")
                    self._logger.error(f"[{self.platform}] Agent error: {error_msg}")
                    final_text = f"处理出错：{error_msg}"

        except Exception as e:
            self._logger.error(
                f"[{self.platform}] Agent processing error: {e}", exc_info=True
            )
            final_text = "处理出错，请稍后重试。"

        return final_text

    async def _get_or_create_session(self, sender_id: str) -> str:
        """获取或创建用户会话

        优先从缓存查找，然后查询数据库中已有的会话，最后才创建新会话。
        避免为同一用户重复创建多个会话。

        Args:
            sender_id: 发送者 ID

        Returns:
            session_id
        """
        cache_key = f"{self.platform}:{sender_id}"

        # 1. 检查内存缓存
        if cache_key in self._sender_sessions:
            session_id = self._sender_sessions[cache_key]
            if await self._session_manager.exists(session_id):
                return session_id
            # session 不存在，删除缓存
            del self._sender_sessions[cache_key]

        # 2. 查询数据库中是否已有该用户的会话
        existing_session = await self._find_existing_session(sender_id)
        if existing_session:
            session_id = existing_session.id
            self._sender_sessions[cache_key] = session_id
            self._logger.info(
                f"[{self.platform}] Reused existing session {session_id} for {sender_id}"
            )
            return session_id

        # 3. 创建新 session
        session = await self._session_manager.create(
            title=f"{self.platform}: {sender_id}",
            metadata={
                "platform": self.platform,
                "sender_id": sender_id,
            },
        )
        session_id = session.id
        self._sender_sessions[cache_key] = session_id

        self._logger.info(
            f"[{self.platform}] Created new session {session_id} for {sender_id}"
        )
        return session_id

    async def _find_existing_session(self, sender_id: str) -> SessionInfo | None:
        """查找已存在的会话（按 platform 和 sender_id）

        Args:
            sender_id: 发送者 ID

        Returns:
            找到的 SessionInfo 或 None
        """
        try:
            # 获取最近的 sessions，检查 metadata 匹配
            recent_sessions = await self._session_manager.list_recent(limit=100)
            for session in recent_sessions:
                meta = session.metadata or {}
                if (
                    meta.get("platform") == self.platform
                    and meta.get("sender_id") == sender_id
                ):
                    return session
        except Exception as e:
            self._logger.warning(
                f"[{self.platform}] Error finding existing session: {e}"
            )
        return None

    def get_session_id(self, sender_id: str) -> str | None:
        """获取 sender_id 对应的 session_id（如果已缓存）"""
        cache_key = f"{self.platform}:{sender_id}"
        return self._sender_sessions.get(cache_key)

    def load_session_mapping(self, mapping: dict[str, str]) -> None:
        """加载 session 映射（用于从文件恢复）"""
        for key, session_id in mapping.items():
            # 兼容旧格式（不带平台前缀）
            if ":" not in key:
                key = f"{self.platform}:{key}"
            self._sender_sessions[key] = session_id
        self._logger.info(f"[{self.platform}] Loaded {len(mapping)} session mappings")

    def dump_session_mapping(self) -> dict[str, str]:
        """导出 session 映射（用于持久化）"""
        return dict(self._sender_sessions)
