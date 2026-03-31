"""
连接管理器。

管理所有长连接的生命周期，包括：
- 连接注册/注销
- 心跳保活
- 消息收发队列
- 会话路由映射
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from .protocol import (
    ConnectionConfig,
    ConnectionState,
    ConnectionType,
    GatewayMessage,
    MessageType,
    PermissionDomain,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 连接信息
# ---------------------------------------------------------------------------

@dataclass
class ConnectionInfo:
    """
    连接运行时信息。

    记录连接的完整状态和统计信息。
    """
    # 基本信息
    connection_id: str
    connection_type: ConnectionType
    config: ConnectionConfig

    # WebSocket 连接对象
    websocket: Any = None

    # 状态
    state: ConnectionState = ConnectionState.CONNECTING
    connected_at: datetime | None = None
    authenticated: bool = False

    # 活动时间
    last_activity: float = 0.0
    last_ping: float = 0.0
    last_pong: float = 0.0

    # 统计
    messages_sent: int = 0
    messages_received: int = 0

    # 发送队列
    send_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    # 权限
    permission_domain: PermissionDomain = PermissionDomain.NODE

    # 订阅的工具列表（仅 tool 类型连接）
    subscribed_tools: list[str] = field(default_factory=list)

    # 元数据
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 连接池
# ---------------------------------------------------------------------------

class ConnectionPool:
    """
    连接池。

    维护所有活跃连接，支持按类型、ID查询。
    """

    def __init__(self):
        # 连接存储
        self._connections: dict[str, ConnectionInfo] = {}

        # 按类型索引
        self._by_type: dict[ConnectionType, list[str]] = defaultdict(list)

        # 按来源索引（source -> connection_id）
        self._by_source: dict[str, str] = {}

        # 按渠道账户索引
        self._by_channel: dict[str, str] = {}

    def add(self, info: ConnectionInfo) -> None:
        """添加连接到池。"""
        self._connections[info.connection_id] = info
        self._by_type[info.connection_type].append(info.connection_id)

        # 建立渠道账户索引
        if info.config.channel_account:
            self._by_channel[info.config.channel_account] = info.connection_id

        logger.info(
            "Connection added: %s (type=%s, domain=%s, total=%d)",
            info.connection_id,
            info.connection_type.value,
            info.permission_domain.value,
            len(self._connections),
        )

    def remove(self, connection_id: str) -> ConnectionInfo | None:
        """从池中移除连接。"""
        info = self._connections.pop(connection_id, None)
        if info:
            # 从类型索引移除
            if connection_id in self._by_type.get(info.connection_type, []):
                self._by_type[info.connection_type].remove(connection_id)

            # 从来源索引移除
            source_key = self._find_source_key(connection_id)
            if source_key:
                self._by_source.pop(source_key, None)

            # 从渠道索引移除
            if info.config.channel_account:
                self._by_channel.pop(info.config.channel_account, None)

        return info

    def get(self, connection_id: str) -> ConnectionInfo | None:
        """获取连接信息。"""
        return self._connections.get(connection_id)

    def get_by_source(self, source: str) -> ConnectionInfo | None:
        """按来源获取连接。"""
        conn_id = self._by_source.get(source)
        return self._connections.get(conn_id) if conn_id else None

    def get_by_channel(self, channel_account: str) -> ConnectionInfo | None:
        """按渠道账户获取连接。"""
        conn_id = self._by_channel.get(channel_account)
        return self._connections.get(conn_id) if conn_id else None

    def list_by_type(self, conn_type: ConnectionType) -> list[ConnectionInfo]:
        """按类型列出连接。"""
        conn_ids = self._by_type.get(conn_type, [])
        return [
            self._connections[cid]
            for cid in conn_ids
            if cid in self._connections
        ]

    def list_all(self) -> list[ConnectionInfo]:
        """列出所有连接。"""
        return list(self._connections.values())

    def count(self, conn_type: ConnectionType | None = None) -> int:
        """统计连接数。"""
        if conn_type:
            return len([
                cid for cid in self._by_type.get(conn_type, [])
                if cid in self._connections
            ])
        return len(self._connections)

    def register_source(self, source: str, connection_id: str) -> None:
        """注册来源标识到连接的映射。"""
        self._by_source[source] = connection_id

    def _find_source_key(self, connection_id: str) -> str | None:
        """查找连接对应的来源键。"""
        for source, conn_id in self._by_source.items():
            if conn_id == connection_id:
                return source
        return None


# ---------------------------------------------------------------------------
# 会话路由器
# ---------------------------------------------------------------------------

class SessionRouter:
    """
    会话路由器。

    管理会话与连接的映射关系，用于消息路由。
    """

    def __init__(self):
        # session_id -> connection_id
        self._session_connections: dict[str, str] = {}

        # connection_id -> session_ids
        self._connection_sessions: dict[str, set[str]] = defaultdict(set)

    def bind(self, session_id: str, connection_id: str) -> None:
        """绑定会话到连接。"""
        self._session_connections[session_id] = connection_id
        self._connection_sessions[connection_id].add(session_id)

        logger.debug(
            "Session bound: %s -> %s",
            session_id,
            connection_id,
        )

    def unbind(self, session_id: str) -> None:
        """解除会话绑定。"""
        connection_id = self._session_connections.pop(session_id, None)
        if connection_id:
            self._connection_sessions[connection_id].discard(session_id)

    def get_connection(self, session_id: str) -> str | None:
        """获取会话关联的连接ID。"""
        return self._session_connections.get(session_id)

    def get_sessions(self, connection_id: str) -> set[str]:
        """获取连接关联的所有会话。"""
        return self._connection_sessions.get(connection_id, set())

    def clear_connection(self, connection_id: str) -> None:
        """清除连接的所有会话绑定。"""
        sessions = self._connection_sessions.pop(connection_id, set())
        for session_id in sessions:
            self._session_connections.pop(session_id, None)


# ---------------------------------------------------------------------------
# 连接管理器
# ---------------------------------------------------------------------------

class ConnectionManager:
    """
    连接管理器。

    管理所有长连接的生命周期：
    - 注册与注销
    - 心跳保活
    - 消息发送
    - 会话路由
    """

    def __init__(self):
        self.pool = ConnectionPool()
        self.router = SessionRouter()

        # 后台任务
        self._heartbeat_tasks: dict[str, asyncio.Task] = {}
        self._send_tasks: dict[str, asyncio.Task] = {}
        self._receive_tasks: dict[str, asyncio.Task] = {}

        # 回调
        self._message_callbacks: list[Callable] = []
        self._event_callbacks: list[Callable] = []

        # 运行状态
        self._running = False

    # ---- 生命周期管理 ----

    async def start(self) -> None:
        """启动连接管理器。"""
        self._running = True
        logger.info("ConnectionManager started")

    async def stop(self) -> None:
        """停止连接管理器。"""
        self._running = False

        # 取消所有后台任务
        for tasks in [self._heartbeat_tasks, self._send_tasks, self._receive_tasks]:
            for task in tasks.values():
                task.cancel()

        # 清空连接池
        for conn_id in list(self.pool._connections.keys()):
            await self.unregister(conn_id)

        logger.info("ConnectionManager stopped")

    # ---- 连接注册 ----

    async def register(
        self,
        websocket: Any,
        config: ConnectionConfig,
    ) -> ConnectionInfo:
        """
        注册新连接。

        Args:
            websocket: WebSocket 连接对象
            config: 连接配置

        Returns:
            ConnectionInfo: 连接信息
        """
        info = ConnectionInfo(
            connection_id=config.connection_id,
            connection_type=config.connection_type,
            config=config,
            websocket=websocket,
            state=ConnectionState.CONNECTED,
            connected_at=datetime.now(),
            last_activity=time.time(),
            permission_domain=config.permission_domain,
        )

        # 添加到连接池
        self.pool.add(info)

        # 启动发送任务
        self._send_tasks[config.connection_id] = asyncio.create_task(
            self._send_loop(info)
        )

        # 启动心跳任务
        if config.heartbeat_interval_ms > 0:
            self._heartbeat_tasks[config.connection_id] = asyncio.create_task(
                self._heartbeat_loop(info)
            )

        logger.info(
            "Connection registered: %s (type=%s, domain=%s)",
            config.connection_id,
            config.connection_type.value,
            config.permission_domain.value,
        )

        return info

    async def unregister(self, connection_id: str) -> None:
        """
        注销连接。

        Args:
            connection_id: 连接ID
        """
        # 取消后台任务
        for tasks in [self._heartbeat_tasks, self._send_tasks, self._receive_tasks]:
            task = tasks.pop(connection_id, None)
            if task:
                task.cancel()

        # 清除会话绑定
        self.router.clear_connection(connection_id)

        # 从连接池移除
        info = self.pool.remove(connection_id)
        if info:
            info.state = ConnectionState.DISCONNECTED
            logger.info("Connection unregistered: %s", connection_id)

    # ---- 来源注册 ----

    def register_source(self, source: str, connection_id: str) -> None:
        """
        注册来源标识。

        用于将消息路由到正确的连接。
        例如：agent_1 -> conn_123
        """
        self.pool.register_source(source, connection_id)
        logger.debug("Source registered: %s -> %s", source, connection_id)

    # ---- 消息发送 ----

    async def send(
        self,
        connection_id: str,
        message: GatewayMessage,
    ) -> bool:
        """
        发送消息到指定连接。

        消息将放入发送队列，由发送循环异步处理。

        Args:
            connection_id: 连接ID
            message: 消息对象

        Returns:
            bool: 是否成功加入队列
        """
        info = self.pool.get(connection_id)
        if not info or info.state != ConnectionState.CONNECTED:
            logger.warning(
                "Cannot send to %s: connection not found or not connected",
                connection_id,
            )
            return False

        try:
            await info.send_queue.put(message)
            return True
        except Exception as e:
            logger.error("Failed to queue message for %s: %s", connection_id, e)
            return False

    async def send_to_source(
        self,
        source: str,
        message: GatewayMessage,
    ) -> bool:
        """发送消息到指定来源。"""
        info = self.pool.get_by_source(source)
        if info:
            return await self.send(info.connection_id, message)
        return False

    async def send_to_channel(
        self,
        channel_account: str,
        message: GatewayMessage,
    ) -> bool:
        """发送消息到指定渠道账户。"""
        info = self.pool.get_by_channel(channel_account)
        if info:
            return await self.send(info.connection_id, message)
        return False

    async def broadcast(
        self,
        message: GatewayMessage,
        connection_type: ConnectionType | None = None,
    ) -> int:
        """
        广播消息。

        Args:
            message: 消息对象
            connection_type: 连接类型过滤（None 表示所有）

        Returns:
            int: 成功发送的连接数
        """
        connections = (
            self.pool.list_by_type(connection_type)
            if connection_type
            else self.pool.list_all()
        )

        sent = 0
        for info in connections:
            if info.state == ConnectionState.CONNECTED:
                try:
                    await info.send_queue.put(message)
                    sent += 1
                except Exception:
                    pass

        return sent

    # ---- 会话路由 ----

    def bind_session(self, session_id: str, connection_id: str) -> None:
        """绑定会话到连接。"""
        self.router.bind(session_id, connection_id)

    def get_connection_for_session(self, session_id: str) -> str | None:
        """获取会话关联的连接ID。"""
        return self.router.get_connection(session_id)

    # ---- 回调注册 ----

    def on_message(self, callback: Callable[[ConnectionInfo, GatewayMessage], None]) -> None:
        """注册消息回调。"""
        self._message_callbacks.append(callback)

    def on_event(self, callback: Callable[[str, dict], None]) -> None:
        """注册事件回调。"""
        self._event_callbacks.append(callback)

    # ---- 内部循环 ----

    async def _send_loop(self, info: ConnectionInfo) -> None:
        """发送循环。"""
        try:
            while info.state in (ConnectionState.CONNECTED, ConnectionState.AUTHENTICATED):
                message = await info.send_queue.get()

                try:
                    await info.websocket.send(message.to_json())
                    info.messages_sent += 1
                    info.last_activity = time.time()

                except Exception as e:
                    logger.error("Send error %s: %s", info.connection_id, e)
                    info.state = ConnectionState.ERROR
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Send loop error %s: %s", info.connection_id, e)

    async def _heartbeat_loop(self, info: ConnectionInfo) -> None:
        """心跳循环。"""
        config = info.config

        try:
            while info.state in (ConnectionState.CONNECTED, ConnectionState.AUTHENTICATED):
                await asyncio.sleep(config.heartbeat_interval_ms / 1000)

                if info.state not in (ConnectionState.CONNECTED, ConnectionState.AUTHENTICATED):
                    continue

                # 发送 PING
                info.last_ping = time.time()
                ping = GatewayMessage(
                    type=MessageType.PING,
                    connection_id=info.connection_id,
                    timestamp=time.time(),
                )
                await info.send_queue.put(ping)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Heartbeat error %s: %s", info.connection_id, e)

    # ---- 消息处理 ----

    async def handle_message(
        self,
        connection_id: str,
        message: GatewayMessage,
    ) -> None:
        """
        处理接收到的消息。

        由 WebSocket 接收循环调用。
        """
        info = self.pool.get(connection_id)
        if not info:
            return

        info.messages_received += 1
        info.last_activity = time.time()

        # 处理 PONG
        if message.type == MessageType.PONG:
            info.last_pong = time.time()
            return

        # 处理 CONNECT 消息
        if message.type == MessageType.CONNECT:
            await self._handle_connect(info, message)
            return

        # 触发消息回调
        for callback in self._message_callbacks:
            try:
                result = callback(info, message)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("Message callback error: %s", e)

    async def _handle_connect(self, info: ConnectionInfo, message: GatewayMessage) -> None:
        """处理 CONNECT 消息。"""
        source = message.source
        source_type = message.source_type

        # 注册来源
        if source:
            self.register_source(source, info.connection_id)

        # 记录工具订阅（仅 tool 类型）
        if source_type == "tool":
            tools = message.metadata.get("tools", [])
            info.subscribed_tools = tools
            logger.info(
                "Tool worker registered: %s, tools: %s",
                source,
                tools,
            )

        # 发送确认
        ack = GatewayMessage(
            type=MessageType.CONNECT_ACK,
            target=source,
            target_type=source_type,
            connection_id=info.connection_id,
            metadata={"status": "connected"},
        )
        await info.send_queue.put(ack)

        info.authenticated = True
        info.state = ConnectionState.AUTHENTICATED

        logger.info(
            "Connection authenticated: %s (source=%s, type=%s)",
            info.connection_id,
            source,
            source_type,
        )

    # ---- 状态查询 ----

    def get_state(self, connection_id: str) -> ConnectionState:
        """获取连接状态。"""
        info = self.pool.get(connection_id)
        return info.state if info else ConnectionState.DISCONNECTED

    def get_stats(self) -> dict:
        """获取统计信息。"""
        return {
            "total": self.pool.count(),
            "by_type": {
                t.value: self.pool.count(t)
                for t in ConnectionType
            },
            "sessions": len(self.router._session_connections),
        }
