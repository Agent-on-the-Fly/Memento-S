"""
消息路由器。

负责消息的路由和分发：
- 会话路由：将消息路由到正确的 Agent
- 工具路由：将工具调用路由到正确的 Tool Worker
- 响应路由：将响应发回正确的渠道
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from .protocol import (
    GatewayMessage,
    MessageType,
    new_session_id,
)
from .connection_manager import ConnectionManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 路由规则
# ---------------------------------------------------------------------------

@dataclass
class RouteRule:
    """
    路由规则。

    定义消息如何被路由。
    """
    # 消息类型过滤
    message_types: list[MessageType] = field(default_factory=list)

    # 来源过滤
    source_types: list[str] = field(default_factory=list)

    # 目标选择器
    target_selector: Callable[[GatewayMessage], str | None] | None = None

    # 优先级（数值越大优先级越高）
    priority: int = 0

    # 规则名称
    name: str = ""

    # 是否启用
    enabled: bool = True


# ---------------------------------------------------------------------------
# 负载均衡策略
# ---------------------------------------------------------------------------

class LoadBalanceStrategy:
    """
    负载均衡策略基类。

    用于在多个目标之间选择。
    """

    def select(self, targets: list[str], message: GatewayMessage) -> str | None:
        """选择一个目标。"""
        raise NotImplementedError


class RoundRobinStrategy(LoadBalanceStrategy):
    """轮询策略。"""

    def __init__(self):
        self._index: dict[str, int] = defaultdict(int)

    def select(self, targets: list[str], message: GatewayMessage) -> str | None:
        if not targets:
            return None

        key = message.target_type or "default"
        idx = self._index[key] % len(targets)
        self._index[key] += 1
        return targets[idx]


class LeastConnectionsStrategy(LoadBalanceStrategy):
    """最少连接策略。"""

    def __init__(self):
        self._connections: dict[str, int] = defaultdict(int)

    def select(self, targets: list[str], message: GatewayMessage) -> str | None:
        if not targets:
            return None

        # 选择连接数最少的目标
        min_conn = float("inf")
        selected = targets[0]

        for target in targets:
            conn = self._connections.get(target, 0)
            if conn < min_conn:
                min_conn = conn
                selected = target

        return selected

    def increment(self, target: str) -> None:
        """增加连接计数。"""
        self._connections[target] += 1

    def decrement(self, target: str) -> None:
        """减少连接计数。"""
        self._connections[target] = max(0, self._connections[target] - 1)


class HashStrategy(LoadBalanceStrategy):
    """哈希策略（基于 session_id）。"""

    def select(self, targets: list[str], message: GatewayMessage) -> str | None:
        if not targets:
            return None

        # 基于 session_id 哈希选择，确保同一会话路由到同一目标
        key = message.session_id or message.chat_id or message.id
        idx = hash(key) % len(targets)
        return targets[idx]


# ---------------------------------------------------------------------------
# 路由表
# ---------------------------------------------------------------------------

class RoutingTable:
    """
    路由表。

    维护目标注册和路由规则。
    """

    def __init__(self):
        # 按类型分组的目标列表
        # {"agent": ["agent_1", "agent_2"], "tool": ["tool_1"]}
        self._targets: dict[str, list[str]] = defaultdict(list)

        # 目标能力
        # {"tool_1": ["bash", "file_ops"], "tool_2": ["web"]}
        self._capabilities: dict[str, list[str]] = {}

        # 目标状态
        # {"agent_1": {"load": 0.5, "connections": 3}}
        self._target_status: dict[str, dict] = {}

        # 路由规则
        self._rules: list[RouteRule] = []

    # ---- 目标管理 ----

    def register(
        self,
        target: str,
        target_type: str,
        capabilities: list[str] | None = None,
    ) -> None:
        """注册目标。"""
        if target not in self._targets[target_type]:
            self._targets[target_type].append(target)

        if capabilities:
            self._capabilities[target] = capabilities

        logger.debug(
            "Target registered: %s (type=%s, capabilities=%s)",
            target,
            target_type,
            capabilities,
        )

    def unregister(self, target: str) -> None:
        """注销目标。"""
        for target_type in self._targets:
            if target in self._targets[target_type]:
                self._targets[target_type].remove(target)

        self._capabilities.pop(target, None)
        self._target_status.pop(target, None)

    def update_status(self, target: str, status: dict) -> None:
        """更新目标状态。"""
        self._target_status[target] = status

    # ---- 目标查询 ----

    def get_targets(self, target_type: str) -> list[str]:
        """获取指定类型的所有目标。"""
        return list(self._targets.get(target_type, []))

    def get_targets_by_capability(self, capability: str) -> list[str]:
        """获取具有指定能力的目标。"""
        return [
            target
            for target, caps in self._capabilities.items()
            if capability in caps
        ]

    def get_capabilities(self, target: str) -> list[str]:
        """获取目标的能力列表。"""
        return self._capabilities.get(target, [])

    # ---- 规则管理 ----

    def add_rule(self, rule: RouteRule) -> None:
        """添加路由规则。"""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def remove_rule(self, name: str) -> None:
        """移除路由规则。"""
        self._rules = [r for r in self._rules if r.name != name]


# ---------------------------------------------------------------------------
# 消息路由器
# ---------------------------------------------------------------------------

class MessageRouter:
    """
    消息路由器。

    核心职责：
    1. 渠道消息 -> Agent（会话路由）
    2. 工具调用 -> Tool Worker（工具路由）
    3. 工具结果 -> Agent（响应路由）
    4. Agent 响应 -> 渠道（响应路由）

    不处理消息内容，只负责路由。
    """

    def __init__(
        self,
        connection_manager: ConnectionManager,
    ):
        self._manager = connection_manager
        self._routing_table = RoutingTable()

        # 负载均衡策略
        self._strategies: dict[str, LoadBalanceStrategy] = {
            "round_robin": RoundRobinStrategy(),
            "least_connections": LeastConnectionsStrategy(),
            "hash": HashStrategy(),
        }
        self._default_strategy = "hash"

        # 会话-目标映射（确保同一会话路由到同一 Agent）
        self._session_targets: dict[str, str] = {}

        # 待处理调用（用于请求-响应匹配）
        self._pending_calls: dict[str, asyncio.Future] = {}

        # 回调
        self._route_callbacks: list[Callable] = []

    # ---- 目标注册 ----

    def register_agent(self, agent_id: str, capabilities: list[str] | None = None) -> None:
        """注册 Agent 目标。"""
        self._routing_table.register(agent_id, "agent", capabilities)

    def register_tool(self, tool_id: str, tools: list[str]) -> None:
        """注册 Tool Worker 目标。"""
        self._routing_table.register(tool_id, "tool", tools)

    def unregister_target(self, target: str) -> None:
        """注销目标。"""
        self._routing_table.unregister(target)

        # 清除会话映射
        for session_id, target_id in list(self._session_targets.items()):
            if target_id == target:
                del self._session_targets[session_id]

    # ---- 路由接口 ----

    async def route(self, message: GatewayMessage) -> bool:
        """
        路由消息。

        根据消息类型决定路由目标。

        Args:
            message: 消息对象

        Returns:
            bool: 是否成功路由
        """
        # 根据消息类型路由
        if message.type == MessageType.CHANNEL_MESSAGE:
            return await self._route_to_agent(message)

        elif message.type == MessageType.TOOL_CALL:
            return await self._route_to_tool(message)

        elif message.type == MessageType.TOOL_RESULT:
            return await self._route_tool_result(message)

        elif message.type == MessageType.AGENT_RESPONSE:
            return await self._route_to_channel(message)

        elif message.type == MessageType.STREAM_CHUNK:
            return await self._route_to_channel(message)

        elif message.type == MessageType.STREAM_END:
            return await self._route_to_channel(message)

        else:
            logger.debug("No routing for message type: %s", message.type)
            return False

    async def _route_to_agent(self, message: GatewayMessage) -> bool:
        """
        将渠道消息路由到 Agent。

        使用会话ID确保同一会话路由到同一 Agent。
        """
        # 生成/获取会话ID
        session_id = message.session_id or new_session_id(
            message.channel_type,
            message.chat_id,
        )

        logger.info(f"[Router] Routing to agent: session_id={session_id}, "
                    f"connection_id={message.connection_id}, chat_id={message.chat_id}")

        # 检查是否已有目标
        target = self._session_targets.get(session_id)

        if not target:
            # 选择 Agent
            agents = self._routing_table.get_targets("agent")
            if not agents:
                logger.warning("No agent available for routing")
                return False

            strategy = self._strategies[self._default_strategy]
            target = strategy.select(agents, message)

            if target:
                self._session_targets[session_id] = target
                logger.debug(
                    "Session routed: %s -> %s",
                    session_id,
                    target,
                )

        if not target:
            return False

        # 设置消息目标
        routed_message = GatewayMessage(
            id=message.id,
            type=message.type,
            timestamp=message.timestamp,
            source=message.source,
            source_type=message.source_type,
            target=target,
            target_type="agent",
            channel_type=message.channel_type,
            channel_account=message.channel_account,
            connection_id=message.connection_id,
            chat_id=message.chat_id,
            sender_id=message.sender_id,
            content=message.content,
            msg_type=message.msg_type,
            session_id=session_id,
            correlation_id=message.correlation_id,
            reply_to=message.reply_to,
            thread_id=message.thread_id,
            media_urls=message.media_urls,
            metadata={
                **message.metadata,
                "original_connection_id": message.connection_id,
            },
        )

        # 发送到 Agent
        # 注意：不在这里绑定 session，让 AgentWorker 管理自己的 session
        # 路由时使用 original_connection_id metadata
        return await self._manager.send_to_source(target, routed_message)

    async def _route_to_tool(self, message: GatewayMessage) -> bool:
        """
        将工具调用路由到 Tool Worker。

        根据工具名称选择具有对应能力的 Worker。
        """
        tool_name = message.metadata.get("tool_name", "")
        if not tool_name:
            logger.warning("Tool call without tool_name")
            return False

        # 查找具有该工具能力的 Worker
        workers = self._routing_table.get_targets_by_capability(tool_name)

        if not workers:
            # 尝试通用 Worker
            workers = self._routing_table.get_targets("tool")

        if not workers:
            logger.warning("No tool worker available for: %s", tool_name)
            return False

        # 选择 Worker
        strategy = self._strategies["hash"]
        target = strategy.select(workers, message)

        if not target:
            return False

        # 设置消息目标
        routed_message = GatewayMessage(
            id=message.id,
            type=message.type,
            timestamp=message.timestamp,
            source=message.source,
            source_type=message.source_type,
            target=target,
            target_type="tool",
            channel_type=message.channel_type,
            session_id=message.session_id,
            correlation_id=message.correlation_id,
            content=message.content,
            metadata={
                **message.metadata,
                "original_source": message.source,
            },
        )

        # 注册待处理调用
        if message.correlation_id:
            loop = asyncio.get_event_loop()
            future = loop.create_future()
            self._pending_calls[message.correlation_id] = future

        # 发送到 Tool Worker
        return await self._manager.send_to_source(target, routed_message)

    async def _route_tool_result(self, message: GatewayMessage) -> bool:
        """
        将工具结果路由回 Agent。

        使用关联ID找到原始调用者。
        """
        correlation_id = message.correlation_id

        # 检查是否有待处理的调用
        if correlation_id and correlation_id in self._pending_calls:
            future = self._pending_calls.pop(correlation_id)
            if not future.done():
                future.set_result(message)
            return True

        # 路由到原始来源（Agent）
        original_source = message.metadata.get("original_source")
        if original_source:
            return await self._manager.send_to_source(original_source, message)

        # 尝试通过 session_id 找到 Agent
        session_id = message.session_id
        if session_id:
            target = self._session_targets.get(session_id)
            if target:
                return await self._manager.send_to_source(target, message)

        logger.warning("Cannot route tool result: no target found")
        return False

    async def _route_to_channel(self, message: GatewayMessage) -> bool:
        """
        将 Agent 响应路由到渠道。

        通过会话ID找到原始渠道连接。
        """
        session_id = message.session_id

        logger.info(f"[Router] Routing to channel: session_id={session_id}, message_type={message.type.value}")

        if not session_id:
            # 尝试从消息元数据获取
            session_id = message.metadata.get("session_id")
            logger.warning(f"[Router] No session_id in message, trying metadata: {session_id}")

        # 优先尝试 original_connection_id（来自 Router 的路由信息）
        original_connection_id = message.metadata.get("original_connection_id", "")
        if original_connection_id:
            logger.info(f"[Router] Using original_connection_id: {original_connection_id}")

            # 检查是否是渠道连接 (格式: channel:{account_id})
            if original_connection_id.startswith("channel:"):
                account_id = original_connection_id[8:]  # 移除 "channel:" 前缀
                logger.info(f"[Router] Routing to channel adapter: account_id={account_id}")

                try:
                    # 导入 Gateway 的全局实例
                    from .gateway import get_gateway
                    gateway = get_gateway()

                    if gateway and hasattr(gateway, 'account_manager'):
                        account_info = gateway.account_manager.getAccount(account_id)
                        if account_info and account_info.adapter:
                            # 直接调用适配器的发送方法
                            if message.chat_id:
                                # 提取必要的参数
                                sender_id = message.sender_id or message.metadata.get("sender_id", "")
                                chat_type = message.metadata.get("chat_type", "")

                                result = await account_info.adapter.send_message(
                                    chat_id=message.chat_id,
                                    content=message.content,
                                    msg_type=message.msg_type or "text",
                                    sender_id=sender_id,  # 用于钉钉单聊
                                    chat_type=chat_type,  # 用于区分单聊/群聊
                                )
                                logger.info(f"[Router] Sent message via adapter: {result}")
                                return True
                            else:
                                logger.warning("[Router] No chat_id in message")
                        else:
                            logger.warning(f"[Router] Account not found: {account_id}")
                    else:
                        logger.warning("[Router] Gateway or account_manager not available")

                except Exception as e:
                    logger.error(f"[Router] Error sending to channel adapter: {e}", exc_info=True)

        if session_id:
            # 获取会话关联的连接
            connection_id = self._manager.get_connection_for_session(session_id)
            logger.info(f"[Router] Found connection {connection_id} for session {session_id}")

            if connection_id:
                # 检查是否是渠道连接 (格式: channel:{account_id})
                if connection_id.startswith("channel:"):
                    account_id = connection_id[8:]  # 移除 "channel:" 前缀
                    logger.info(f"[Router] Routing to channel adapter: account_id={account_id}")

                    # 通过 Gateway 的 account_manager 发送
                    # 需要从 message 获取必要的字段
                    try:
                        # 导入 Gateway 的全局实例
                        from .gateway import get_gateway
                        gateway = get_gateway()

                        if gateway and hasattr(gateway, 'account_manager'):
                            account_info = gateway.account_manager.getAccount(account_id)
                            if account_info and account_info.adapter:
                                # 直接调用适配器的发送方法
                                if message.chat_id:
                                    # 提取必要的参数
                                    sender_id = message.sender_id or message.metadata.get("sender_id", "")
                                    chat_type = message.metadata.get("chat_type", "")

                                    result = await account_info.adapter.send_message(
                                        chat_id=message.chat_id,
                                        content=message.content,
                                        msg_type=message.msg_type or "text",
                                        sender_id=sender_id,  # 用于钉钉单聊
                                        chat_type=chat_type,  # 用于区分单聊/群聊
                                    )
                                    logger.info(f"[Router] Sent message via adapter: {result}")
                                    return True
                                else:
                                    logger.warning("[Router] No chat_id in message")
                            else:
                                logger.warning(f"[Router] Account not found: {account_id}")
                        else:
                            logger.warning("[Router] Gateway or account_manager not available")

                    except Exception as e:
                        logger.error(f"[Router] Error sending to channel adapter: {e}", exc_info=True)

                else:
                    # WebSocket 连接，使用原来的方式
                    routed_message = GatewayMessage(
                        id=message.id,
                        type=message.type,
                        timestamp=message.timestamp,
                        source=message.source,
                        source_type=message.source_type,
                        connection_id=connection_id,
                        channel_type=message.channel_type,
                        channel_account=message.channel_account,
                        chat_id=message.chat_id,
                        content=message.content,
                        msg_type=message.msg_type,
                        session_id=session_id,
                        correlation_id=message.correlation_id,
                        metadata=message.metadata,
                    )

                    logger.info(f"[Router] Sending to WebSocket connection {connection_id}")
                    return await self._manager.send(connection_id, routed_message)
            else:
                logger.warning(f"[Router] No connection found for session {session_id}")

        # 尝试通过 connection_id 直接发送
        if message.connection_id:
            logger.info(f"[Router] Trying direct send via connection_id: {message.connection_id}")
            return await self._manager.send(message.connection_id, message)

        logger.warning("Cannot route to channel: no connection found")
        return False

    # ---- 调用接口 ----

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict,
        session_id: str,
        timeout_ms: int = 30000,
    ) -> GatewayMessage | None:
        """
        调用工具并等待结果。

        这是一个便捷方法，用于 Agent 同步调用工具。

        Args:
            tool_name: 工具名称
            arguments: 工具参数
            session_id: 会话ID
            timeout_ms: 超时时间

        Returns:
            工具结果消息，超时返回 None
        """
        import uuid

        correlation_id = str(uuid.uuid4())[:8]

        # 创建调用消息
        call_message = GatewayMessage(
            type=MessageType.TOOL_CALL,
            source="router",
            source_type="router",
            session_id=session_id,
            correlation_id=correlation_id,
            metadata={
                "tool_name": tool_name,
                "arguments": arguments,
            },
        )

        # 注册 Future
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self._pending_calls[correlation_id] = future

        # 路由调用
        routed = await self._route_to_tool(call_message)

        if not routed:
            self._pending_calls.pop(correlation_id, None)
            return None

        # 等待结果
        try:
            result = await asyncio.wait_for(
                future,
                timeout=timeout_ms / 1000,
            )
            return result
        except asyncio.TimeoutError:
            self._pending_calls.pop(correlation_id, None)
            logger.warning("Tool call timeout: %s", tool_name)
            return None

    # ---- 状态查询 ----

    def get_session_target(self, session_id: str) -> str | None:
        """获取会话关联的目标。"""
        return self._session_targets.get(session_id)

    def get_routing_stats(self) -> dict:
        """获取路由统计信息。"""
        return {
            "agents": len(self._routing_table.get_targets("agent")),
            "tools": len(self._routing_table.get_targets("tool")),
            "sessions": len(self._session_targets),
            "pending_calls": len(self._pending_calls),
        }

    def set_strategy(self, strategy_name: str) -> None:
        """设置默认负载均衡策略。"""
        if strategy_name in self._strategies:
            self._default_strategy = strategy_name

    # ---- 回调注册 ----

    def on_route(self, callback: Callable[[GatewayMessage, str], None]) -> None:
        """注册路由回调。"""
        self._route_callbacks.append(callback)
