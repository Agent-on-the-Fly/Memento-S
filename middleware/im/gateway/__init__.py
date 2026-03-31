"""
Gateway 网关模块。

提供统一的消息接收和分发能力。
"""

from .protocol import (
    # 枚举
    ChannelType,
    ConnectionMode,
    ConnectionType,
    ConnectionState,
    PermissionDomain,
    MessageType,
    ChannelCapability,
    # 消息
    GatewayMessage,
    # 配置
    ConnectionConfig,
    AccountConfig,
    # 协议
    ChannelAdapterProtocol,
    # 版本
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_HISTORY,
    # 工具函数
    new_message_id,
    new_session_id,
)
from .connection_manager import (
    ConnectionManager,
    ConnectionInfo,
    ConnectionPool,
    SessionRouter,
)
from .router import (
    MessageRouter,
    RoutingTable,
    RouteRule,
    LoadBalanceStrategy,
    RoundRobinStrategy,
    LeastConnectionsStrategy,
    HashStrategy,
)
from .webhook_server import (
    WebhookServer,
    WebhookHandler,
)
from .gateway import (
    Gateway,
    ChannelRegistry,
    AccountManager,
    AccountInfo,
    register_channel,
    get_gateway,
    set_gateway,
)

__all__ = [
    # 协议
    "ChannelType",
    "ConnectionMode",
    "ConnectionType",
    "ConnectionState",
    "PermissionDomain",
    "MessageType",
    "ChannelCapability",
    "PROTOCOL_VERSION",
    "PROTOCOL_VERSION_HISTORY",
    "ChannelAdapterProtocol",
    # 消息
    "GatewayMessage",
    "ConnectionConfig",
    "AccountConfig",
    # 工具函数
    "new_message_id",
    "new_session_id",
    # 连接管理
    "ConnectionManager",
    "ConnectionInfo",
    "ConnectionPool",
    "SessionRouter",
    # 路由
    "MessageRouter",
    "RoutingTable",
    "RouteRule",
    "LoadBalanceStrategy",
    "RoundRobinStrategy",
    "LeastConnectionsStrategy",
    "HashStrategy",
    # Webhook
    "WebhookServer",
    "WebhookHandler",
    # Gateway
    "Gateway",
    "ChannelRegistry",
    "AccountManager",
    "AccountInfo",
    "register_channel",
    "get_gateway",
    "set_gateway",
]
