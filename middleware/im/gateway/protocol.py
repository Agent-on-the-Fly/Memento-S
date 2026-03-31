"""
Gateway 协议定义。

所有消息类型、连接模式、接口协议在此定义。
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# 协议版本
# ---------------------------------------------------------------------------

PROTOCOL_VERSION = 3

PROTOCOL_VERSION_HISTORY = {
    1: "基础消息收发",
    2: "流式响应、消息编辑",
    3: "生命周期钩子、权限域、连接模式配置、Webhook 支持、分布式部署",
}


# ---------------------------------------------------------------------------
# 枚举定义
# ---------------------------------------------------------------------------

class ChannelType(str, Enum):
    """渠道类型。"""
    TELEGRAM = "telegram"
    DISCORD = "discord"
    WHATSAPP = "whatsapp"
    FEISHU = "feishu"
    DINGTALK = "dingtalk"
    WECOM = "wecom"
    WECHAT = "wechat"  # WeChat Personal via iLink API
    SLACK = "slack"
    WEBSOCKET = "websocket"
    HTTP_API = "http_api"


class ConnectionMode(Enum):
    """连接模式。"""
    POLLING = "polling"        # 轮询模式：适配器主动轮询 API
    WEBHOOK = "webhook"        # Webhook 模式：接收 HTTP 回调
    WEBSOCKET = "websocket"    # WebSocket 模式：双向长连接
    HYBRID = "hybrid"          # 混合模式：polling + webhook


class ConnectionType(str, Enum):
    """连接类型。"""
    CHANNEL = "channel"        # 渠道连接
    AGENT = "agent"            # Agent 连接
    TOOL = "tool"              # Tool Worker 连接
    OPERATOR = "operator"      # Operator 控制台连接


class ConnectionState(Enum):
    """连接状态。"""
    CONNECTING = auto()
    CONNECTED = auto()
    AUTHENTICATING = auto()
    AUTHENTICATED = auto()
    READY = auto()
    IDLE = auto()
    BUSY = auto()
    DISCONNECTING = auto()
    DISCONNECTED = auto()
    ERROR = auto()


class PermissionDomain(str, Enum):
    """权限域。"""
    OPERATOR = "operator"      # 运维管理员 - 完全控制
    NODE = "node"              # 节点服务 - 受限执行
    VIEWER = "viewer"          # 观察者 - 只读


class MessageType(Enum):
    """消息类型。"""
    # ---- 连接管理 ----
    CONNECT = "connect"                # 连接请求
    CONNECT_ACK = "connect_ack"        # 连接确认
    DISCONNECT = "disconnect"          # 断开连接
    PING = "ping"                      # 心跳请求
    PONG = "pong"                      # 心跳响应

    # ---- 渠道消息 ----
    CHANNEL_MESSAGE = "channel_message"    # 渠道消息（入站）
    CHANNEL_EVENT = "channel_event"        # 渠道事件

    # ---- Agent 消息 ----
    AGENT_RESPONSE = "agent_response"      # Agent 响应
    STREAM_CHUNK = "stream_chunk"          # 流式片段
    STREAM_END = "stream_end"              # 流式结束

    # ---- 工具调用 ----
    TOOL_CALL = "tool_call"                # 工具调用请求
    TOOL_RESULT = "tool_result"            # 工具调用结果

    # ---- 系统消息 ----
    EVENT = "event"                        # 事件通知
    ERROR = "error"                        # 错误消息
    ACK = "ack"                            # 确认消息


# ---------------------------------------------------------------------------
# 消息模型
# ---------------------------------------------------------------------------

@dataclass
class GatewayMessage:
    """
    Gateway 统一消息格式。

    所有通过 Gateway 传输的消息都使用此格式，
    包括渠道消息、Agent 响应、工具调用等。

    Attributes:
        id: 消息唯一标识
        type: 消息类型
        timestamp: 时间戳
        source: 来源标识
        source_type: 来源类型
        target: 目标标识
        target_type: 目标类型
        channel_type: 渠道类型
        channel_account: 渠道账户ID
        connection_id: Gateway 连接ID
        chat_id: 会话/群组ID
        sender_id: 发送者ID
        content: 消息内容
        msg_type: 消息类型
        session_id: Agent 会话ID（关联消息上下文）
        correlation_id: 关联ID（用于请求-响应匹配）
        reply_to: 回复的消息ID
        thread_id: 话题/线程ID
        media_urls: 媒体URL列表
        metadata: 元数据
    """

    # 消息标识
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: MessageType = MessageType.CHANNEL_MESSAGE
    timestamp: float = field(default_factory=time.time)

    # 来源/目标
    source: str = ""
    source_type: str = ""              # channel / agent / tool / operator
    target: str = ""
    target_type: str = ""

    # 渠道信息
    channel_type: ChannelType = ChannelType.WEBSOCKET
    channel_account: str = ""
    connection_id: str = ""

    # 消息内容
    chat_id: str = ""
    sender_id: str = ""
    content: str = ""
    msg_type: str = "text"             # text / image / file / interactive

    # 会话上下文
    session_id: str = ""               # Agent 会话ID，关联消息上下文
    correlation_id: str = ""           # 请求-响应匹配

    # 元数据
    reply_to: str = ""
    thread_id: str = ""
    media_urls: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> str:
        """序列化为 JSON 字符串。"""
        return json.dumps({
            "id": self.id,
            "type": self.type.value,
            "timestamp": self.timestamp,
            "source": self.source,
            "source_type": self.source_type,
            "target": self.target,
            "target_type": self.target_type,
            "channel_type": self.channel_type.value,
            "channel_account": self.channel_account,
            "connection_id": self.connection_id,
            "chat_id": self.chat_id,
            "sender_id": self.sender_id,
            "content": self.content,
            "msg_type": self.msg_type,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "thread_id": self.thread_id,
            "media_urls": self.media_urls,
            "metadata": self.metadata,
        }, ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str | dict) -> "GatewayMessage":
        """从 JSON 反序列化。"""
        if isinstance(data, str):
            obj = json.loads(data)
        else:
            obj = data

        return cls(
            id=obj.get("id", str(uuid.uuid4())[:8]),
            type=MessageType(obj.get("type", "channel_message")),
            timestamp=obj.get("timestamp", time.time()),
            source=obj.get("source", ""),
            source_type=obj.get("source_type", ""),
            target=obj.get("target", ""),
            target_type=obj.get("target_type", ""),
            channel_type=ChannelType(obj.get("channel_type", "websocket")),
            channel_account=obj.get("channel_account", ""),
            connection_id=obj.get("connection_id", ""),
            chat_id=obj.get("chat_id", ""),
            sender_id=obj.get("sender_id", ""),
            content=obj.get("content", ""),
            msg_type=obj.get("msg_type", "text"),
            session_id=obj.get("session_id", ""),
            correlation_id=obj.get("correlation_id", ""),
            reply_to=obj.get("reply_to", ""),
            thread_id=obj.get("thread_id", ""),
            media_urls=obj.get("media_urls", []),
            metadata=obj.get("metadata", {}),
        )

    def to_agent_input(self) -> dict:
        """
        转换为 Agent 输入格式。

        用于将渠道消息转换为 Agent 可处理的格式。
        """
        return {
            "session_id": self.session_id or f"{self.channel_type.value}:{self.chat_id}",
            "user_content": self.content,
            "metadata": {
                "channel_type": self.channel_type.value,
                "channel_account": self.channel_account,
                "chat_id": self.chat_id,
                "sender_id": self.sender_id,
                "message_id": self.id,
                "connection_id": self.connection_id,
                "reply_to": self.reply_to,
                **self.metadata,
            },
        }

    def create_response(
        self,
        content: str,
        msg_type: MessageType = MessageType.AGENT_RESPONSE,
    ) -> "GatewayMessage":
        """
        创建响应消息。

        自动设置目标、关联ID等信息。
        """
        return GatewayMessage(
            type=msg_type,
            source="gateway",
            source_type="gateway",
            target=self.source,
            target_type=self.source_type,
            channel_type=self.channel_type,
            channel_account=self.channel_account,
            connection_id=self.connection_id,
            chat_id=self.chat_id,
            content=content,
            session_id=self.session_id,
            correlation_id=self.id,
        )


# ---------------------------------------------------------------------------
# 连接配置
# ---------------------------------------------------------------------------

@dataclass
class ConnectionConfig:
    """
    连接配置。

    包含连接的所有配置信息。
    """
    connection_id: str
    connection_type: ConnectionType

    # 渠道信息（仅 channel 类型）
    channel_type: ChannelType | None = None
    channel_account: str = ""

    # 连接模式（仅 channel 类型）
    mode: ConnectionMode = ConnectionMode.POLLING

    # Webhook 配置
    webhook_path: str = ""
    webhook_events: list[str] = field(default_factory=list)

    # 权限
    permission_domain: PermissionDomain = PermissionDomain.NODE
    allowed_tools: list[str] = field(default_factory=list)

    # 心跳
    heartbeat_interval_ms: int = 30000
    heartbeat_timeout_ms: int = 10000

    # 重连
    auto_reconnect: bool = True
    reconnect_delay_ms: int = 1000
    max_reconnect_attempts: int = 5

    # 元数据
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 账户配置
# ---------------------------------------------------------------------------

@dataclass
class AccountConfig:
    """
    账户配置。

    用于启动渠道账户。
    """
    account_id: str
    channel_type: ChannelType
    credentials: dict

    # 连接模式
    mode: ConnectionMode = ConnectionMode.POLLING

    # Webhook 配置
    webhook_config: dict = field(default_factory=dict)
    polling_config: dict = field(default_factory=dict)

    # 权限
    permission_domain: PermissionDomain = PermissionDomain.NODE
    allowed_tools: list[str] = field(default_factory=list)

    # 元数据
    metadata: dict = field(default_factory=dict)

    def to_connection_config(self) -> ConnectionConfig:
        """转换为连接配置。"""
        return ConnectionConfig(
            connection_id=f"channel:{self.account_id}",
            connection_type=ConnectionType.CHANNEL,
            channel_type=self.channel_type,
            channel_account=self.account_id,
            mode=self.mode,
            webhook_path=f"/webhook/{self.channel_type.value}/{self.account_id}",
            webhook_events=self.webhook_config.get("events", []),
            permission_domain=self.permission_domain,
            allowed_tools=self.allowed_tools,
            metadata={**self.credentials, **self.metadata},
        )


# ---------------------------------------------------------------------------
# 渠道适配器协议
# ---------------------------------------------------------------------------

@runtime_checkable
class ChannelAdapterProtocol(Protocol):
    """
    渠道适配器协议 v3。

    所有渠道适配器必须实现此协议。
    支持多种连接模式：polling / webhook / websocket / hybrid。

    生命周期：
    1. initialize() - 初始化适配器
    2. start() - 启动服务（根据 mode 决定行为）
    3. 运行中 - 接收消息，触发回调
    4. stop() - 停止服务
    """

    @property
    def protocol_version(self) -> int:
        """返回适配器支持的协议版本。"""
        ...

    @property
    def channel_type(self) -> ChannelType:
        """返回渠道类型。"""
        ...

    @property
    def capabilities(self) -> list[str]:
        """返回渠道能力列表。"""
        ...

    @property
    def supported_modes(self) -> list[ConnectionMode]:
        """返回支持的连接模式列表。"""
        ...

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
        ...

    async def start(self) -> None:
        """
        启动渠道服务。

        根据连接模式执行不同操作：
        - POLLING: 启动轮询循环
        - WEBHOOK: 准备接收回调（无需启动连接）
        - WEBSOCKET: 建立 WebSocket 连接
        - HYBRID: 同时启动 polling 和 webhook 处理
        """
        ...

    async def stop(self) -> None:
        """停止渠道服务。"""
        ...

    async def health_check(self) -> bool:
        """健康检查。"""
        ...

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
        ...

    # ---- Webhook 支持 ----

    async def parse_webhook(self, payload: dict) -> list[GatewayMessage]:
        """
        解析 Webhook 回调。

        将平台回调转换为 GatewayMessage 列表。

        Args:
            payload: Webhook 请求体

        Returns:
            消息列表
        """
        ...

    async def verify_webhook(
        self,
        signature: str,
        body: bytes,
        headers: dict = None,
    ) -> bool:
        """
        验证 Webhook 签名。

        Args:
            signature: 签名字符串
            body: 请求体
            headers: 请求头

        Returns:
            是否验证通过
        """
        ...

    # ---- 事件回调 ----

    def on_message(self, callback: Callable[[GatewayMessage], None]) -> None:
        """
        注册消息回调。

        当收到渠道消息时触发。
        """
        ...

    def on_event(self, callback: Callable[[dict], None]) -> None:
        """
        注册事件回调。

        当发生渠道事件时触发。
        """
        ...


# ---------------------------------------------------------------------------
# 渠道能力
# ---------------------------------------------------------------------------

class ChannelCapability(str, Enum):
    """渠道能力枚举。"""
    TEXT = "text"                      # 文本消息
    RICH_TEXT = "rich_text"            # 富文本/Markdown
    IMAGE = "image"                    # 图片消息
    VIDEO = "video"                    # 视频消息
    AUDIO = "audio"                    # 音频消息
    FILE = "file"                      # 文件消息
    INTERACTIVE = "interactive"        # 交互式卡片
    LOCATION = "location"              # 位置消息
    CONTACT = "contact"                # 联系人消息
    REPLY = "reply"                    # 回复消息
    EDIT = "edit"                      # 编辑消息
    DELETE = "delete"                  # 删除消息
    REACTION = "reaction"              # 消息反应
    THREAD = "thread"                  # 话题/线程
    TYPING = "typing"                  # 打字提示
    STREAMING = "streaming"            # 流式响应


# ---------------------------------------------------------------------------
# 消息类型常量
# ---------------------------------------------------------------------------

# 用于 msg_type 字段
MSG_TYPE_TEXT = "text"
MSG_TYPE_IMAGE = "image"
MSG_TYPE_VIDEO = "video"
MSG_TYPE_AUDIO = "audio"
MSG_TYPE_FILE = "file"
MSG_TYPE_INTERACTIVE = "interactive"
MSG_TYPE_LOCATION = "location"
MSG_TYPE_CONTACT = "contact"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def new_message_id() -> str:
    """生成新的消息ID。"""
    return str(uuid.uuid4())[:8]


def new_session_id(channel_type: ChannelType | str, chat_id: str) -> str:
    """
    生成会话ID。

    Args:
        channel_type: 渠道类型
        chat_id: 会话/群组ID

    Returns:
        格式为 "{channel_type}:{chat_id}" 的会话ID
    """
    if isinstance(channel_type, ChannelType):
        channel = channel_type.value
    else:
        channel = channel_type
    return f"{channel}:{chat_id}"
