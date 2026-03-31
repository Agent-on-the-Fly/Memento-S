# Gateway 网关层设计方案

## 概述

Gateway 是 Memento-S 多渠道消息分发网关，负责统一管理各消息平台的消息接收与分发。Gateway 作为消息路由中心，连接渠道适配器、Agent 和工具执行器，支持分布式部署。

## 架构设计

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              Gateway Layer (网关层)                                  │
│                                                                                     │
│  职责：消息路由 + 消息分发 (支持分布式部署)                                           │
│                                                                                     │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │                              Gateway                                           │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │  │
│  │  │ Account     │  │ Connection  │  │ Message     │  │ WebhookServer       │  │  │
│  │  │ Manager     │  │ Manager     │  │ Router      │  │ (HTTP 端点)         │  │  │
│  │  │             │  │             │  │             │  │                     │  │  │
│  │  │startAccount │  │register()   │  │route()      │  │/webhook/{channel}/  │  │  │
│  │  │stopAccount  │  │broadcast()  │  │send()       │  │{account}            │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘  │  │
│  │                                                                                │  │
│  │  ┌─────────────────────────────────────────────────────────────────────────┐  │  │
│  │  │ ChannelRegistry (插件注册表)                                             │  │  │
│  │  └─────────────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                             │
│                                        │ WebSocket 长连接                            │
│                                        ▼                                             │
└────────────────────────────────────────┼─────────────────────────────────────────────┘
                                         │
         ┌───────────────────────────────┼───────────────────────────────┐
         │                               │                               │
    ┌────▼────┐                    ┌─────▼─────┐                  ┌─────▼─────┐
    │ Channel │                    │   Agent   │                  │   Tool    │
    │ Adapter │                    │ Connection│                  │ Connection│
    │ (渠道)  │                    │  (Agent)  │                  │ (工具执行)│
    └─────────┘                    └───────────┘                  └───────────┘
         │                               │                               │
         │    channel_message            │                               │
         │◄──────────────────────────────┤                               │
         │                               │                               │
         │                               │  tool_call                    │
         │                               ├──────────────────────────────►│
         │                               │                               │
         │                               │  tool_result                  │
         │                               │◄──────────────────────────────┤
         │                               │                               │
         │  agent_response               │                               │
         │◄──────────────────────────────┤                               │
         │                               │                               │
```

## 核心设计理念

### 1. 消息路由中心

Gateway 作为消息路由中心，不持有 Agent 或 Tool 实例，而是通过 WebSocket 连接进行消息分发：

- **Channel Adapter**: 渠道适配器，接收用户消息
- **Agent Connection**: Agent 连接，处理消息、生成响应
- **Tool Connection**: 工具执行器连接，执行工具调用

### 2. 统一连接通道

所有消息类型在同一连接通道中传输，通过 `MessageType` 区分：

```
消息流向：
channel_message → tool_call → tool_result → ... → agent_response

协议层消息类型：
- CHANNEL_MESSAGE: 渠道消息（用户输入）
- TOOL_CALL: 工具调用请求
- TOOL_RESULT: 工具执行结果
- AGENT_RESPONSE: Agent 响应（最终输出）
- STREAM_CHUNK: 流式响应片段
```

### 3. 分布式部署支持

Gateway 支持 Agent 和 Tool Worker 独立部署：

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Gateway Instance                             │
│                                                                      │
│  WebSocket Server: ws://gateway:8765                                │
│  Webhook Server:  http://gateway:18080/webhook/{channel}/{account}  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
    ┌────▼────┐           ┌─────▼─────┐         ┌─────▼─────┐
    │ Agent 1 │           │ Agent 2   │         │ Agent N   │
    │ Instance│           │ Instance  │         │ Instance  │
    └─────────┘           └───────────┘         └───────────┘
         │                      │                      │
    ┌────▼────┐           ┌─────▼─────┐         ┌─────▼─────┐
    │ Tool 1  │           │ Tool 2    │         │ Tool N    │
    │ Worker  │           │ Worker    │         │ Worker    │
    └─────────┘           └───────────┘         └───────────┘
```

## 五大关键设计点

### 1. 适配器模式

将不同平台差异抽象为统一接口，支持多种连接模式：

```python
class ConnectionMode(Enum):
    """连接模式。"""
    POLLING = "polling"        # 轮询模式
    WEBHOOK = "webhook"        # Webhook 回调模式
    WEBSOCKET = "websocket"    # WebSocket 长连接
    HYBRID = "hybrid"          # 混合模式（polling + webhook）

@runtime_checkable
class ChannelAdapterProtocol(Protocol):
    @property
    def protocol_version(self) -> int: ...
    
    @property
    def channel_type(self) -> ChannelType: ...
    
    @property
    def supported_modes(self) -> list[ConnectionMode]:
        """支持的连接模式。"""
        ...
    
    async def initialize(
        self, 
        config: ConnectionConfig,
        mode: ConnectionMode = ConnectionMode.POLLING,
    ) -> None: ...
    
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    
    async def send_message(self, chat_id: str, content: str, **kwargs) -> str: ...
    
    # Webhook 模式需要实现
    async def parse_webhook(self, payload: dict) -> list[GatewayMessage]: ...
    async def verify_webhook(self, signature: str, body: bytes) -> bool: ...
    
    def on_message(self, callback: Callable[[GatewayMessage], None]) -> None: ...
```

### 2. 插件系统

支持插件式扩展新渠道：

```python
# 方式一：装饰器注册
@register_channel(ChannelType.TELEGRAM)
class TelegramChannelAdapter:
    supported_modes = [ConnectionMode.POLLING, ConnectionMode.WEBHOOK]
    ...

# 方式二：手动注册
gateway.registerChannel(ChannelType.TELEGRAM, TelegramChannelAdapter)
```

### 3. 生命周期钩子

通过 startAccount / stopAccount 实现账户生命周期管理：

```python
# 启动账户（支持选择连接模式）
account = await gateway.startAccount(
    account_id="telegram_bot",
    channel_type=ChannelType.TELEGRAM,
    credentials={"bot_token": "..."},
    mode=ConnectionMode.POLLING,  # 或 WEBHOOK / HYBRID
    permission_domain=PermissionDomain.NODE,
)

# 停止账户
await gateway.stopAccount("telegram_bot")

# 关闭 Gateway
await gateway.shutdown()
```

### 4. 协议版本化

支持向后兼容：

```python
PROTOCOL_VERSION = 3  # 当前协议版本

class ChannelAdapterProtocol(Protocol):
    @property
    def protocol_version(self) -> int:
        """适配器声明支持的协议版本"""
        ...
```

版本历史：
- v1: 基础消息收发
- v2: 流式响应、消息编辑
- v3: 生命周期钩子、权限域、连接模式配置、Webhook 支持

### 5. 角色权限域

| 权限域 | 说明 | 权限 |
|--------|------|------|
| `operator` | 运维管理员 | 完全控制（startAccount/stopAccount/shutdown） |
| `node` | 节点服务 | 受限执行（消息收发、工具调用） |
| `viewer` | 观察者 | 只读权限 |

## 连接模式详解

### 各渠道支持的连接模式

| 渠道 | Polling | Webhook | WebSocket | 混合模式 |
|------|---------|---------|-----------|----------|
| Telegram | ✅ | ✅ | ❌ | ✅ (polling接收 + webhook事件) |
| Discord | ❌ | ❌ | ✅ | ❌ |
| Feishu | ✅ | ✅ | ❌ | ✅ |
| WhatsApp | ❌ | ✅ | ❌ | ❌ |
| DingTalk | ✅ | ✅ | ❌ | ✅ |
| Wecom | ✅ | ✅ | ❌ | ✅ |

### Polling 模式

适配器主动轮询平台 API 获取消息：

```python
@register_channel(ChannelType.TELEGRAM)
class TelegramChannelAdapter:
    supported_modes = [ConnectionMode.POLLING, ConnectionMode.WEBHOOK]
    
    async def start(self) -> None:
        if self._mode == ConnectionMode.POLLING:
            # 启动 Bot polling
            self._application.updater.start_polling()
```

### Webhook 模式

Gateway 提供统一的 Webhook 端点，适配器负责解析和验证：

```python
# Webhook 端点格式
# POST /webhook/{channel_type}/{account_id}

@register_channel(ChannelType.WHATSAPP)
class WhatsAppChannelAdapter:
    supported_modes = [ConnectionMode.WEBHOOK]
    
    async def parse_webhook(self, payload: dict) -> list[GatewayMessage]:
        """解析 Webhook 回调，返回消息列表。"""
        messages = []
        for entry in payload.get("entry", []):
            for change in entry.get("changes", []):
                # 解析 WhatsApp 消息格式
                ...
        return messages
    
    async def verify_webhook(self, signature: str, body: bytes) -> bool:
        """验证 Webhook 签名。"""
        # WhatsApp 签名验证逻辑
        ...
```

### 混合模式

同时使用 Polling 和 Webhook：

```python
# 配置示例
await gateway.startAccount(
    account_id="telegram_main",
    channel_type=ChannelType.TELEGRAM,
    credentials={"bot_token": "..."},
    mode=ConnectionMode.HYBRID,
    webhook_config={
        "enabled": True,
        "path": "/webhook/telegram/telegram_main",
        "events": ["callback_query", "inline_query"],  # Webhook 接收事件
    },
    polling_config={
        "enabled": True,
        "timeout": 30,
    },
)
```

## 统一消息格式

所有消息都转换为 `GatewayMessage`：

```python
@dataclass
class GatewayMessage:
    # 消息标识
    id: str
    type: MessageType
    timestamp: float
    
    # 来源/目标
    channel_type: ChannelType
    channel_account: str
    connection_id: str
    
    # 消息内容
    chat_id: str
    sender_id: str
    content: str
    msg_type: str
    
    # 会话上下文
    session_id: str              # 关联 Agent 会话
    correlation_id: str          # 请求-响应匹配
    
    # 元数据
    reply_to: str
    thread_id: str
    media_urls: list[str]
    metadata: dict
```

## 消息流程

### 基础消息流程

```
Channel          Gateway              Agent              Tool Worker
   │                │                    │                    │
   │ channel_message│                    │                    │
   │───────────────►│                    │                    │
   │                │                    │                    │
   │                │ channel_message    │                    │
   │                │ (route by session) │                    │
   │                ├───────────────────►│                    │
   │                │                    │                    │
   │                │                    │ process message    │
   │                │                    │ call tools         │
   │                │                    │                    │
   │                │                    │ tool_call          │
   │                │                    ├───────────────────►│
   │                │                    │                    │
   │                │                    │ tool_result        │
   │                │                    │◄───────────────────┤
   │                │                    │                    │
   │                │                    │ generate response  │
   │                │                    │                    │
   │                │ agent_response     │                    │
   │                │◄───────────────────┤                    │
   │                │                    │                    │
   │ agent_response │                    │                    │
   │◄───────────────┤                    │                    │
   │                │                    │                    │
```

### 带流式响应的消息流程

```
Channel          Gateway              Agent
   │                │                    │
   │ channel_message│                    │
   │───────────────►│                    │
   │                ├───────────────────►│
   │                │                    │
   │                │ stream_chunk #1    │
   │                │◄───────────────────┤
   │ stream_chunk   │                    │
   │◄───────────────┤                    │
   │                │                    │
   │                │ stream_chunk #2    │
   │                │◄───────────────────┤
   │ stream_chunk   │                    │
   │◄───────────────┤                    │
   │                │                    │
   │                │ agent_response     │
   │                │◄───────────────────┤
   │ agent_response │                    │
   │◄───────────────┤                    │
```

## 使用示例

### 1. 启动 Gateway

```python
from daemon.gateway.gateway import Gateway

# 创建 Gateway 实例
gateway = Gateway(
    websocket_port=8765,
    webhook_port=18080,
)

# 启动服务
await gateway.start()
```

### 2. Agent 连接 Gateway

```python
import websockets
from daemon.gateway.protocol import GatewayMessage, MessageType

async def agent_main():
    # 连接 Gateway
    ws = await websockets.connect("ws://gateway:8765")
    
    # 发送连接消息
    connect_msg = GatewayMessage(
        type=MessageType.CONNECT,
        source="agent_1",
        source_type="agent",
    )
    await ws.send(connect_msg.to_json())
    
    # 接收消息循环
    async for data in ws:
        msg = GatewayMessage.from_json(data)
        
        if msg.type == MessageType.CHANNEL_MESSAGE:
            # 处理渠道消息
            response = await process_message(msg)
            
            # 发送响应
            await ws.send(response.to_json())
        
        elif msg.type == MessageType.TOOL_RESULT:
            # 处理工具执行结果
            ...
```

### 3. Tool Worker 连接 Gateway

```python
async def tool_worker_main():
    # 连接 Gateway
    ws = await websockets.connect("ws://gateway:8765")
    
    # 发送连接消息
    connect_msg = GatewayMessage(
        type=MessageType.CONNECT,
        source="tool_worker_1",
        source_type="tool",
        metadata={"tools": ["bash", "file_ops", "web"]},
    )
    await ws.send(connect_msg.to_json())
    
    # 接收工具调用请求
    async for data in ws:
        msg = GatewayMessage.from_json(data)
        
        if msg.type == MessageType.TOOL_CALL:
            # 执行工具
            result = await execute_tool(
                msg.metadata["tool_name"],
                msg.metadata["arguments"],
            )
            
            # 返回结果
            result_msg = GatewayMessage(
                type=MessageType.TOOL_RESULT,
                correlation_id=msg.id,
                content=result,
            )
            await ws.send(result_msg.to_json())
```

### 4. 启动渠道账户

```python
# Telegram (Polling 模式)
await gateway.startAccount(
    account_id="telegram_bot",
    channel_type=ChannelType.TELEGRAM,
    credentials={"bot_token": "..."},
    mode=ConnectionMode.POLLING,
)

# WhatsApp (Webhook 模式)
await gateway.startAccount(
    account_id="whatsapp_business",
    channel_type=ChannelType.WHATSAPP,
    credentials={
        "phone_number_id": "...",
        "access_token": "...",
        "verify_token": "...",
    },
    mode=ConnectionMode.WEBHOOK,
)

# Feishu (混合模式)
await gateway.startAccount(
    account_id="feishu_app",
    channel_type=ChannelType.FEISHU,
    credentials={
        "app_id": "...",
        "app_secret": "...",
    },
    mode=ConnectionMode.HYBRID,
)
```

## Webhook 端点

Gateway 提供 Webhook HTTP 服务：

```
POST /webhook/{channel_type}/{account_id}

示例:
POST /webhook/telegram/telegram_main
POST /webhook/whatsapp/whatsapp_business
POST /webhook/feishu/feishu_app
```

适配器只需实现 `parse_webhook` 和 `verify_webhook` 方法：

```python
class TelegramChannelAdapter:
    async def parse_webhook(self, payload: dict) -> list[GatewayMessage]:
        """解析 Telegram Update。"""
        update = payload
        if "message" in update:
            msg = update["message"]
            return [GatewayMessage(
                channel_type=ChannelType.TELEGRAM,
                chat_id=str(msg["chat"]["id"]),
                sender_id=str(msg["from"]["id"]),
                content=msg.get("text", ""),
                ...
            )]
        return []
    
    async def verify_webhook(self, signature: str, body: bytes) -> bool:
        """验证 Telegram Webhook 签名。"""
        # Telegram 使用 secret_token 验证
        expected = self._secret_token
        return signature == expected
```

## 文件结构

```
daemon/gateway/
├── __init__.py              # 模块入口
├── protocol.py              # 协议定义（消息模型、枚举、接口）
├── connection_manager.py    # 连接管理（连接池、生命周期）
├── router.py                # 消息路由器（会话路由、分发）
├── webhook_server.py        # Webhook HTTP 服务
├── gateway.py               # Gateway 主类
└── channels/                # 渠道适配器目录
    ├── __init__.py
    ├── base.py              # 适配器基类
    ├── telegram.py          # Telegram 适配器
    ├── discord.py           # Discord 适配器
    ├── whatsapp.py          # WhatsApp 适配器
    ├── feishu.py            # 飞书适配器
    ├── dingtalk.py          # 钉钉适配器
    └── wecom.py             # 企业微信适配器
```

## 配置文件

```yaml
# config/gateway.yaml

# 协议版本
protocol_version: 3

# Gateway 服务配置
server:
  websocket:
    host: "0.0.0.0"
    port: 8765
  webhook:
    host: "0.0.0.0"
    port: 18080
    base_url: "https://your-domain.com"

# 账户配置
accounts:
  telegram_main:
    channel: telegram
    enabled: true
    mode: polling           # polling / webhook / hybrid
    credentials:
      bot_token: "${TELEGRAM_BOT_TOKEN}"
    permission_domain: node
    
  whatsapp_business:
    channel: whatsapp
    enabled: true
    mode: webhook           # WhatsApp 只支持 webhook
    credentials:
      phone_number_id: "${WA_PHONE_NUMBER_ID}"
      access_token: "${WA_ACCESS_TOKEN}"
      verify_token: "${WA_VERIFY_TOKEN}"
    permission_domain: node
    
  feishu_app:
    channel: feishu
    enabled: true
    mode: hybrid            # 混合模式
    credentials:
      app_id: "${FEISHU_APP_ID}"
      app_secret: "${FEISHU_APP_SECRET}"
    permission_domain: node

# 连接配置
connection:
  heartbeat_interval_ms: 30000
  heartbeat_timeout_ms: 10000
  auto_reconnect: true
  reconnect_delay_ms: 1000
  max_reconnect_attempts: 5

# 权限配置
permissions:
  operator:
    - account:start
    - account:stop
    - gateway:shutdown
  node:
    - message:send
    - message:receive
    - tool:call
    - tool:execute
  viewer:
    - message:view
```

## 扩展新渠道

1. 创建适配器文件 `daemon/gateway/channels/new_channel.py`：

```python
from daemon.gateway.channels.base import BaseChannelAdapter, register_channel
from daemon.gateway.protocol import (
    ChannelType,
    ConnectionConfig,
    ConnectionMode,
    GatewayMessage,
    MessageType,
    PROTOCOL_VERSION,
)

@register_channel(ChannelType.NEW_CHANNEL)
class NewChannelAdapter(BaseChannelAdapter):
    """新渠道适配器。"""
    
    protocol_version = PROTOCOL_VERSION
    channel_type = ChannelType.NEW_CHANNEL
    supported_modes = [ConnectionMode.WEBHOOK]  # 只支持 Webhook
    capabilities = ["text", "image"]
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
    
    async def initialize(
        self, 
        config: ConnectionConfig,
        mode: ConnectionMode = ConnectionMode.WEBHOOK,
    ) -> None:
        self._mode = mode
        # 初始化逻辑
    
    async def start(self) -> None:
        # Webhook 模式无需启动连接
        if self._mode == ConnectionMode.POLLING:
            # 启动轮询
            pass
    
    async def stop(self) -> None:
        pass
    
    async def send_message(
        self, 
        chat_id: str, 
        content: str, 
        msg_type: str = "text",
        **kwargs,
    ) -> str:
        # 通过 API 发送消息
        response = await self._api.send(chat_id, content)
        return response["message_id"]
    
    async def parse_webhook(self, payload: dict) -> list[GatewayMessage]:
        """解析 Webhook 回调。"""
        messages = []
        for event in payload.get("events", []):
            messages.append(GatewayMessage(
                channel_type=self.channel_type,
                chat_id=event["chat_id"],
                sender_id=event["sender_id"],
                content=event["content"],
                ...
            ))
        return messages
    
    async def verify_webhook(self, signature: str, body: bytes) -> bool:
        """验证签名。"""
        import hmac
        import hashlib
        expected = hmac.new(
            self.api_secret.encode(),
            body,
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(signature, expected)
```

2. 在配置中添加账户：

```yaml
accounts:
  new_channel_main:
    channel: new_channel
    enabled: true
    mode: webhook
    credentials:
      api_key: "${NEW_CHANNEL_API_KEY}"
      api_secret: "${NEW_CHANNEL_API_SECRET}"
```

## 总结

| 特性 | 说明 |
|------|------|
| 消息路由中心 | Gateway 作为路由中心，不持有 Agent/Tool 实例 |
| 统一连接通道 | 所有消息在同一通道，通过 MessageType 区分 |
| 分布式部署 | Agent 和 Tool Worker 可独立部署扩展 |
| 连接模式配置 | 支持 polling / webhook / websocket / hybrid |
| Webhook 统一入口 | Gateway 提供 HTTP 端点，适配器负责解析验证 |
| 会话上下文关联 | session_id 关联消息与 Agent 会话 |
| 插件扩展 | 新渠道只需实现 ChannelAdapterProtocol |
| 协议版本化 | PROTOCOL_VERSION = 3，支持向后兼容 |
| 权限隔离 | operator / node / viewer 三级权限域 |
| 生命周期控制 | startAccount / stopAccount 钩子 |
