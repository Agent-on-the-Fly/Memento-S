# Memento-S 架构文档

## 一、项目概述

Memento-S 是一个 Python agent runtime。桌面界面使用 Flet，并在同一 Python
进程中直接调用 agent、配置、存储和技能服务。CLI、HTTP 服务和 IM gateway
复用相同的核心与中间件层。

## 二、整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Interface Layer                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Flet desktop GUI (`gui/app.py`)                            │   │
│  │  Typer CLI (`cli/main.py`)                                  │   │
│  │  HTTP/SSE service and IM gateway (`server/`, `middleware/`) │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                   Direct Python calls / events                      │
│                              ▼                                      │
├─────────────────────────────────────────────────────────────────────┤
│                     核心逻辑层 (Core)                               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  MementoSAgent          │ SkillGateway                        │  │
│  │  - Agent 核心逻辑       │ - Skill 系统管理                    │  │
│  │  - 多阶段执行流程       │ - 技能发现/检索/执行                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Protocol Layer (AG-UI Events)                               │  │
│  │  - RunEmitter, Event Pipeline                                │  │
│  │  - 事件流式传输 (用于流式响应)                                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                      中间件层 (Middleware)                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐   │
│  │ LLM Client  │ │   Config    │ │  Storage    │ │    IM      │   │
│  │  (litellm)  │ │  Manager    │ │  (SQLite)   │ │  Gateway   │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                      共享层 (Shared)                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  ChatManager (Session/Conversation 管理)                      │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## 三、Flet GUI 调用的运行时组件

### 3.1 GUI 直接导入的运行时模块

| 前端调用 | 后端类/模块 | 功能说明 |
|---------|------------|---------|
| `from core.memento_s import MementoSAgent` | **MementoSAgent** | Agent 核心类，处理用户消息、意图识别、计划生成、执行、反思 |
| `from core.skill import SkillGateway` | **SkillGateway** | 技能网关，管理技能的发现、检索、执行、下载 |
| `from middleware.llm import LLMClient` | **LLMClient** | LLM 客户端，基于 litellm 调用各种大模型 |
| `from middleware.config import g_config` | **ConfigManager** (g_config) | 全局配置管理器 |
| `from shared.chat import ChatManager` | **ChatManager** | 会话管理器，管理 Session 和 Conversation |
| `from utils.event_bus import event_bus` | **EventBus** | 事件总线，用于跨层通信 |

### 3.2 Flet GUI 关键模块及其调用的运行时功能

| GUI 模块 | 主要功能 | 调用的后端 |
|---------|---------|-----------|
| **`gui/app.py`** | 发送消息、处理 agent 事件流 | `MementoSAgent` / protocol events |
| **会话管理** | 会话管理 | `ChatManager` (Session/Conversation) |
| **认证面板** | 认证服务 | `gui/modules/auth_service.py` |
| **设置面板** | 设置面板 | `g_config` (`ConfigManager`) |
| **技能市场** | 技能市场 | `SkillGateway`, `SkillMarket` |

## 四、后端核心类详解

### 4.1 MementoSAgent (核心 Agent)

**文件**: `core/memento_s/agent.py`

```python
class MementoSAgent:
    async def run_stream(self, session_id: str, input_text: str) -> AsyncGenerator[dict, None]:
        """流式执行主入口"""

    async def reply_stream(self, session_id: str, ...):
        """回复流 - 协调多阶段执行"""
```

**功能**:
- **意图识别** (Intent Recognition): 识别用户输入是直接回复、计划执行还是中断
- **计划生成** (Planning): 生成执行计划
- **执行** (Execution): 执行计划中的步骤
- **反思** (Reflection): 评估执行结果
- **上下文管理**: 使用 ContextManager 管理对话上下文

### 4.2 SkillGateway (技能网关)

**文件**: `core/skill/gateway.py`

```python
class SkillGateway:
    async def discover(self) -> list[SkillManifest]:
        """发现可用技能"""

    async def recall(self, query: str, ...):
        """检索相关技能"""

    async def execute(self, skill_name: str, ...):
        """执行技能"""
```

**功能**:
- 技能发现与注册
- 技能检索 (本地/远程)
- 技能执行与治理
- 技能下载管理

### 4.3 LLMClient (大模型客户端)

**文件**: `middleware/llm/llm_client.py`

```python
class LLMClient:
    async def chat_completion(self, messages: list[Message], ...):
        """聊天完成"""

    async def stream_chat(self, messages: list[Message], ...):
        """流式聊天完成"""
```

**功能**:
- 基于 **litellm** 的多提供商支持 (OpenAI, Anthropic, Azure, 本地模型等)
- 自动重试、熔断保护、超时控制
- Token 计算

### 4.4 ChatManager (会话管理)

**文件**: `shared/chat/chat_manager.py`

```python
class ChatManager:
    @classmethod
    async def create_session(cls, title: str) -> SessionInfo:
        """创建新会话"""

    @classmethod
    async def create_conversation(cls, session_id: str, role: str, ...):
        """创建对话"""

    @classmethod
    async def get_conversation_history(cls, session_id: str, limit: int):
        """获取历史对话"""
```

**功能**:
- 统一的 Session 和 Conversation 管理入口
- GUI、Agent、CLI、IM 的统一接口

### 4.5 IM Gateway (即时通讯网关)

**文件**: `middleware/im/gateway/gateway.py`

```python
class Gateway:
    """支持 WebSocket 连接和 Webhook 回调"""

    async def start_server(self):
        """启动 WebSocket 服务器"""

    async def handle_websocket(self, websocket):
        """处理 WebSocket 连接"""
```

**功能**:
- 支持多种 IM 渠道: **飞书、钉钉、企业微信**
- WebSocket 服务器: Agent/Tool Worker 连接
- Webhook 服务器: 渠道回调
- 消息路由和分发

## 五、界面与运行时通信方式

### 5.1 Flet 进程内调用（桌面 GUI）

Flet GUI 与 agent runtime 位于同一 Python 进程。界面 controller 直接调用
`MementoSAgent`、`ChatManager` 和 `SkillGateway`，并消费 protocol event stream：

```python
from core.memento_s import MementoSAgent
from shared.chat import ChatManager

agent = MementoSAgent()
# GUI controllers iterate the agent's async event stream and update Flet controls.
```

The optional HTTP/SSE service remains a separate interface for API clients; it
is not an Electron-to-Python bridge.

### 5.2 AG-UI 协议 (事件流)

用于流式 AI 响应，通过 **事件流** 传输：

核心协议文件:
- `core/protocol/run_emitter.py`
- `core/protocol/pipeline.py`
- `core/protocol/events.py`

事件类型:
- `TEXT_MESSAGE_START` / `CONTENT` / `END` - 文本消息
- `RUN_STARTED` / `RUN_FINISHED` / `RUN_ERROR` - 运行状态
- `STEP_STARTED` / `STEP_FINISHED` - 步骤执行
- `PLAN_GENERATED` - 计划生成
- `TOOL_CALL_STARTED` / `TOOL_CALL_ENDED` - 工具调用

### 5.3 Event Bus (事件总线)

用于**跨层通信**的发布/订阅机制：

```python
# 文件: utils/event_bus.py

# 订阅事件
event_bus.subscribe(EventType.IM_SERVICE_STARTED, on_im_started)

# 发布事件
event_bus.publish(EventType.AUTH_REQUIRED, {"token": "..."})
```

事件类型:
- IM 服务事件: `IM_SERVICE_STARTED`, `IM_SERVICE_STOPPED`, etc.
- 网关事件: `GATEWAY_STARTED`, `GATEWAY_STOPPED`
- 认证事件: `AUTH_REQUIRED` (HTTP 401)
- 通用事件: `CONFIG_CHANGED`, `ERROR_OCCURRED`

### 5.4 WebSocket (IM 集成)

对于 IM 渠道集成 (飞书、钉钉、企业微信)，使用 WebSocket：

```python
# WebSocket 服务器
import websockets

async def start_server(self):
    await websockets.serve(self.handle_websocket, host, port)
```

## 六、数据存储层

使用 **SQLite** 作为主要存储:

| 服务 | 文件位置 | 功能 |
|-----|---------|-----|
| **DatabaseManager** | `middleware/storage/core/engine.py` | SQLite 数据库管理 |
| **SessionService** | `middleware/storage/services/session_service.py` | 会话存储 |
| **ConversationService** | `middleware/storage/services/conversation_service.py` | 对话存储 |
| **SkillService** | `middleware/storage/services/skill_service.py` | 技能存储 |
| **VectorStorage** | `middleware/storage/vector_storage.py` | 向量存储 (可选) |

### 6.1 数据库文件位置

```
{paths.db_dir}/memento_s.db          # 主数据库（SQLAlchemy + aiosqlite）
{paths.data_dir}/memento.db          # 向量索引（sqlite-vec，可选）
```

SQLite 启动配置：`PRAGMA foreign_keys = ON` / `journal_mode = WAL` / `synchronous = NORMAL`。

### 6.2 对话存储表结构（两层架构）

**`sessions` 表** — 会话层

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | UUID | 主键 |
| `title` | String(255) | 会话标题（默认取用户消息前50字） |
| `status` | String | `active / paused / completed / archived` |
| `meta_info` | JSON | 扩展元数据（模型名称、标签等） |
| `conversation_count` | Integer | 消息条数（写入时自动统计） |
| `total_tokens` | Integer | 累计 token 消耗（写入时自动统计） |
| `created_at / updated_at` | DateTime | 东八区时间戳 |

**`conversations` 表** — 消息层

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | UUID | 主键 |
| `session_id` | FK | 级联删除 |
| `sequence` | Integer | 会话内顺序编号（从1开始，自动分配） |
| `role` | String | `user / assistant / system / tool / system_event`（规划中） |
| `title` | String(255) | 内容前50字，用于侧边栏预览 |
| `content` | Text | 消息正文 |
| `content_detail` | JSON | 多模态内容（预留） |
| `tool_calls` | JSON | 工具调用列表（OpenAI 格式） |
| `tool_call_id` | String | 关联工具调用 ID |
| `meta_info` | JSON | 附加信息（见下文） |
| `tokens` | Integer | 本条消息 token 数 |

`meta_info` 按角色存储的内容：

| role | meta_info 内容 |
|------|---------------|
| `user` | `{timestamp}` |
| `assistant` | `{timestamp, steps, duration_seconds, tokens, reply_to}` |
| `tool` | 无额外字段 |
| `system_event`（规划中） | `{event_type, run_id, intent, plan, reflections, stats}` |

### 6.3 一次完整对话的写入时序

```
POST /api/v1/chat/stream
        │
        ▼
ChatService.stream_reply()
        │
① ChatManager.ensure_session()               → sessions 表：新建或复用
② ChatManager.create_conversation(role="user") → conversations 表
        │
③ agent.reply_stream() 流式生成
        │  ├─ tool_call 发生时 → ToolTranscriptSink
        │  │   → create_conversation(role="assistant", tool_calls=[...])
        │  │   → create_conversation(role="tool", tool_call_id=...)
        │  │
        │  └─ RUN_FINISHED 事件
        │         │
④          ChatManager.create_conversation(role="assistant") → conversations 表
⑤          ChatManager.update_session(title=...)             → sessions 表标题更新
⑥          yield RUN_FINISHED（附带 conversation_id / user_conversation_id）
```

关键约束：
- 用户消息在流开始前**同步**写入（步骤②）
- 助手回复在流结束后**一次性**写入（步骤④），中途中断不保存
- 工具调用由 Agent 层的 `ToolTranscriptSink` 写入，与 `ChatService` 无关

### 6.4 历史消息加载给 Agent 的流程

`ContextManager.load_history()` 从 `conversations` 表读取历史（按 `sequence` 升序），经过两层过滤后传给 LLM：

1. **两层窗口截断**：最近 N 轮（`recent_rounds_keep`）直接保留；更早的消息按 token 预算压缩或通过 embedding 语义检索召回
2. **`tool` 角色过滤**：`list_by_session(exclude_tool=False)` 保留工具消息，但 `system_event` 类型（规划中）需单独过滤，不传给 LLM
3. **tool result 精简**：历史中的大结果替换为 `[historical] preview...`，节省 token

最终格式：
```python
[
    {"role": "user",      "content": "...", "conversation_id": "...", "tokens": 42},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool",      "content": "...", "tool_call_id": "..."},
    ...
]
```

### 6.5 可选向量索引（conversation embeddings）

配置了 embedding 服务后，每条 `user` / `assistant` 消息写入 SQLite 后会**异步（fire-and-forget）**生成向量，存入 `memento.db` 的 sqlite-vec 虚拟表：

```sql
CREATE VIRTUAL TABLE conversation_embeddings USING vec0(
    conversation_id TEXT PRIMARY KEY,
    embedding float[N] distance_metric=cosine
)
```

用于 `load_history` 中对早期消息的语义召回。未配置 embedding 时静默跳过，退化为 token 预算截断。

### 6.6 run 元数据存储（规划中）

当前 `INTENT_RECOGNIZED`、`PLAN_GENERATED`、`REFLECTION_RESULT` 等过程事件不持久化。规划方案：写入一条 `role="system_event"` 的 Conversation，`meta_info` 存储完整 run 摘要，通过独立接口 `GET /api/v1/chat/sessions/{id}/runs` 查询，不混入对话历史，也不传给 LLM。

## 七、关键文件路径

| 模块 | 路径 |
|-----|------|
| GUI 入口 | `gui/app.py` (`memento-gui`) |
| Agent 核心 | `core/memento_s/agent.py` |
| Skill 网关 | `core/skill/gateway.py` |
| LLM 客户端 | `middleware/llm/llm_client.py` |
| 协议层 | `core/protocol/` |
| 事件总线 | `utils/event_bus.py` |
| IM 网关 | `middleware/im/gateway/gateway.py` |
| 会话管理 | `shared/chat/chat_manager.py` |

## 八、架构总结

| 方面 | 说明 |
|-----|------|
| **架构类型** | 分层 Python runtime，共享 CLI、GUI、HTTP/SSE 和 IM 接口 |
| **GUI 框架** | Flet |
| **通信方式** | 进程内调用 + protocol event stream + event bus；HTTP/SSE 为可选外部接口 |
| **AI 能力** | MementoSAgent (多阶段 Agent) |
| **LLM 集成** | litellm (多提供商) |
| **技能系统** | SkillGateway (技能市场) |
| **IM 集成** | WebSocket + Webhook (飞书/钉钉/企业微信) |
| **数据存储** | SQLite |
| **配置管理** | ConfigManager |
