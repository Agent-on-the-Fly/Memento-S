"""
Gateway 主类。

整合所有组件，提供统一的网关入口：
- WebSocket 服务器：Agent/Tool Worker 连接
- Webhook 服务器：渠道回调
- 消息路由：消息分发
- 账户管理：生命周期控制
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import websockets
from websockets.server import WebSocketServerProtocol

from .protocol import (
    AccountConfig,
    ChannelType,
    ConnectionConfig,
    ConnectionMode,
    ConnectionType,
    GatewayMessage,
    MessageType,
    PermissionDomain,
    PROTOCOL_VERSION,
    ChannelAdapterProtocol,
)
from .connection_manager import ConnectionManager
from .router import MessageRouter
from .webhook_server import WebhookServer

# 导入事件总线用于转发适配器事件
from utils.event_bus import EventType, publish

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 渠道注册表
# ---------------------------------------------------------------------------


class ChannelRegistry:
    """
    渠道适配器注册表。

    支持插件式注册新渠道。
    """

    def __init__(self):
        # 适配器类注册表
        self._adapter_classes: dict[ChannelType, type] = {}

        # 适配器实例（运行时）
        self._adapters: dict[str, ChannelAdapterProtocol] = {}

        # 已发现的插件路径
        self._discovered_paths: list[Path] = []

    def register(
        self,
        channel_type: ChannelType,
        adapter_class: type,
    ) -> None:
        """
        注册渠道适配器类。

        Args:
            channel_type: 渠道类型
            adapter_class: 适配器类（必须实现 ChannelAdapterProtocol）
        """
        if channel_type in self._adapter_classes:
            logger.warning(
                "Replacing existing adapter for channel: %s",
                channel_type.value,
            )

        self._adapter_classes[channel_type] = adapter_class

        logger.info(
            "Channel adapter class registered: %s -> %s",
            channel_type.value,
            adapter_class.__name__,
        )

    def create_adapter(
        self,
        channel_type: ChannelType,
        account_id: str,
        **kwargs,
    ) -> ChannelAdapterProtocol | None:
        """
        创建适配器实例。

        Args:
            channel_type: 渠道类型
            account_id: 账户ID
            **kwargs: 适配器构造参数

        Returns:
            适配器实例，未注册返回 None
        """
        adapter_class = self._adapter_classes.get(channel_type)

        if not adapter_class:
            logger.warning(
                "No adapter class for channel: %s",
                channel_type.value,
            )
            return None

        try:
            adapter = adapter_class(**kwargs)
            self._adapters[account_id] = adapter

            logger.info(
                "Adapter instance created: %s/%s",
                channel_type.value,
                account_id,
            )

            return adapter

        except Exception as e:
            logger.error(
                "Failed to create adapter %s/%s: %s",
                channel_type.value,
                account_id,
                e,
            )
            return None

    def get_adapter(self, account_id: str) -> ChannelAdapterProtocol | None:
        """获取适配器实例。"""
        return self._adapters.get(account_id)

    def remove_adapter(self, account_id: str) -> None:
        """移除适配器实例。"""
        self._adapters.pop(account_id, None)

    def list_supported(self) -> list[ChannelType]:
        """列出支持的渠道类型。"""
        return list(self._adapter_classes.keys())

    async def discover_plugins(self, plugin_dir: Path) -> int:
        """
        自动发现并注册插件。

        插件目录结构：
        plugins/
        └── telegram/
            ├── __init__.py
            └── adapter.py  # 包含 TelegramAdapter 类

        Args:
            plugin_dir: 插件目录路径

        Returns:
            发现并注册的插件数量
        """
        if not plugin_dir.exists():
            return 0

        count = 0

        for channel_dir in plugin_dir.iterdir():
            if not channel_dir.is_dir():
                continue

            channel_name = channel_dir.name

            try:
                channel_type = ChannelType(channel_name)
            except ValueError:
                # 不是有效的渠道类型
                continue

            # 尝试导入适配器
            try:
                import importlib.util

                adapter_file = channel_dir / "adapter.py"

                if not adapter_file.exists():
                    continue

                spec = importlib.util.spec_from_file_location(
                    f"channel_{channel_name}",
                    adapter_file,
                )

                if not spec or not spec.loader:
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # 查找适配器类
                adapter_class = getattr(module, "Adapter", None)

                if adapter_class:
                    self.register(channel_type, adapter_class)
                    count += 1

                    logger.info(
                        "Discovered plugin: %s from %s",
                        channel_type.value,
                        channel_dir,
                    )

            except Exception as e:
                logger.debug(
                    "Failed to discover plugin in %s: %s",
                    channel_dir,
                    e,
                )

        self._discovered_paths.append(plugin_dir)

        logger.info("Discovered %d plugins from %s", count, plugin_dir)

        return count


# ---------------------------------------------------------------------------
# 装饰器注册
# ---------------------------------------------------------------------------

# 全局注册表实例
_global_registry = ChannelRegistry()


def register_channel(channel_type: ChannelType):
    """
    装饰器：注册渠道适配器。

    用法：
        @register_channel(ChannelType.TELEGRAM)
        class TelegramAdapter:
            ...
    """

    def decorator(cls):
        _global_registry.register(channel_type, cls)
        return cls

    return decorator


# ---------------------------------------------------------------------------
# 账户管理
# ---------------------------------------------------------------------------


@dataclass
class AccountInfo:
    """账户运行时信息。"""

    account_id: str
    channel_type: ChannelType
    adapter: ChannelAdapterProtocol
    config: AccountConfig
    state: str = "stopped"

    # Webhook 处理器（仅 webhook 模式）
    webhook_handler: Any = None


class AccountManager:
    """
    账户管理器。

    提供 startAccount / stopAccount 生命周期管理。
    """

    def __init__(
        self,
        registry: ChannelRegistry,
        webhook_server: WebhookServer | None = None,
        connection_manager: ConnectionManager | None = None,
        router: MessageRouter | None = None,
    ):
        self._registry = registry
        self._webhook_server = webhook_server
        self._connection_manager = connection_manager
        self._router = router

        # 主线程事件循环引用（供后台线程回调使用）
        self._main_event_loop: asyncio.AbstractEventLoop | None = None

        # 账户存储
        self._accounts: dict[str, AccountInfo] = {}

    async def startAccount(
        self,
        account_id: str,
        channel_type: ChannelType,
        credentials: dict,
        mode: ConnectionMode = ConnectionMode.POLLING,
        permission_domain: PermissionDomain = PermissionDomain.NODE,
        webhook_config: dict | None = None,
        **kwargs,
    ) -> AccountInfo:
        """
        启动账户。

        Args:
            account_id: 账户唯一标识
            channel_type: 渠道类型
            credentials: 认证凭证
            mode: 连接模式
            permission_domain: 权限域
            webhook_config: Webhook 配置（webhook 模式时使用）
            **kwargs: 适配器额外参数

        Returns:
            AccountInfo: 账户信息
        """
        # 检查是否已存在
        if account_id in self._accounts:
            existing = self._accounts[account_id]
            logger.info(
                "[AccountManager] Account %s already exists, state=%s",
                account_id,
                existing.state,
            )

            if existing.state == "running":
                logger.warning(
                    "[AccountManager] Account %s already running - "
                    "skipping initialization, returning existing",
                    account_id,
                )
                return existing

            # 停止旧账户
            logger.info(
                "[AccountManager] Stopping existing account %s before restart",
                account_id,
            )
            await self.stopAccount(account_id)

        logger.info(
            "Starting account: %s (channel=%s, mode=%s, domain=%s)",
            account_id,
            channel_type.value,
            mode.value,
            permission_domain.value,
        )

        # 创建适配器
        adapter = self._registry.create_adapter(
            channel_type,
            account_id,
            **credentials,
            **kwargs,
        )

        if not adapter:
            raise ValueError(
                f"Failed to create adapter for channel: {channel_type.value}"
            )

        # 检查协议版本
        if adapter.protocol_version > PROTOCOL_VERSION:
            logger.warning(
                "Adapter version (%d) > Gateway version (%d), "
                "some features may not work",
                adapter.protocol_version,
                PROTOCOL_VERSION,
            )

        # 创建配置
        config = AccountConfig(
            account_id=account_id,
            channel_type=channel_type,
            credentials=credentials,
            mode=mode,
            permission_domain=permission_domain,
            webhook_config=webhook_config or {},
            metadata=kwargs,
        )

        # 初始化适配器
        connection_config = config.to_connection_config()
        logger.info(
            "[AccountManager] About to initialize adapter for %s (type=%s)",
            account_id,
            type(adapter).__name__,
        )
        try:
            await adapter.initialize(connection_config, mode)
            logger.info(
                "[AccountManager] ✓ Adapter initialized successfully for %s", account_id
            )
        except Exception as e:
            logger.error(
                "[AccountManager] ✗ Adapter initialization FAILED for %s: %s",
                account_id,
                e,
                exc_info=True,
            )
            raise

        # 注册消息回调 - 使用同步包装器处理异步回调
        # 因为回调可能从后台线程调用，需要使用 run_coroutine_threadsafe
        def sync_message_callback(msg: GatewayMessage) -> None:
            """同步包装器：在后台线程中通过主线程事件循环调度异步处理"""
            try:
                if self._router:
                    # 设置 connection_id
                    msg.connection_id = f"channel:{account_id}"

                    # 使用主线程的事件循环通过 run_coroutine_threadsafe 调度
                    loop = self._main_event_loop
                    if loop and not loop.is_closed():
                        coro = self._router.route(msg)
                        asyncio.run_coroutine_threadsafe(coro, loop)
                    else:
                        # 主线程没有事件循环，尝试创建任务
                        try:
                            asyncio.create_task(self._router.route(msg))
                        except RuntimeError as e:
                            logger.error(
                                f"[_on_adapter_message] Cannot schedule task: {e}"
                            )
                else:
                    logger.warning(
                        "[_on_adapter_message] No router available, message dropped"
                    )
            except Exception as e:
                logger.error(f"[_on_adapter_message] Error: {e}", exc_info=True)

        adapter.on_message(sync_message_callback)

        # 注册事件回调 - 处理适配器事件（如 session_expired）
        def on_adapter_event(event: dict) -> None:
            """处理适配器事件，转发到事件总线"""
            event_type = event.get("type")
            if event_type == "session_expired":
                platform = event.get("account_id", "unknown")
                logger.warning(
                    f"[AccountManager] Session expired for {platform}, emitting event"
                )
                # 发布到全局事件总线
                publish(
                    EventType.IM_SERVICE_START_FAILED,
                    {
                        "platform": "微信",
                        "platform_id": platform,
                        "error": "微信会话已过期，请重新登录",
                        "event_type": "session_expired",
                        "requires_relogin": True,
                    },
                )
            else:
                logger.debug(f"[AccountManager] Adapter event: {event}")

        adapter.on_event(on_adapter_event)

        # Webhook 模式：注册处理器
        webhook_handler = None
        if mode in (ConnectionMode.WEBHOOK, ConnectionMode.HYBRID):
            if self._webhook_server:
                webhook_handler = self._webhook_server.register(
                    channel_type,
                    account_id,
                    adapter,
                )

                webhook_handler.on_message(
                    lambda msgs: self._on_webhook_messages(account_id, msgs)
                )

        # 启动适配器
        if mode != ConnectionMode.WEBHOOK:
            await adapter.start()

        # 创建账户信息
        account_info = AccountInfo(
            account_id=account_id,
            channel_type=channel_type,
            adapter=adapter,
            config=config,
            state="running",
            webhook_handler=webhook_handler,
        )

        self._accounts[account_id] = account_info

        logger.info("Account started: %s", account_id)

        return account_info

    async def stopAccount(self, account_id: str) -> None:
        """
        停止账户。

        Args:
            account_id: 账户ID
        """
        if account_id not in self._accounts:
            return

        account = self._accounts[account_id]

        logger.info("Stopping account: %s", account_id)

        # 停止适配器
        try:
            await account.adapter.stop()
        except Exception as e:
            logger.error("Error stopping adapter %s: %s", account_id, e)

        # 注销 Webhook 处理器
        if account.webhook_handler and self._webhook_server:
            self._webhook_server.unregister(
                account.channel_type,
                account_id,
            )

        # 移除适配器实例
        self._registry.remove_adapter(account_id)

        account.state = "stopped"
        del self._accounts[account_id]

        logger.info("Account stopped: %s", account_id)

    async def stopAll(self) -> None:
        """停止所有账户。"""
        for account_id in list(self._accounts.keys()):
            await self.stopAccount(account_id)

    def getAccount(self, account_id: str) -> AccountInfo | None:
        """获取账户信息。"""
        return self._accounts.get(account_id)

    def listAccounts(self) -> list[AccountInfo]:
        """列出所有账户。"""
        return list(self._accounts.values())

    def set_main_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """设置主线程事件循环，供后台线程回调使用。"""
        self._main_event_loop = loop
        logger.info("[AccountManager] Main event loop set: %s", loop)

    def _on_adapter_message(self, account_id: str, message: GatewayMessage) -> None:
        """适配器消息回调。"""
        # 设置来源信息
        message.connection_id = f"channel:{account_id}"

        logger.info(
            "Adapter message received: account=%s, type=%s, connection_id=%s, session_id=%s, content=%s",
            account_id,
            message.type.value,
            message.connection_id,
            message.session_id,
            (message.content or "")[:50],
        )

        # 直接通过路由器分发消息到 Agent
        if self._router:
            asyncio.create_task(self._router.route(message))
        else:
            logger.warning("No router available, message dropped")

    def _on_webhook_messages(
        self,
        account_id: str,
        messages: list[GatewayMessage],
    ) -> None:
        """Webhook 消息回调。"""
        for message in messages:
            message.connection_id = f"channel:{account_id}"

            if self._connection_manager:
                asyncio.create_task(
                    self._connection_manager.handle_message(
                        message.connection_id,
                        message,
                    )
                )


# ---------------------------------------------------------------------------
# Gateway 主类
# ---------------------------------------------------------------------------


class Gateway:
    """
    Gateway - 网关主类。

    整合所有组件：
    - WebSocket 服务器：Agent/Tool Worker 连接
    - Webhook 服务器：渠道回调
    - 消息路由：消息分发
    - 账户管理：生命周期控制

    使用示例：
        gateway = Gateway()
        await gateway.start()

        # 启动渠道账户
        await gateway.startAccount(
            account_id="telegram_bot",
            channel_type=ChannelType.TELEGRAM,
            credentials={"bot_token": "..."},
        )

        # 运行
        await gateway.wait_closed()

        # 关闭
        await gateway.shutdown()
    """

    def __init__(
        self,
        websocket_host: str = "0.0.0.0",
        websocket_port: int = 8765,
        webhook_host: str = "0.0.0.0",
        webhook_port: int = 18080,
        registry: ChannelRegistry | None = None,
    ):
        # 配置
        self.websocket_host = websocket_host
        self.websocket_port = websocket_port
        self.webhook_host = webhook_host
        self.webhook_port = webhook_port

        # 组件
        self.registry = registry or _global_registry
        self.connection_manager = ConnectionManager()
        self.router = MessageRouter(self.connection_manager)
        self.webhook_server = WebhookServer(webhook_host, webhook_port)
        self.account_manager = AccountManager(
            self.registry,
            self.webhook_server,
            self.connection_manager,
            self.router,
        )

        # WebSocket 服务器
        self._ws_server: websockets.Serve | None = None

        # 运行状态
        self._running = False
        self._closed = asyncio.Event()

        # 主线程事件循环引用（用于从后台线程调度任务）
        self._main_event_loop: asyncio.AbstractEventLoop | None = None

        # 注册消息回调
        self.connection_manager.on_message(self._on_connection_message)

    # ---- 生命周期 ----

    async def start(self) -> None:
        """启动 Gateway。"""
        if self._running:
            return

        # 保存主线程事件循环引用，供后台线程回调使用
        try:
            self._main_event_loop = asyncio.get_running_loop()
            logger.debug("[Gateway] Saved main event loop: %s", self._main_event_loop)
            # 同步到 AccountManager
            self.account_manager.set_main_event_loop(self._main_event_loop)
        except RuntimeError:
            self._main_event_loop = None
            logger.warning("[Gateway] No event loop in start()")

        logger.info("Starting Gateway...")

        # 启动连接管理器
        await self.connection_manager.start()

        # 启动 Webhook 服务器
        await self.webhook_server.start()

        # 启动 WebSocket 服务器
        self._ws_server = await websockets.serve(
            self._handle_websocket,
            self.websocket_host,
            self.websocket_port,
        )

        self._running = True

        logger.info(
            "Gateway started:\n"
            "  - WebSocket: ws://%s:%d\n"
            "  - Webhook: http://%s:%d/webhook/{{channel}}/{{account}}",
            self.websocket_host,
            self.websocket_port,
            self.webhook_host,
            self.webhook_port,
        )

    async def shutdown(self) -> None:
        """关闭 Gateway。"""
        if not self._running:
            return

        logger.info("Shutting down Gateway...")

        self._running = False

        # 停止账户
        await self.account_manager.stopAll()

        # 关闭 WebSocket 服务器
        if self._ws_server:
            self._ws_server.close()
            await self._ws_server.wait_closed()

        # 停止 Webhook 服务器
        await self.webhook_server.stop()

        # 停止连接管理器
        await self.connection_manager.stop()

        # 设置关闭事件
        self._closed.set()

        logger.info("Gateway shutdown complete")

    async def wait_closed(self) -> None:
        """等待 Gateway 关闭。"""
        await self._closed.wait()

    # ---- 账户管理 ----

    async def startAccount(
        self,
        account_id: str,
        channel_type: ChannelType,
        credentials: dict,
        mode: ConnectionMode = ConnectionMode.POLLING,
        permission_domain: PermissionDomain = PermissionDomain.NODE,
        **kwargs,
    ) -> AccountInfo:
        """启动账户。"""
        return await self.account_manager.startAccount(
            account_id=account_id,
            channel_type=channel_type,
            credentials=credentials,
            mode=mode,
            permission_domain=permission_domain,
            **kwargs,
        )

    async def stopAccount(self, account_id: str) -> None:
        """停止账户。"""
        await self.account_manager.stopAccount(account_id)

    # ---- 消息发送 ----

    async def send_response(self, message: GatewayMessage) -> bool:
        """发送响应消息。"""
        return await self.router.route(message)

    async def send_to_connection(
        self,
        connection_id: str,
        message: GatewayMessage,
    ) -> bool:
        """发送消息到指定连接。"""
        return await self.connection_manager.send(connection_id, message)

    # ---- WebSocket 处理 ----

    async def _handle_websocket(
        self,
        websocket: WebSocketServerProtocol,
    ) -> None:
        """处理 WebSocket 连接。"""
        connection_id = ""
        source = ""

        try:
            # 等待 CONNECT 消息
            try:
                data = await asyncio.wait_for(
                    websocket.recv(),
                    timeout=10.0,
                )
                message = GatewayMessage.from_json(data)

                if message.type != MessageType.CONNECT:
                    logger.warning("First message must be CONNECT")
                    await websocket.close()
                    return

                source = message.source
                source_type = message.source_type or "unknown"

            except asyncio.TimeoutError:
                logger.warning("Connection timeout: no CONNECT message")
                await websocket.close()
                return

            # 创建连接配置
            connection_id = f"ws:{source}"
            try:
                conn_type = ConnectionType(source_type.lower())
            except ValueError:
                # 未知的连接类型，使用默认值
                conn_type = ConnectionType.CHANNEL
            config = ConnectionConfig(
                connection_id=connection_id,
                connection_type=conn_type,
                metadata={
                    "source": source,
                    "source_type": source_type,
                },
            )

            # 注册连接
            info = await self.connection_manager.register(websocket, config)

            # 注册来源
            self.connection_manager.register_source(source, connection_id)

            # 注册目标（Agent/Tool）
            if source_type == "agent":
                self.router.register_agent(source)
            elif source_type == "tool":
                tools = message.metadata.get("tools", [])
                self.router.register_tool(source, tools)

            # 发送确认
            ack = GatewayMessage(
                type=MessageType.CONNECT_ACK,
                target=source,
                target_type=source_type,
                connection_id=connection_id,
                metadata={"status": "connected"},
            )
            await info.send_queue.put(ack)

            # 接收消息循环
            async for data in websocket:
                try:
                    msg = GatewayMessage.from_json(data)
                    await self.connection_manager.handle_message(connection_id, msg)

                except Exception as e:
                    logger.error("Error handling message: %s", e)

        except websockets.ConnectionClosed:
            pass
        except Exception as e:
            logger.error("WebSocket handler error: %s", e)
        finally:
            # 注销连接
            if connection_id:
                await self.connection_manager.unregister(connection_id)

            # 注销目标
            if source:
                self.router.unregister_target(source)

    # ---- 消息回调 ----

    async def _on_connection_message(
        self,
        conn_info: Any,
        message: GatewayMessage,
    ) -> None:
        """连接消息回调。"""
        # 路由消息
        await self.router.route(message)

    # ---- 插件管理 ----

    def registerChannel(
        self,
        channel_type: ChannelType,
        adapter_class: type,
    ) -> None:
        """注册渠道适配器。"""
        self.registry.register(channel_type, adapter_class)

    async def discoverPlugins(self, plugin_dir: Path) -> int:
        """发现并注册插件。"""
        return await self.registry.discover_plugins(plugin_dir)

    # ---- 状态查询 ----

    def listAccounts(self) -> list[AccountInfo]:
        """列出所有账户。"""
        return self.account_manager.listAccounts()

    def listChannels(self) -> list[ChannelType]:
        """列出支持的渠道类型。"""
        return self.registry.list_supported()

    def getStats(self) -> dict:
        """获取统计信息。"""
        return {
            "running": self._running,
            "accounts": len(self.account_manager.listAccounts()),
            "connections": self.connection_manager.get_stats(),
            "routing": self.router.get_routing_stats(),
            "channels": [c.value for c in self.registry.list_supported()],
        }

    def getWebhookUrl(
        self,
        channel_type: ChannelType,
        account_id: str,
        base_url: str = "",
    ) -> str:
        """获取 Webhook URL。"""
        return self.webhook_server.get_webhook_url(
            channel_type,
            account_id,
            base_url,
        )


# ---------------------------------------------------------------------------
# 全局实例
# ---------------------------------------------------------------------------

_gateway_instance: Gateway | None = None


def get_gateway() -> Gateway:
    """获取全局 Gateway 实例。"""
    global _gateway_instance
    if _gateway_instance is None:
        _gateway_instance = Gateway()
    return _gateway_instance


def set_gateway(gateway: Gateway) -> None:
    """设置全局 Gateway 实例。"""
    global _gateway_instance
    _gateway_instance = gateway
