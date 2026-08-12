# SPDX-License-Identifier: Apache-2.0
"""
server/endpoint/im/service.py
统一 IM 接入服务。

整合 Gateway + Agent Worker，统一管理所有 IM 渠道生命周期。
所有调用方（CLI/GUI/Bootstrap/HTTP API）通过此类访问 IM 接入能力。
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime
from typing import Any

import websockets

from utils.logger import get_logger
from shared.chat import ChatManager
from .base import BaseEndpoint

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 渠道凭证配置
# ---------------------------------------------------------------------------

# 定义每个渠道的凭证字段及其默认值
CHANNEL_CREDENTIALS_CONFIG = {
    "feishu": {
        "fields": ["app_id", "app_secret", "encrypt_key", "verification_token", "webhook_url", "base_url"],
        "required_fields": ["app_id", "app_secret"],
        "default_mode": "websocket",
    },
    "dingtalk": {
        "fields": ["app_key", "app_secret", "webhook_url", "webhook_secret", "base_url"],
        "required_fields": ["app_key"],
        "default_mode": "websocket",
    },
    "wecom": {
        "fields": ["corp_id", "agent_id", "secret", "token", "bot_id", "bot_secret", "webhook_url", "encoding_aes_key", "base_url"],
        "required_fields": [],  # 至少需要 bot_id 或 corp_id
        "default_mode": "websocket",
    },
    "wechat": {
        "fields": ["token", "bot_token", "base_url"],
        "required_fields": ["token"],
        "default_mode": "polling",
    },
}


def extract_channel_credentials(platform_name: str, pcfg: Any) -> dict[str, Any]:
    """从平台配置对象中提取所有凭证字段。

    Args:
        platform_name: 平台名称（feishu, dingtalk, wecom, wechat）
        pcfg: 平台配置对象（Pydantic 模型或字典）

    Returns:
        包含所有凭证字段的字典
    """
    config = CHANNEL_CREDENTIALS_CONFIG.get(platform_name, {"fields": []})
    fields = config.get("fields", [])

    credentials = {}
    for field in fields:
        # 从配置对象中获取字段值
        if hasattr(pcfg, field):
            value = getattr(pcfg, field, None)
        elif isinstance(pcfg, dict):
            value = pcfg.get(field)
        else:
            value = None

        # 处理默认值
        if value is None or value == "":
            # 为特定字段设置默认值
            if field == "base_url":
                if platform_name == "feishu":
                    value = "https://open.feishu.cn"
                elif platform_name == "dingtalk":
                    value = "https://api.dingtalk.com"
                elif platform_name == "wecom":
                    value = "https://qyapi.weixin.qq.com"
                elif platform_name == "wechat":
                    value = "https://ilinkai.weixin.qq.com"
            else:
                value = ""

        credentials[field] = value

    return credentials


def check_channel_required_fields(platform_name: str, credentials: dict[str, Any]) -> tuple[bool, list[str]]:
    """检查渠道凭证是否包含必需的字段。

    Args:
        platform_name: 平台名称
        credentials: 凭证字典

    Returns:
        (是否有效, 缺失的必需字段列表)
    """
    config = CHANNEL_CREDENTIALS_CONFIG.get(platform_name, {"required_fields": []})
    required_fields = config.get("required_fields", [])

    missing_fields = []
    for field in required_fields:
        value = credentials.get(field)
        if not value or (isinstance(value, str) and not value.strip()):
            missing_fields.append(field)

    return len(missing_fields) == 0, missing_fields


# ---------------------------------------------------------------------------
# Agent Worker（从 middleware/im/gateway/agent_worker.py 合并）
# ---------------------------------------------------------------------------


class AgentWorker:
    """Agent WebSocket 客户端，连接到 Gateway 并处理消息。"""

    def __init__(
        self,
        gateway_url: str = "ws://127.0.0.1:8765",
        agent_id: str = "agent_main",
    ):
        self.gateway_url = gateway_url
        self.agent_id = agent_id
        self._ws = None
        self._agent = None
        self._running = False
        self._sender_sessions: dict[str, str] = {}
        self._pending_tasks: set[asyncio.Task] = set()

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        if self._running:
            return
        logger.info(f"[AgentWorker] Starting, connecting to {self.gateway_url}...")
        await self._init_agent()
        await self._connect_gateway()
        self._running = True
        logger.info("[AgentWorker] Started successfully")

    async def _init_agent(self) -> None:
        from core.skill import init_skill_system
        from core.memento_s.agent import MementoSAgent

        skill_gateway = await init_skill_system()
        self._agent = MementoSAgent(skill_gateway=skill_gateway)
        logger.info("[AgentWorker] MementoSAgent initialized")

    async def _connect_gateway(self) -> None:
        from middleware.im.gateway.protocol import GatewayMessage, MessageType

        try:
            self._ws = await websockets.connect(self.gateway_url)
            connect_msg = GatewayMessage(
                type=MessageType.CONNECT,
                source=self.agent_id,
                source_type="agent",
                metadata={"capabilities": ["chat", "skill_execution"]},
            )
            await self._ws.send(connect_msg.to_json())
            response = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
            ack_msg = GatewayMessage.from_json(response)
            if ack_msg.type != MessageType.CONNECT_ACK:
                raise RuntimeError(f"Gateway did not acknowledge: {ack_msg.type}")
            logger.info(f"[AgentWorker] Connected to Gateway as {self.agent_id}")
            task = asyncio.create_task(self._message_loop())
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)
        except Exception as e:
            logger.error(f"[AgentWorker] Failed to connect to Gateway: {e}")
            raise

    async def _message_loop(self) -> None:
        from middleware.im.gateway.protocol import GatewayMessage, MessageType

        try:
            async for data in self._ws:
                try:
                    msg = GatewayMessage.from_json(data)
                    if msg.type == MessageType.CHANNEL_MESSAGE:
                        task = asyncio.create_task(self._handle_channel_message(msg))
                        self._pending_tasks.add(task)
                        task.add_done_callback(self._pending_tasks.discard)
                    elif msg.type == MessageType.PING:
                        pong = GatewayMessage(
                            type=MessageType.PONG,
                            source=self.agent_id,
                            correlation_id=msg.id,
                        )
                        await self._ws.send(pong.to_json())
                except Exception as e:
                    logger.error(f"[AgentWorker] Error processing message: {e}")
        except websockets.ConnectionClosed:
            logger.warning("[AgentWorker] Gateway connection closed")
            self._running = False
        except Exception as e:
            logger.error(f"[AgentWorker] Message loop error: {e}")
            self._running = False

    async def _handle_channel_message(self, msg: Any) -> None:
        from middleware.im.gateway.protocol import GatewayMessage, MessageType

        try:
            sender_id = msg.sender_id or msg.metadata.get("sender_id", "")
            content = msg.content or ""
            channel = (
                msg.channel_type.value
                if msg.channel_type
                else msg.metadata.get("channel", "")
            )
            account_id = msg.channel_account or msg.metadata.get("account_id", "")
            original_connection_id = msg.metadata.get("original_connection_id", "")

            if not sender_id or not content:
                logger.warning(
                    f"[AgentWorker] Empty sender_id or content, skipping"
                )
                return

            if not self._is_channel_enabled(channel):
                logger.warning(f"[AgentWorker] Channel {channel} is disabled, skipping")
                return

            session_id = await self._get_or_create_session(sender_id, channel, content)
            logger.info(f"[AgentWorker] Using session: {session_id}")

            # 构建用户消息事件（用于前端展示）
            import uuid
            from server.schema.uni_response import UniResponse

            user_event = {
                "type": "USER_INPUT",
                "timestamp": datetime.utcnow().isoformat(),
                "session_id": session_id,
                "conversation_id": str(uuid.uuid4()),
                "run_id": str(uuid.uuid4()),
                "role": "user",
                "content": content,
                "event_id": f"msg_{uuid.uuid4().hex[:12]}",
                "payload": {"messages": "", "content": content},
                "meta": {
                    "channel": channel,
                    "sender_id": sender_id,
                    "account_id": account_id,
                    "im_message_id": msg.id,
                    "chat_id": msg.chat_id,
                    "connection_id": msg.connection_id,
                    "chat_type": msg.metadata.get("chat_type", ""),
                    "msg_type": msg.msg_type,
                    "media_urls": msg.media_urls,
                },
            }
            user_content_detail = UniResponse.from_event(user_event).model_dump(mode="json", exclude_none=True)

            user_conv = await ChatManager.create_conversation(
                session_id=session_id,
                conversation_id=user_content_detail.get("conversation_id") or "",
                role="user",
                title=content[:50] + "..." if len(content) > 50 else content,
                content=content,
                content_detail=user_content_detail,
                meta_info={
                    "channel": channel,
                    "sender_id": sender_id,
                    "account_id": account_id,
                    "im_message_id": msg.id,
                    "chat_id": msg.chat_id,
                    "connection_id": msg.connection_id,
                    "chat_type": msg.metadata.get("chat_type", ""),
                    "msg_type": msg.msg_type,
                    "media_urls": msg.media_urls,
                    "raw_metadata": dict(msg.metadata),
                },
            )

            final_text = ""
            conversation_id = user_content_detail.get("conversation_id", "")
            _text_start_event: dict[str, Any] | None = None
            _text_buf: list[str] = []
            _text_run_id: str | None = None

            async def _save_event(event: dict[str, Any], role: str, title: str, content: str, tokens: int = 0, conv_id: str | None = None) -> None:
                """Save a single conversation event to DB."""
                content_detail = UniResponse.from_event(event).model_dump(mode="json", exclude_none=True)
                await ChatManager.create_conversation(
                    session_id=session_id,
                    conversation_id=conv_id or conversation_id,
                    role=role,
                    title=title,
                    content=content,
                    content_detail=content_detail,
                    meta_info={
                        "reply_to": user_conv.id,
                        "channel": channel,
                        "account_id": account_id,
                        "chat_id": msg.chat_id,
                        "connection_id": msg.connection_id,
                        "chat_type": msg.metadata.get("chat_type", ""),
                    },
                    tokens=tokens,
                )

            async for event in self._agent.reply_stream(
                session_id=session_id, user_content=content
            ):
                event_type = event.get("type", "")

                # 管理 run_id
                if event_type == "TEXT_MESSAGE_START":
                    _text_run_id = str(uuid.uuid4())
                elif event_type in ("TEXT_MESSAGE_CONTENT", "TEXT_MESSAGE_END"):
                    pass  # 复用同一个 _text_run_id
                else:
                    _text_run_id = str(uuid.uuid4())

                # 更新事件中的 run_id 和 conversation_id
                if _text_run_id:
                    event["run_id"] = _text_run_id
                event["conversation_id"] = conversation_id

                # TEXT_MESSAGE 三段式：累积后在 END 时统一持久化
                if event_type == "TEXT_MESSAGE_START":
                    _text_start_event = event
                    _text_buf = []
                    continue

                if event_type == "TEXT_MESSAGE_CONTENT":
                    _text_buf.append(event.get("delta") or "")
                    continue

                if event_type == "TEXT_MESSAGE_END":
                    full_text = "".join(_text_buf)

                    # 持久化 START
                    if _text_start_event:
                        await _save_event(
                            _text_start_event,
                            role="assistant",
                            title="",
                            content="",
                        )

                    # 持久化 CONTENT（payload.content 放完整文本）
                    content_event = {
                        **event,
                        "type": "TEXT_MESSAGE_CONTENT",
                        "delta": full_text,
                        "payload": {"messages": "", "content": full_text},
                    }
                    await _save_event(
                        content_event,
                        role="assistant",
                        title=full_text[:50] + "..." if len(full_text) > 50 else full_text,
                        content=full_text,
                    )

                    # 持久化 END
                    await _save_event(
                        event,
                        role="assistant",
                        title="",
                        content="",
                    )

                    final_text = full_text
                    _text_start_event = None
                    _text_buf = []
                    continue

                # RUN_FINISHED / RUN_ERROR
                if event_type in ("RUN_FINISHED", "RUN_ERROR"):
                    output_text = event.get("outputText") or ""
                    if not final_text and output_text:
                        final_text = output_text
                    # RUN_FINISHED 的 payload.messages 放完整文本，content 为空
                    run_event = {
                        **event,
                        "payload": {"messages": output_text, "content": ""},
                    }
                    await _save_event(
                        run_event,
                        role="assistant",
                        title=output_text[:50] + "..." if len(output_text) > 50 else output_text if output_text else "RUN_FINISHED",
                        content=output_text,
                    )

            if final_text:
                response_msg = GatewayMessage(
                    type=MessageType.AGENT_RESPONSE,
                    source=self.agent_id,
                    target=msg.source,
                    content=final_text,
                    session_id=session_id,
                    channel_type=msg.channel_type,
                    channel_account=msg.channel_account,
                    chat_id=msg.chat_id,
                    correlation_id=msg.id,
                    metadata={
                        "sender_id": sender_id,
                        "channel": channel,
                        "account_id": account_id,
                        "original_connection_id": original_connection_id,
                        "chat_type": msg.metadata.get("chat_type", ""),
                    },
                )
                await self._ws.send(response_msg.to_json())
                logger.info(
                    f"[AgentWorker] Sent response to {sender_id}, session_id={session_id}"
                )
        except Exception as e:
            logger.error(f"[AgentWorker] Error handling channel message: {e}")

    def _is_channel_enabled(self, channel: str) -> bool:
        try:
            from middleware.config import g_config

            config = g_config._runtime_config
            if config is None:
                config = g_config.load()

            gateway_enabled = (
                getattr(config.gateway, "enabled", False)
                if hasattr(config, "gateway")
                else False
            )
            if not gateway_enabled:
                return False

            im_config = getattr(config, "im", None)
            if not im_config:
                return True

            channel_lower = channel.lower()
            if channel_lower in ("feishu", "lark"):
                platform_cfg = getattr(im_config, "feishu", None)
            elif channel_lower in ("dingtalk", "dingding"):
                platform_cfg = getattr(im_config, "dingtalk", None)
            elif channel_lower in ("wecom", "wechatwork", "企业微信"):
                platform_cfg = getattr(im_config, "wecom", None)
            elif channel_lower == "wechat":
                platform_cfg = getattr(im_config, "wechat", None)
            else:
                return True

            if platform_cfg is None:
                return True
            return getattr(platform_cfg, "enabled", True)
        except Exception as e:
            logger.error(f"[AgentWorker] Error checking channel status: {e}")
            return True

    async def _get_or_create_session(self, sender_id: str, channel: str, content: str = "") -> str:
        cache_key = f"{channel}:{sender_id}"
        if cache_key in self._sender_sessions:
            session_id = self._sender_sessions[cache_key]
            if await ChatManager.get_session(session_id):
                return session_id
            del self._sender_sessions[cache_key]

        # 用用户发送的内容生成标题，加上渠道名称前缀
        if content:
            title = f"{channel}: {content[:50]}..." if len(content) > 50 else f"{channel}: {content}"
        else:
            title = f"{channel}"

        session = await ChatManager.create_session(
            title=title,
            metadata={"channel": channel, "sender_id": sender_id},
        )
        session_id = session.id
        self._sender_sessions[cache_key] = session_id
        logger.info(f"[AgentWorker] Created session {session_id} for {cache_key} with title: {title}")
        return session_id

    async def stop(self) -> None:
        if not self._running:
            return
        logger.info("[AgentWorker] Stopping...")
        self._running = False

        # Cancel all pending tasks
        for task in self._pending_tasks:
            if not task.done():
                task.cancel()
        self._pending_tasks.clear()

        if self._ws:
            await self._ws.close()
            self._ws = None
        logger.info("[AgentWorker] Stopped")


# ---------------------------------------------------------------------------
# EndpointService
# ---------------------------------------------------------------------------


class EndpointService:
    """
    统一 IM 接入服务。

    整合 Gateway + Agent Worker，统一管理所有 IM 渠道生命周期。
    """

    _instance: "EndpointService | None" = None

    def __init__(self):
        self._gateway = None
        self._endpoints: dict[str, BaseEndpoint] = {}
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._agent_worker: AgentWorker | None = None

    @classmethod
    def get_instance(cls) -> "EndpointService":
        if cls._instance is None:
            cls._instance = EndpointService()
        return cls._instance

    # ---- 生命周期 ----

    async def start(self) -> None:
        """启动服务（内部启动 Gateway + Agent Worker + 渠道）"""
        if self._running:
            return
        from middleware.im.gateway import Gateway, set_gateway

        # 导入所有渠道适配器以触发装饰器注册
        from middleware.im.gateway import channels

        self._gateway = Gateway()
        set_gateway(self._gateway)
        await self._gateway.start()
        await self._start_agent_worker()

        # 启动配置的渠道
        await self._refresh_channels()

        self._running = True
        logger.info("[EndpointService] Started")

    async def stop(self) -> None:
        """停止服务"""
        if not self._running:
            return
        if self._agent_worker:
            await self._agent_worker.stop()
            self._agent_worker = None
        if self._gateway:
            await self._gateway.shutdown()
            self._gateway = None
        self._running = False
        logger.info("[EndpointService] Stopped")

    def start_in_background(self) -> None:
        """后台线程启动（非阻塞）"""
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run_in_thread, daemon=True, name="endpoint-service"
        )
        self._thread.start()
        logger.info("[EndpointService] Background thread started")

    def _run_in_thread(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self.start())
            while self._running:
                loop.run_until_complete(asyncio.sleep(1))
        finally:
            loop.run_until_complete(self.stop())
            loop.close()
            self._loop = None

    # ---- Agent Worker ----

    async def _start_agent_worker(self) -> None:
        """启动 Agent Worker，连接到 Gateway"""
        if not self._gateway:
            return
        # 使用 127.0.0.1 连接，因为 Gateway 在 0.0.0.0 监听
        # 0.0.0.0 是绑定地址，不能直接用于客户端连接
        ws_host = "127.0.0.1"
        ws_port = self._gateway.websocket_port
        self._agent_worker = AgentWorker(gateway_url=f"ws://{ws_host}:{ws_port}")
        await self._agent_worker.start()

    @property
    def agent_worker(self) -> AgentWorker | None:
        return self._agent_worker

    # ---- 渠道管理 ----

    async def start_channel(
        self,
        account_id: str,
        channel_type: Any,
        credentials: dict,
        mode: Any = None,
    ) -> BaseEndpoint:
        """启动一个 IM 渠道"""
        if not self._gateway:
            await self.start()
        if mode is None:
            from middleware.im.gateway import ConnectionMode

            mode = ConnectionMode.WEBSOCKET
        await self._gateway.startAccount(account_id, channel_type, credentials, mode)
        ctype_str = (
            channel_type.value if hasattr(channel_type, "value") else str(channel_type)
        )
        endpoint = BaseEndpoint(account_id, ctype_str)
        endpoint._running = True
        self._endpoints[account_id] = endpoint
        logger.info(f"[EndpointService] Channel started: {account_id} ({ctype_str})")
        return endpoint

    async def stop_channel(self, account_id: str) -> None:
        """停止一个 IM 渠道"""
        if account_id in self._endpoints:
            if self._gateway:
                await self._gateway.stopAccount(account_id)
            del self._endpoints[account_id]
            logger.info(f"[EndpointService] Channel stopped: {account_id}")

    def list_channels(self) -> list[dict[str, Any]]:
        """列出所有活跃渠道"""
        return [e.get_status() for e in self._endpoints.values()]

    def get_channel(self, account_id: str) -> BaseEndpoint | None:
        return self._endpoints.get(account_id)

    @property
    def gateway(self) -> Any:
        return self._gateway

    @property
    def is_running(self) -> bool:
        return self._running

    def get_startup_error(self) -> str | None:
        """获取服务启动错误信息"""
        return getattr(self, "_startup_error", None)

    async def _refresh_channels(self) -> None:
        """刷新渠道配置（启动/停止已配置的渠道）"""
        if not self._gateway:
            return

        from middleware.config import g_config
        from middleware.im.gateway import ChannelType, ConnectionMode

        config = g_config.load()
        gateway_enabled = (
            getattr(config.gateway, "enabled", False)
            if hasattr(config, "gateway") else False
        )
        if not gateway_enabled:
            return

        platform_map = [
            ("feishu_main", ChannelType.FEISHU, "feishu"),
            ("dingtalk_main", ChannelType.DINGTALK, "dingtalk"),
            ("wecom_main", ChannelType.WECOM, "wecom"),
            ("wechat_main", ChannelType.WECHAT, "wechat"),
        ]
        for account_id, channel_type, platform_name in platform_map:
            if not hasattr(config.im, platform_name):
                continue
            pcfg = getattr(config.im, platform_name)
            enabled = getattr(pcfg, "enabled", False) if hasattr(pcfg, "enabled") else False
            is_running = account_id in self._endpoints
            if enabled and not is_running:
                # 使用通用凭证提取
                credentials = extract_channel_credentials(platform_name, pcfg)

                # 检查必需字段
                is_valid, missing_fields = check_channel_required_fields(platform_name, credentials)
                if not is_valid:
                    logger.warning(
                        f"[EndpointService] {platform_name} credentials missing required fields: {missing_fields}, skipping start"
                    )
                    continue

                # 确定连接模式
                platform_config = CHANNEL_CREDENTIALS_CONFIG.get(platform_name, {})
                default_mode = platform_config.get("default_mode", "websocket")
                mode = ConnectionMode.POLLING if default_mode == "polling" else ConnectionMode.WEBSOCKET

                await self._gateway.startAccount(
                    account_id=account_id,
                    channel_type=channel_type,
                    credentials=credentials,
                    mode=mode,
                )
                self._endpoints[account_id] = BaseEndpoint(account_id, platform_name)
                self._endpoints[account_id]._running = True
                logger.info(f"[EndpointService] Channel started: {account_id} ({platform_name})")
            elif not enabled and is_running:
                await self._gateway.stopAccount(account_id)
                del self._endpoints[account_id]

    def refresh_channels_sync(self) -> None:
        """同步刷新渠道（供非异步代码调用）"""
        if not self._loop or not self._running:
            logger.warning("[EndpointService] Cannot refresh channels: not running")
            return

        import concurrent.futures

        future = asyncio.run_coroutine_threadsafe(self._refresh_channels(), self._loop)
        try:
            future.result(timeout=30)
        except Exception as e:
            logger.error(f"[EndpointService] Channel refresh error: {e}")
