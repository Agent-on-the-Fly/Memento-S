"""
Agent Worker - 连接 Gateway 并处理消息

Agent Worker 是一个独立的组件，通过 WebSocket 连接到 Gateway，
接收来自 IM 渠道的消息，使用本地 MementoSAgent 处理，并将响应发送回 Gateway。

架构：
  IM Channel → Gateway → Agent Worker → MementoSAgent
                                    ↓
                IM Channel ← Gateway ← Agent Worker
"""

from __future__ import annotations

import asyncio

from utils.logger import get_logger
import websockets
from middleware.im.gateway.protocol import GatewayMessage, MessageType
from core.memento_s.agent import MementoSAgent
from shared.chat import ChatManager
from core.skill import init_skill_system
from middleware.config import g_config

logger = get_logger(__name__)


class AgentWorker:
    """Agent Worker - 连接 Gateway 并使用本地 Agent 处理消息。"""

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
        self._sender_sessions: dict[str, str] = {}  # sender_id -> session_id 映射

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """启动 Agent Worker。"""
        if self._running:
            return

        logger.info(f"[AgentWorker] Starting, connecting to {self.gateway_url}...")

        # 1. 初始化本地 Agent
        await self._init_agent()

        # 2. 连接 Gateway
        await self._connect_gateway()

        self._running = True
        logger.info("[AgentWorker] Started successfully")

    async def _init_agent(self) -> None:
        """初始化本地 MementoSAgent。"""
        # 获取已初始化的 SkillGateway
        skill_gateway = await init_skill_system()

        # 创建 Agent
        self._agent = MementoSAgent(
            skill_gateway=skill_gateway,
        )

        logger.info("[AgentWorker] MementoSAgent initialized")

    async def _connect_gateway(self) -> None:
        """连接到 Gateway 并注册。"""
        try:
            self._ws = await websockets.connect(self.gateway_url)
            # 发送 CONNECT 消息注册为 Agent
            connect_msg = GatewayMessage(
                type=MessageType.CONNECT,
                source=self.agent_id,
                source_type="agent",
                metadata={
                    "capabilities": ["chat", "skill_execution"],
                },
            )
            await self._ws.send(connect_msg.to_json())

            # 等待确认
            response = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
            ack_msg = GatewayMessage.from_json(response)

            if ack_msg.type != MessageType.CONNECT_ACK:
                raise RuntimeError(
                    f"Gateway did not acknowledge connection: {ack_msg.type}"
                )

            logger.info(f"[AgentWorker] Connected to Gateway as {self.agent_id}")

            # 启动消息处理循环
            asyncio.create_task(self._message_loop())

        except Exception as e:
            logger.error(f"[AgentWorker] Failed to connect to Gateway: {e}")
            raise

    async def _message_loop(self) -> None:
        """消息处理循环。"""

        try:
            async for data in self._ws:
                try:
                    msg = GatewayMessage.from_json(data)

                    if msg.type == MessageType.CHANNEL_MESSAGE:
                        # 处理来自 IM 渠道的消息
                        asyncio.create_task(self._handle_channel_message(msg))

                    elif msg.type == MessageType.PING:
                        # 响应心跳
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

    async def _handle_channel_message(self, msg: GatewayMessage) -> None:
        """处理来自 IM 渠道的消息。"""
        from middleware.im.gateway.protocol import GatewayMessage, MessageType
        from core.protocol import AGUIEventPipeline, PersistenceSink

        try:
            # 从消息中获取发送者信息（优先从顶层字段获取，回退到 metadata）
            sender_id = msg.sender_id or msg.metadata.get("sender_id", "")
            content = msg.content or ""
            channel = (
                msg.channel_type.value
                if msg.channel_type
                else msg.metadata.get("channel", "")
            )
            account_id = msg.channel_account or msg.metadata.get("account_id", "")
            original_connection_id = msg.metadata.get("original_connection_id", "")

            logger.info(
                f"[AgentWorker] Received message from {channel}/{sender_id}: "
                f"{content[:50]}..., session_id={msg.session_id}, connection_id={original_connection_id}"
            )

            if not sender_id or not content:
                logger.warning(
                    f"[AgentWorker] Empty sender_id or content, skipping. sender_id={sender_id}, "
                    f"content={content[:50]}"
                )
                return

            # 检查渠道是否已启用
            logger.info(f"[AgentWorker] Checking if channel '{channel}' is enabled...")
            is_enabled = self._is_channel_enabled(channel)
            logger.info(
                f"[AgentWorker] Channel '{channel}' enabled result: {is_enabled}"
            )
            if not is_enabled:
                logger.warning(
                    f"[AgentWorker] Channel {channel} is disabled, skipping message"
                )
                return

            # AgentWorker 始终管理自己的 session
            # 使用 sender_id 作为 session key，确保每个用户有独立的对话上下文
            session_id = await self._get_or_create_session(sender_id, channel)
            logger.info(f"[AgentWorker] Using session: {session_id}")

            # 保存用户消息
            user_conv = await ChatManager.create_conversation(
                session_id=session_id,
                role="user",
                title=content[:50] + "..." if len(content) > 50 else content,
                content=content,
                meta_info={"channel": channel, "sender_id": sender_id},
            )

            # 处理消息并收集响应
            final_text = ""

            async def _persist_reply(text: str) -> None:
                nonlocal final_text
                final_text = text
                await ChatManager.create_conversation(
                    session_id=session_id,
                    role="assistant",
                    title=text[:50] + "..." if len(text) > 50 else text,
                    content=text,
                    meta_info={"reply_to": user_conv.id, "channel": channel},
                )

            pipeline = AGUIEventPipeline()
            pipeline.add_sink(PersistenceSink(callback=_persist_reply))

            # Agent 流式处理
            async for event in self._agent.reply_stream(
                session_id=session_id, user_content=content
            ):
                await pipeline.emit(event)

            # 发送响应回 Gateway
            if final_text:
                response_msg = GatewayMessage(
                    type=MessageType.AGENT_RESPONSE,
                    source=self.agent_id,
                    target=msg.source,  # 发回给渠道
                    content=final_text,
                    session_id=session_id,  # 关键: 包含 session_id 用于路由
                    channel_type=msg.channel_type,  # 渠道类型
                    channel_account=msg.channel_account,  # 账户ID
                    chat_id=msg.chat_id,  # 聊天ID
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
                    f"[AgentWorker] Sent response to {sender_id}, session_id={session_id}, "
                    f"connection_id={original_connection_id}"
                )

        except Exception as e:
            logger.error(f"[AgentWorker] Error handling channel message: {e}")

    def _is_channel_enabled(self, channel: str) -> bool:
        """检查渠道是否已启用。

        Args:
            channel: 渠道类型名称（如 'feishu', 'dingtalk', 'wecom'）

        Returns:
            bool: 渠道是否已启用（需要同时满足 Gateway 总开关和平台开关都开启）
        """
        try:
            # 使用 g_config 的缓存配置（不重新加载文件）
            # g_config.set() 已经更新了内存中的配置
            config = g_config._config
            if config is None:
                # 如果缓存为空，则加载
                config = g_config.load()

            # 首先检查 Gateway 总开关
            gateway_enabled = (
                getattr(config.gateway, "enabled", False)
                if hasattr(config, "gateway")
                else False
            )
            if not gateway_enabled:
                logger.info(
                    f"[AgentWorker] Gateway is disabled, channel '{channel}' not allowed"
                )
                return False

            im_config = getattr(config, "im", None)
            if not im_config:
                logger.info(
                    f"[AgentWorker] No im_config found, defaulting to True for channel '{channel}'"
                )
                return True  # 无配置时默认允许

            # Debug: log what im_config contains
            logger.info(f"[AgentWorker] im_config type: {type(im_config)}")
            logger.info(
                f"[AgentWorker] im_config has wechat: {hasattr(im_config, 'wechat')}"
            )
            if hasattr(im_config, "wechat"):
                wechat_cfg = getattr(im_config, "wechat", None)
                logger.info(f"[AgentWorker] wechat_cfg: {wechat_cfg}")
                if wechat_cfg:
                    logger.info(
                        f"[AgentWorker] wechat enabled: {getattr(wechat_cfg, 'enabled', 'N/A')}"
                    )

            # 标准化渠道名称
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
                logger.info(
                    f"[AgentWorker] Unknown channel '{channel_lower}', defaulting to True"
                )
                return True  # 未知渠道默认允许

            if platform_cfg is None:
                logger.info(
                    f"[AgentWorker] No platform_cfg for '{channel_lower}', defaulting to True"
                )
                return True  # 无配置时默认允许

            enabled = getattr(platform_cfg, "enabled", True)
            logger.info(f"[AgentWorker] Channel '{channel_lower}' enabled={enabled}")
            return enabled

        except Exception as e:
            logger.error(
                f"[AgentWorker] Error checking channel status: {e}", exc_info=True
            )
            return True  # 出错时默认允许（避免阻塞正常流程）

    async def _get_or_create_session(self, sender_id: str, channel: str) -> str:
        """获取或创建用户会话。"""
        cache_key = f"{channel}:{sender_id}"

        if cache_key in self._sender_sessions:
            session_id = self._sender_sessions[cache_key]
            if await ChatManager.exists(session_id):
                return session_id
            del self._sender_sessions[cache_key]

        # 创建新会话
        session = await ChatManager.create_session(
            title=f"{channel}: {sender_id}",
            metadata={"channel": channel, "sender_id": sender_id},
        )
        session_id = session.id
        self._sender_sessions[cache_key] = session_id

        logger.info(f"[AgentWorker] Created session {session_id} for {cache_key}")
        return session_id

    async def stop(self) -> None:
        """停止 Agent Worker。"""
        if not self._running:
            return

        logger.info("[AgentWorker] Stopping...")

        self._running = False

        if self._ws:
            await self._ws.close()
            self._ws = None

        logger.info("[AgentWorker] Stopped")


# 便捷函数
_agent_worker: AgentWorker | None = None


def get_agent_worker() -> AgentWorker | None:
    """获取全局 Agent Worker 实例。"""
    return _agent_worker


async def start_agent_worker(gateway_url: str = "ws://127.0.0.1:8765") -> AgentWorker:
    """启动全局 Agent Worker。"""
    global _agent_worker
    if _agent_worker is None:
        _agent_worker = AgentWorker(gateway_url=gateway_url)
        await _agent_worker.start()
    return _agent_worker


async def stop_agent_worker() -> None:
    """停止全局 Agent Worker。"""
    global _agent_worker
    if _agent_worker:
        await _agent_worker.stop()
        _agent_worker = None
