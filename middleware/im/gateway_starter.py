"""
Gateway 模式启动器

在 GUI 或其他单进程场景下，启动 Gateway 并注册本地 Agent，
替代原有的 Bridge 模式。

架构：
  IM Channel (飞书/钉钉等)
        ↓
  Gateway (消息路由中心)
        ↓
  Agent Connection (本地Agent)
        ↓
  MementoSAgent (处理消息)

用法：
    from daemon.gateway_starter import GatewayManager

    manager = GatewayManager()
    await manager.start()  # 启动 Gateway + Agent
    await manager.stop()   # 停止
"""

from __future__ import annotations

import asyncio
import threading

from utils.logger import get_logger

# 导入事件总线（utils 层，不依赖 GUI）
from utils.event_bus import EventType, publish
import middleware.im.gateway

# 平台名称映射（用于事件数据）
_PLATFORM_NAMES = {
    "feishu": "飞书",
    "dingtalk": "钉钉",
    "wecom": "企业微信",
    "wechat": "微信",
}

# 创建 logger（必须在其他使用 logger 的代码之前）
logger = get_logger(__name__)


def _publish_im_event(event_type: EventType, platform: str, error: str = None):
    """发布 IM 服务事件

    Args:
        event_type: 事件类型
        platform: 平台标识（feishu/dingtalk/wecom）
        error: 错误信息（可选）
    """
    data = {
        "platform": _PLATFORM_NAMES.get(platform, platform),
        "platform_id": platform,
    }
    if error:
        data["error"] = error

    publish(event_type, data, source="gateway_starter")

    # 同时记录日志
    if error:
        logger.warning(f"[GatewayManager] {data['platform']}: {error}")
    else:
        logger.info(f"[GatewayManager] {data['platform']} event: {event_type.name}")


# 先导入 gateway 模块（初始化全局注册表）
# 然后导入渠道适配器以触发装饰器注册
logger.info("[GatewayManager] Registering channel adapters...")
try:
    import middleware.im.gateway.channels.feishu

    logger.info("[GatewayManager] ✓ Feishu adapter registered")
except Exception as e:
    logger.error(f"[GatewayManager] ✗ Failed to register Feishu adapter: {e}")

try:
    import middleware.im.gateway.channels.dingtalk

    logger.info("[GatewayManager] ✓ DingTalk adapter registered")
except Exception as e:
    logger.error(f"[GatewayManager] ✗ Failed to register DingTalk adapter: {e}")

try:
    import middleware.im.gateway.channels.wecom

    logger.info("[GatewayManager] ✓ WeCom adapter registered")
except Exception as e:
    logger.error(f"[GatewayManager] ✗ Failed to register WeCom adapter: {e}")

try:
    import middleware.im.gateway.channels.wechat_ilinkai

    logger.info("[GatewayManager] ✓ WeChat adapter registered")
except Exception as e:
    logger.error(f"[GatewayManager] ✗ Failed to register WeChat adapter: {e}")

# 检查注册结果
from middleware.im.gateway.gateway import _global_registry

registered = _global_registry.list_supported()
logger.info(f"[GatewayManager] Registered channels: {[c.value for c in registered]}")

# 全局 GatewayManager 实例
_gateway_manager: GatewayManager | None = None


def get_gateway_manager() -> GatewayManager | None:
    """获取全局 GatewayManager 实例。"""
    return _gateway_manager


class GatewayManager:
    """Gateway 管理器 - 管理 Gateway 和本地 Agent 的生命周期。"""

    def __init__(
        self,
        websocket_host: str = "127.0.0.1",
        websocket_port: int = 8765,
        webhook_host: str = "127.0.0.1",
        webhook_port: int = 18080,
    ):
        self.websocket_host = websocket_host
        self.websocket_port = websocket_port
        self.webhook_host = webhook_host
        self.webhook_port = webhook_port

        self._gateway = None
        self._agent_worker = None
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._startup_error: str | None = None
        self._error_lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def gateway(self):
        return self._gateway

    def get_startup_error(self) -> str | None:
        """获取启动错误信息。"""
        with self._error_lock:
            return self._startup_error

    def start_in_background(self) -> None:
        """在后台线程中启动 Gateway（非阻塞）。"""
        if self._running or (self._thread and self._thread.is_alive()):
            logger.warning("[GatewayManager] Already running")
            return

        self._thread = threading.Thread(
            target=self._run_in_thread,
            name="gateway-manager",
            daemon=True,
        )
        self._thread.start()
        logger.info("[GatewayManager] Background thread started")

    def _run_in_thread(self) -> None:
        """后台线程的主函数。"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception as e:
            logger.error(f"[GatewayManager] Error in main loop: {e}")
        finally:
            self._loop.close()

    async def _main(self) -> None:
        """主协程：启动 Gateway 和 Agent Worker。"""
        try:
            await self.start()
            # 启动成功后清除之前的错误
            with self._error_lock:
                self._startup_error = None
            # 保持运行
            while self._running:
                await asyncio.sleep(1)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[GatewayManager] Main loop error: {e}")
            with self._error_lock:
                self._startup_error = error_msg
        finally:
            await self.stop()

    async def start(self) -> None:
        """启动 Gateway 和本地 Agent Worker。"""
        if self._running:
            return

        logger.info("[GatewayManager] Starting...")

        # 1. 创建并启动 Gateway
        from middleware.im.gateway import (
            Gateway,
            set_gateway,
        )

        self._gateway = Gateway(
            websocket_host=self.websocket_host,
            websocket_port=self.websocket_port,
            webhook_host=self.webhook_host,
            webhook_port=self.webhook_port,
        )
        set_gateway(self._gateway)
        await self._gateway.start()

        logger.info(
            f"[GatewayManager] Gateway started: "
            f"ws://{self.websocket_host}:{self.websocket_port}, "
            f"http://{self.webhook_host}:{self.webhook_port}"
        )

        # 2. 启动本地 Agent Worker
        await self._start_agent_worker()

        # 3. 启动配置的 IM 渠道
        await self._start_channels()

        self._running = True
        logger.info("[GatewayManager] Started successfully")

    async def stop(self) -> None:
        """停止 Gateway 和所有渠道。"""
        if not self._running:
            return

        logger.info("[GatewayManager] Stopping...")

        # 1. 停止所有渠道
        await self._stop_all_channels()

        # 2. 停止 Agent Worker
        if self._agent_worker:
            try:
                await self._agent_worker.stop()
            except Exception as e:
                logger.error(f"[GatewayManager] Error stopping Agent Worker: {e}")
            self._agent_worker = None

        # 3. 停止 Gateway
        if self._gateway:
            try:
                await self._gateway.stop()
            except Exception as e:
                logger.error(f"[GatewayManager] Error stopping Gateway: {e}")
            self._gateway = None

        self._running = False
        logger.info("[GatewayManager] Stopped successfully")

    async def _start_agent_worker(self) -> None:
        """启动本地 Agent Worker，连接到 Gateway。"""
        from im.gateway.agent_worker import GatewayAgentWorker

        self._agent_worker = GatewayAgentWorker(
            gateway_url=f"ws://{self.websocket_host}:{self.websocket_port}"
        )
        await self._agent_worker.start()
        logger.info("[GatewayManager] Agent Worker started")

    async def _start_channels(self) -> None:
        """根据配置启动 IM 渠道。"""
        from middleware.config import g_config
        from middleware.im.gateway import ChannelType, ConnectionMode, PermissionDomain

        config = g_config.load()

        # 飞书
        feishu_cfg = (
            config.im.feishu
            if hasattr(config, "im") and hasattr(config.im, "feishu")
            else None
        )
        if feishu_cfg and feishu_cfg.enabled:
            try:
                await self._gateway.startAccount(
                    account_id="feishu_main",
                    channel_type=ChannelType.FEISHU,
                    credentials={
                        "app_id": feishu_cfg.app_id,
                        "app_secret": feishu_cfg.app_secret,
                        "encrypt_key": feishu_cfg.encrypt_key or "",
                        "verification_token": feishu_cfg.verification_token or "",
                    },
                    mode=ConnectionMode.WEBSOCKET,
                    permission_domain=PermissionDomain.NODE,
                )
                logger.info("[GatewayManager] Feishu channel started")
                _publish_im_event(EventType.IM_SERVICE_STARTED, "feishu")
            except Exception as e:
                logger.error(f"[GatewayManager] Failed to start Feishu: {e}")
                _publish_im_event(EventType.IM_SERVICE_START_FAILED, "feishu", str(e))

        # 钉钉
        dingtalk_cfg = (
            config.im.dingtalk
            if hasattr(config, "im") and hasattr(config.im, "dingtalk")
            else None
        )
        if dingtalk_cfg and dingtalk_cfg.enabled:
            try:
                await self._gateway.startAccount(
                    account_id="dingtalk_main",
                    channel_type=ChannelType.DINGTALK,
                    credentials={},  # 凭证从配置文件读取
                    mode=ConnectionMode.WEBSOCKET,
                    permission_domain=PermissionDomain.NODE,
                )
                logger.info("[GatewayManager] DingTalk channel started")
                _publish_im_event(EventType.IM_SERVICE_STARTED, "dingtalk")
            except Exception as e:
                logger.error(f"[GatewayManager] Failed to start DingTalk: {e}")
                _publish_im_event(EventType.IM_SERVICE_START_FAILED, "dingtalk", str(e))

        # 企业微信
        wecom_cfg = (
            config.im.wecom
            if hasattr(config, "im") and hasattr(config.im, "wecom")
            else None
        )
        if wecom_cfg and wecom_cfg.enabled:
            try:
                await self._gateway.startAccount(
                    account_id="wecom_main",
                    channel_type=ChannelType.WECOM,
                    credentials={},  # 凭证从配置文件读取
                    mode=ConnectionMode.WEBSOCKET,
                    permission_domain=PermissionDomain.NODE,
                )
                logger.info("[GatewayManager] WeCom channel started")
                _publish_im_event(EventType.IM_SERVICE_STARTED, "wecom")
            except Exception as e:
                logger.error(f"[GatewayManager] Failed to start WeCom: {e}")
                _publish_im_event(EventType.IM_SERVICE_START_FAILED, "wecom", str(e))

        # 微信（个人号）
        wechat_cfg = (
            config.im.wechat
            if hasattr(config, "im") and hasattr(config.im, "wechat")
            else None
        )
        if wechat_cfg and wechat_cfg.enabled:
            try:
                # 直接从配置读取（扁平结构，与其他平台一致）
                base_url = getattr(
                    wechat_cfg, "base_url", "https://ilinkai.weixin.qq.com"
                )
                token = getattr(wechat_cfg, "token", "")

                if token:
                    await self._gateway.startAccount(
                        account_id="wechat_main",
                        channel_type=ChannelType.WECHAT,
                        credentials={
                            "base_url": base_url,
                            "token": token,
                        },
                        mode=ConnectionMode.POLLING,
                        permission_domain=PermissionDomain.NODE,
                    )
                    logger.info("[GatewayManager] WeChat channel started")
                    _publish_im_event(EventType.IM_SERVICE_STARTED, "wechat")
                else:
                    logger.warning(
                        "[GatewayManager] WeChat enabled but token not configured"
                    )
            except Exception as e:
                logger.error(f"[GatewayManager] Failed to start WeChat: {e}")
                _publish_im_event(EventType.IM_SERVICE_START_FAILED, "wechat", str(e))

    async def _stop_all_channels(self) -> None:
        """停止所有 IM 渠道。"""
        if not self._gateway:
            return

        logger.info("[GatewayManager] Stopping all IM channels...")

        # 停止所有账户
        try:
            await self._gateway.stopAll()
            logger.info("[GatewayManager] All IM channels stopped")
        except Exception as e:
            logger.error(f"[GatewayManager] Error stopping channels: {e}")

    async def refresh_channels(self) -> None:
        """根据当前配置刷新 IM 渠道（启动新启用的，停止已禁用的）。

        当用户在 GUI 中切换 IM 平台开关时调用。

        Returns:
            dict: 各平台的操作结果，格式：
            {
                "feishu": {"action": "started", "success": True, "error": None},
                "dingtalk": {"action": "stopped", "success": True, "error": None},
            }
        """
        results = {}

        # 防止并发刷新
        if not self._running or not self._gateway:
            logger.warning("[GatewayManager] Cannot refresh channels: not running")
            return results

        from middleware.config import g_config
        from middleware.im.gateway import ChannelType, ConnectionMode, PermissionDomain

        config = g_config.load()

        # 检查总开关（Gateway enabled）是否开启
        gateway_enabled = (
            getattr(config.gateway, "enabled", False)
            if hasattr(config, "gateway")
            else False
        )
        logger.info(f"[GatewayManager] Gateway enabled={gateway_enabled}")

        # 定义平台配置映射
        platform_configs = [
            ("feishu", ChannelType.FEISHU, "feishu_main"),
            ("dingtalk", ChannelType.DINGTALK, "dingtalk_main"),
            ("wecom", ChannelType.WECOM, "wecom_main"),
            ("wechat", ChannelType.WECHAT, "wechat_main"),  # ✅ 添加微信支持
        ]

        for platform_name, channel_type, account_id in platform_configs:
            platform_cfg = getattr(config.im, platform_name, None)

            if not platform_cfg:
                continue

            # 检查该账户是否已启动
            is_running = account_id in getattr(
                self._gateway.account_manager, "_accounts", {}
            )

            # 综合判断：渠道启用需要同时满足 Gateway 总开关开启 AND 平台开关开启
            # 微信使用 accounts 列表，检查第一个账户
            if platform_name == "wechat":
                platform_enabled = (
                    platform_cfg.get("enabled", False)
                    if isinstance(platform_cfg, dict)
                    else getattr(platform_cfg, "enabled", False)
                )
            else:
                platform_enabled = platform_cfg.enabled

            effective_enabled = gateway_enabled and platform_enabled

            logger.info(
                f"[GatewayManager] {platform_name} status: gateway_enabled={gateway_enabled}, "
                f"platform_enabled={platform_enabled}, effective_enabled={effective_enabled}, running={is_running}"
            )

            if effective_enabled and not is_running:
                # 需要启动
                logger.info(f"[GatewayManager] ▶ Starting {platform_name}...")
                try:
                    logger.info(
                        f"[GatewayManager] >>> Calling startAccount: "
                        f"account_id={account_id}, channel={channel_type.value}, "
                        f"mode=POLLING"
                    )
                    credentials = {}
                    if platform_name == "feishu":
                        credentials = {
                            "app_id": platform_cfg.app_id or "",
                            "app_secret": platform_cfg.app_secret or "",
                            "encrypt_key": platform_cfg.encrypt_key or "",
                            "verification_token": platform_cfg.verification_token or "",
                        }
                    elif platform_name == "dingtalk":
                        credentials = {
                            "app_key": platform_cfg.app_key or "",
                            "app_secret": platform_cfg.app_secret or "",
                            "webhook_url": platform_cfg.webhook_url or "",
                        }
                    elif platform_name == "wecom":
                        credentials = {
                            "corp_id": platform_cfg.corp_id or "",
                            "agent_id": platform_cfg.agent_id or "",
                            "secret": platform_cfg.secret or "",
                        }
                    elif platform_name == "wechat":
                        # 微信配置（扁平结构，与其他平台一致）
                        if isinstance(platform_cfg, dict):
                            credentials = {
                                "base_url": platform_cfg.get(
                                    "base_url", "https://ilinkai.weixin.qq.com"
                                ),
                                "token": platform_cfg.get("token", ""),
                            }
                        else:
                            credentials = {
                                "base_url": getattr(
                                    platform_cfg,
                                    "base_url",
                                    "https://ilinkai.weixin.qq.com",
                                ),
                                "token": getattr(platform_cfg, "token", ""),
                            }

                    await self._gateway.startAccount(
                        account_id=account_id,
                        channel_type=channel_type,
                        credentials=credentials,
                        mode=ConnectionMode.POLLING
                        if platform_name == "wechat"
                        else ConnectionMode.WEBSOCKET,
                        permission_domain=PermissionDomain.NODE,
                    )
                    logger.info(f"[GatewayManager] startAccount returned successfully")

                    # 验证启动成功
                    is_now_running = account_id in getattr(
                        self._gateway.account_manager, "_accounts", {}
                    )
                    if is_now_running:
                        logger.info(
                            f"[GatewayManager] ✓ {platform_name} started successfully"
                        )
                        _publish_im_event(EventType.IM_SERVICE_STARTED, platform_name)
                        results[platform_name] = {
                            "action": "started",
                            "success": True,
                            "error": None,
                        }
                    else:
                        raise RuntimeError("Account not found after start")

                except Exception as e:
                    import traceback

                    error_msg = f"[GatewayManager] ✗ Failed to start {platform_name}: {e}\n{traceback.format_exc()}"
                    logger.error(error_msg)
                    _publish_im_event(
                        EventType.IM_SERVICE_START_FAILED, platform_name, str(e)
                    )
                    results[platform_name] = {
                        "action": "start_failed",
                        "success": False,
                        "error": str(e),
                    }

            elif not effective_enabled and is_running:
                # 需要停止
                logger.info(f"[GatewayManager] ▶ Stopping {platform_name}...")
                try:
                    await self._gateway.stopAccount(account_id)

                    # 验证停止成功
                    is_still_running = account_id in getattr(
                        self._gateway.account_manager, "_accounts", {}
                    )
                    if not is_still_running:
                        logger.info(
                            f"[GatewayManager] ✓ {platform_name} stopped successfully"
                        )
                        _publish_im_event(EventType.IM_SERVICE_STOPPED, platform_name)
                        results[platform_name] = {
                            "action": "stopped",
                            "success": True,
                            "error": None,
                        }
                    else:
                        raise RuntimeError("Account still exists after stop")

                except Exception as e:
                    import traceback

                    error_msg = f"{type(e).__name__}: {e}"
                    logger.error(
                        f"[GatewayManager] ✗ Failed to stop {platform_name}: {error_msg}\n{traceback.format_exc()}"
                    )
                    _publish_im_event(
                        EventType.IM_SERVICE_STOP_FAILED, platform_name, error_msg
                    )
                    results[platform_name] = {
                        "action": "stop_failed",
                        "success": False,
                        "error": error_msg,
                    }
            else:
                # 状态已符合，无需操作
                status = "running" if is_running else "stopped"
                logger.debug(
                    f"[GatewayManager] {platform_name} already {status}, no action needed"
                )
                results[platform_name] = {
                    "action": "none",
                    "success": True,
                    "status": status,
                }

        # 打印汇总结果
        self._log_refresh_summary(results)
        return results

    def _log_refresh_summary(self, results: dict) -> None:
        """打印刷新结果汇总"""
        if not results:
            return

        started = [p for p, r in results.items() if r.get("action") == "started"]
        stopped = [p for p, r in results.items() if r.get("action") == "stopped"]
        failed = [p for p, r in results.items() if "failed" in r.get("action", "")]
        unchanged = [p for p, r in results.items() if r.get("action") == "none"]

        summary_parts = []
        if started:
            summary_parts.append(f"启动: {', '.join(started)}")
        if stopped:
            summary_parts.append(f"停止: {', '.join(stopped)}")
        if failed:
            summary_parts.append(f"失败: {', '.join(failed)}")
        if unchanged:
            summary_parts.append(f"未变更: {len(unchanged)}个")

        if summary_parts:
            logger.info(f"[GatewayManager] 刷新结果 - {'; '.join(summary_parts)}")

        if failed:
            for platform in failed:
                error = results[platform].get("error", "Unknown error")
                logger.error(f"[GatewayManager] {platform} 失败原因: {error}")

    def refresh_channels_sync(self) -> None:
        """同步接口：刷新 IM 渠道（供非异步代码调用）。"""
        if not self._loop or not self._running:
            logger.warning("[GatewayManager] Cannot refresh channels: not running")
            return

        # 在后台线程中执行刷新并等待结果
        import concurrent.futures

        future = asyncio.run_coroutine_threadsafe(self.refresh_channels(), self._loop)
        logger.info("[GatewayManager] Channel refresh scheduled")

        # 等待结果（带超时）
        try:
            results = future.result(timeout=30)  # 最多等30秒，给渠道停止足够时间
            if results:
                # 打印简化结果
                actions = [
                    f"{p}:{r['action']}"
                    for p, r in results.items()
                    if r.get("action") != "none"
                ]
                if actions:
                    logger.info(
                        f"[GatewayManager] Refresh completed: {', '.join(actions)}"
                    )
        except concurrent.futures.TimeoutError:
            logger.warning("[GatewayManager] Channel refresh timeout")
        except Exception as e:
            logger.error(f"[GatewayManager] Channel refresh error: {e}")


def start_gateway_if_configured() -> None:
    """根据配置启动 Gateway 模式（替代 Bridge 模式）。

    在 bootstrap 或 GUI 初始化时调用。
    """
    global _gateway_manager

    from middleware.config import g_config

    config = g_config.load()

    # 检查是否启用 Gateway 模式
    gateway_enabled = False
    if hasattr(config, "gateway"):
        gateway_enabled = getattr(config.gateway, "enabled", False)

    if not gateway_enabled:
        logger.info("[GatewayManager] Gateway mode not enabled, skipping")
        return

    # 检查是否有任何 IM 渠道启用
    # 注意：允许在没有 IM 渠道时启动 Gateway，以便后续动态启用
    has_im = False
    if hasattr(config, "im"):
        # 检查各平台是否启用（统一格式）
        if hasattr(config.im, "feishu"):
            feishu_cfg = config.im.feishu
            if hasattr(feishu_cfg, "enabled") and feishu_cfg.enabled:
                has_im = True

        if hasattr(config.im, "dingtalk"):
            dingtalk_cfg = config.im.dingtalk
            if hasattr(dingtalk_cfg, "enabled") and dingtalk_cfg.enabled:
                has_im = True

        if hasattr(config.im, "wecom"):
            wecom_cfg = config.im.wecom
            if hasattr(wecom_cfg, "enabled") and wecom_cfg.enabled:
                has_im = True

        # 检查微信（支持 dict 格式）
        if hasattr(config.im, "wechat"):
            wechat_cfg = config.im.wechat
            if isinstance(wechat_cfg, dict):
                if wechat_cfg.get("enabled", False):
                    has_im = True
                    logger.info("[GatewayManager] WeChat channel enabled (dict format)")
            elif hasattr(wechat_cfg, "enabled") and wechat_cfg.enabled:
                has_im = True
                logger.info("[GatewayManager] WeChat channel enabled")

    if has_im:
        logger.info("[GatewayManager] Found enabled IM channels, will start them")
    else:
        logger.info(
            "[GatewayManager] No IM channels enabled at startup, will wait for dynamic enable"
        )

    # 获取 Gateway 配置
    ws_host = "127.0.0.1"
    ws_port = 8765
    wh_host = "127.0.0.1"
    wh_port = 18080

    if hasattr(config, "gateway"):
        ws_host = getattr(config.gateway, "websocket_host", ws_host)
        ws_port = getattr(config.gateway, "websocket_port", ws_port)
        wh_host = getattr(config.gateway, "webhook_host", wh_host)
        wh_port = getattr(config.gateway, "webhook_port", wh_port)

    # 创建并启动 GatewayManager
    _gateway_manager = GatewayManager(
        websocket_host=ws_host,
        websocket_port=ws_port,
        webhook_host=wh_host,
        webhook_port=wh_port,
    )
    _gateway_manager.start_in_background()

    logger.info("[GatewayManager] Gateway mode started in background")


def stop_gateway() -> None:
    """停止 Gateway。"""
    global _gateway_manager
    if _gateway_manager:
        # 由于 Gateway 运行在后台线程，这里只是标记停止
        _gateway_manager._running = False
        _gateway_manager = None
