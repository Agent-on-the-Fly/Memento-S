# SPDX-License-Identifier: Apache-2.0
"""
Memento-S 启动引导（Bootstrap）

职责：
1. 配置系统初始化（ConfigManager 单例）
2. 配置版本迁移（ConfigMigrator）
3. 日志系统初始化（Loguru）
4. 数据库初始化（DatabaseManager 单例 + 表创建）
5. 数据库迁移检测和执行
6. 目录结构初始化
7. Skill 系统初始化（包含孤儿清理）
8. 所有全局单例的一次性初始化
"""

from __future__ import annotations

# Suppress litellm logging before importing litellm
import os

os.environ["LITELLM_LOG"] = "WARNING"

# Configure SSL certificates for HTTPS requests (important for packaged apps)
try:
    import certifi
    import ssl

    # Set the SSL certificate bundle path
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    # Create default SSL context with certifi certificates
    ssl._create_default_https_context = ssl.create_default_context(
        cafile=certifi.where()
    )
except ImportError:
    pass

import asyncio
import json
import os
import threading
import traceback
from pathlib import Path
from typing import Any

# 防止飞书长链接被重复启动（bootstrap + 手动 feishu 命令共用）
_feishu_bridge_started: bool = False

# Skill 后台初始化状态（全局单例）
_skill_sync_started: bool = False
_skill_sync_lock = threading.Lock()

try:
    from dotenv import load_dotenv

    # 始终从项目根目录加载 .env，不依赖当前工作目录
    _dotenv_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=_dotenv_path, override=False)
except ImportError:
    pass

from middleware.config.config_manager import ConfigManager, GlobalConfig, g_config
from middleware.config.bootstrap import bootstrap_configs
from middleware.config.migrations import (
    ConfigMigrator,
    MigrationResult,
    merge_template_defaults,
)
from middleware.config.mcp_config_manager import McpConfigManager, g_mcp_config_manager
from middleware.storage.core.engine import DatabaseManager, get_db_manager
from middleware.storage.migrations.db_updater import run_migrations_to_head
from middleware.storage.models import Base
from utils.logger import setup_logger, logger

def _init_logging(config: GlobalConfig, enable_console: bool = False) -> None:
    """初始化日志系统（Loguur）。

    Args:
        config: 全局配置
        enable_console: 是否启用控制台输出（Electron 模式下应设为 True）
    """
    log_level = config.logging.level
    setup_logger(
        console_level="DEBUG",  # 控制台默认显示DEBUG级别
        file_level=log_level,  # 文件级别跟随全局配置
        rotation="00:00",
        retention="30 days",
        daily_separate=True,
        enable_console=enable_console,
    )


def _get_bundled_uv_path() -> Path | None:
    """查找打包在应用内的 uv 二进制文件。

    根据 RuntimeMode 选择查找策略：
    - PRODUCTION: 从 sys._MEIPASS、Electron Resources 或可执行文件所在目录查找
    - DEV: 从项目根目录向上查找

    Returns:
        uv 可执行文件的路径，如果不存在则返回 None
    """
    import platform
    from utils.runtime_mode import get_runtime_mode, RuntimeMode

    _OS_TAG_MAP = {"Windows": "windows", "Darwin": "macos", "Linux": "linux"}
    uv_name = "uv.exe" if platform.system() == "Windows" else "uv"
    os_tag = _OS_TAG_MAP[platform.system()]

    mode = get_runtime_mode()

    if mode == RuntimeMode.PRODUCTION:
        # PRODUCTION: 支持多种打包模式
        # 1. 优先从 Electron Resources 查找（通过环境变量）
        # Electron 启动时会设置 ELECTRON_RESOURCES_PATH
        electron_resources = os.environ.get("ELECTRON_RESOURCES_PATH")
        if electron_resources:
            candidate = Path(electron_resources) / "assets" / "resource" / os_tag / uv_name
            if candidate.exists():
                return candidate

        # 2. PyInstaller 模式: 资源在 sys._MEIPASS
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidate = Path(meipass) / "assets" / "resource" / os_tag / uv_name
            if candidate.exists():
                return candidate

        # 3. 兜底：从可执行文件所在目录查找
        if sys.executable:
            exe_path = Path(sys.executable)
            candidate = exe_path.parent / "assets" / "resource" / os_tag / uv_name
            if candidate.exists():
                return candidate
    else:
        # DEV: 从项目根向上查找
        candidate = Path(__file__).resolve().parent / "assets" / "resource" / os_tag / uv_name
        if candidate.exists():
            return candidate

    return None


def _check_uv_installation() -> None:
    """检查 uv 是否可用（优先使用打包的 bundled uv，再查系统 PATH）。

    如果找到 bundled uv，将其所在目录前置注入 os.environ["PATH"]，
    使得后续 shutil.which("uv") 在整个进程内均可找到。

    Raises:
        RuntimeError: 如果 uv 既未打包也未安装在系统中
    """
    import os
    import shutil
    import sys

    # 1. 优先使用打包内置的 uv
    bundled = _get_bundled_uv_path()
    if bundled:
        bin_dir = str(bundled.parent)
        current_path = os.environ.get("PATH", "")
        if bin_dir not in current_path:
            os.environ["PATH"] = bin_dir + os.pathsep + current_path
        from utils.logger import logger

        logger.info(f"[bootstrap] uv found (bundled): {bundled}")
        return

    # 2. 回退到系统已安装的 uv
    uv_path = shutil.which("uv")
    if uv_path:
        from utils.logger import logger

        logger.info(f"[bootstrap] uv found (system): {uv_path}")
        return

    # 3. 都找不到，报错
    error_msg = (
        "\n" + "=" * 70 + "\n"
        "UV NOT INSTALLED\n"
        "=" * 70 + "\n"
        "The sandbox_provider is set to 'uv', but uv is not installed.\n"
        "\n"
        "To install uv:\n"
        "  macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh\n"
        '  Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"\n'
        "\n"
        "Or visit: https://github.com/astral-sh/uv\n"
        "\n"
        "After installation, restart the application.\n"
        "=" * 70 + "\n"
    )
    print(error_msg, file=sys.stderr)
    raise RuntimeError("uv is not installed")


def _check_db_migration_status(db_url: str) -> tuple[bool, str | None, str | None]:
    """检查数据库是否需要迁移。

    Returns:
        tuple: (是否需要迁移, 当前版本, 最新版本)
    """
    from alembic.config import Config
    from alembic.script import ScriptDirectory
    from alembic.runtime.migration import MigrationContext
    from sqlalchemy import create_engine
    import sys

    # 解析项目根目录（根据 RuntimeMode）
    from utils.runtime_mode import get_runtime_mode, RuntimeMode

    mode = get_runtime_mode()
    if mode == RuntimeMode.PRODUCTION:
        root = Path(getattr(sys, "_MEIPASS", Path.cwd()))
    else:
        root = Path(__file__).resolve().parent

    alembic_ini = root / "middleware" / "storage" / "migrations" / "alembic.ini"
    script_location = root / "middleware" / "storage" / "migrations"

    if not alembic_ini.exists():
        return False, None, None

    alembic_cfg = Config(str(alembic_ini))
    alembic_cfg.set_main_option("script_location", str(script_location))
    alembic_cfg.set_main_option("sqlalchemy.url", db_url)

    try:
        # 创建同步引擎来检查版本
        sync_url = db_url.replace("+aiosqlite", "")
        engine = create_engine(sync_url)

        with engine.connect() as connection:
            context = MigrationContext.configure(connection)
            current_rev = context.get_current_revision()

        # 获取最新版本
        script = ScriptDirectory.from_config(alembic_cfg)
        head_rev = script.get_current_head()

        needs_migration = current_rev != head_rev

        return needs_migration, current_rev, head_rev

    except Exception:
        # 如果无法获取版本（新数据库），需要执行迁移
        return True, None, "head"


def _run_db_migration(db_url: str) -> None:
    """执行数据库迁移。"""
    run_migrations_to_head(db_url=db_url)


async def _init_database(manager: ConfigManager) -> None:
    """初始化数据库（DatabaseManager 单例 + 表创建）。"""
    # 使用 from_config 确保单例被初始化（协程安全）
    db_manager = await DatabaseManager.from_config(
        db_url=manager.get_db_url(),
        echo=False,
    )

    # 创建所有表
    async with db_manager.engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def _sync_skills() -> None:
    """执行 Skill 系统初始化（bootstrap 入口）：

    1. 检测 builtin/skills 和用户目录的 skills 是否有丢失，复制丢失的 builtin skills
    2. 扫描用户的 skills 目录，同步到 db 中
    3. 扫描 db 中存储的 skills，删除用户 skills 目录下已不存在的记录
    4. 初始化 Skill 系统
    """
    from core.skill import init_skill_system
    from shared.schema import SkillConfig

    # 从全局配置创建 SkillConfig
    config = SkillConfig.from_global_config()

    # 初始化技能系统（包含完整的 builtin skills 同步和注册表初始化）
    await init_skill_system(config)


def _print_bootstrap_info(
    config_dir: Path,
    config_file: Path,
    dirs: dict[str, Path],
    manager: ConfigManager,
    mcp_manager: McpConfigManager,
    config_migration: MigrationResult | None = None,
    db_migration: tuple[bool, str | None, str | None] | None = None,
) -> None:
    """打印启动引导信息。"""
    logger.info(f"[bootstrap] config dir ready: {config_dir}")
    logger.info(f"[bootstrap] config file ready: {config_file}")

    if config_migration and config_migration.migrated:
        logger.info(
            f"[bootstrap] config migrated: {config_migration.old_version} -> {config_migration.new_version}"
        )
        if config_migration.backup_path:
            logger.info(
                f"[bootstrap] old config backup: {config_migration.backup_path}"
            )
        if config_migration.changes:
            logger.info(
                f"[bootstrap] detected changes: {len(config_migration.changes)}"
            )

    # 打印数据库迁移信息
    if db_migration:
        needs, current_rev, head_rev = db_migration
        if needs:
            logger.info(
                f"[bootstrap] db migration: {current_rev or 'None'} -> {head_rev}"
            )
        else:
            logger.info(f"[bootstrap] db version: {current_rev} (up to date)")

    logger.info(f"[bootstrap] workspace dir ready: {dirs['workspace']}")
    logger.info(f"[bootstrap] db path ready: {dirs['db']}")
    logger.info(f"[bootstrap] db url: {manager.get_db_url()}")
    logger.info(f"[bootstrap] skills dir ready: {dirs['skills']}")
    logger.info(f"[bootstrap] logs dir ready: {dirs['logs']}")
    logger.info(f"[bootstrap] context dir ready: {dirs.get('context')}")
    logger.info(f"[bootstrap] mcp config file ready: {mcp_manager.mcp_config_path}")
    servers = mcp_manager.get_servers()
    logger.info(f"[bootstrap] mcp servers: {list(servers.keys())}")
    logger.info("[bootstrap] all singletons initialized: OK")
    logger.info("[bootstrap] config validation: OK")


async def bootstrap(background_skill_sync: bool = True) -> ConfigManager:
    """执行完整的启动引导流程。

    Args:
        background_skill_sync: 是否将 skill 同步放到后台线程执行（默认开启）

    Returns:
        配置管理器实例（已加载并验证配置）

    Raises:
        RuntimeError: 如果初始化失败
    """
    try:
        # ========== 阶段 1: 配置系统初始化（委托给 config/bootstrap.py）==========
        global g_config
        config, _, _, config_migration = await bootstrap_configs()
        config_dir = g_config.user_config_dir
        config_file = g_config.user_config_path

        # 校验必要配置
        if config.paths.workspace_dir is None:
            raise ValueError("paths.workspace_dir 不应为空，请检查配置补全逻辑")

        # ========== 阶段 2: 目录结构初始化（已在 bootstrap_configs 中完成）==========
        dirs = {
            "workspace": config.paths.workspace_dir,
            "skills": config.paths.skills_dir,
            "db": config.paths.db_dir,
            "logs": config.paths.logs_dir,
            "context": config.paths.context_dir,
        }

        # ========== 阶段 2.5: MCP 配置初始化（已在 bootstrap_configs 中完成）==========
        mcp_config = g_mcp_config_manager.get_mcp_config()

        # ========== 阶段 3: 日志系统初始化 ==========
        # 注：setup_logger 有防重复机制，如果 Electron GUI 已调用则跳过
        _init_logging(config, enable_console=True)

        # 导入 logger 用于后续日志记录
        from utils.logger import logger

        logger.info("[bootstrap] phase 1: config system initialized")
        logger.info(f"[bootstrap] config version: {config.version}")

        if config_migration and config_migration.migrated:
            logger.info(
                f"[bootstrap] config migrated: {config_migration.old_version} -> {config_migration.new_version}"
            )
            logger.info(f"[bootstrap] backup created: {config_migration.backup_path}")

        # ========== 阶段 4: 数据库迁移检测和执行 ==========
        db_url = g_config.get_db_url()
        db_migration_status = _check_db_migration_status(db_url)
        needs_db_migration, current_rev, head_rev = db_migration_status

        if needs_db_migration:
            logger.info(
                f"[bootstrap] db migration needed: {current_rev or 'None'} -> {head_rev}"
            )
            try:
                _run_db_migration(db_url)
                logger.info("[bootstrap] db migration completed successfully")
            except Exception as e:
                logger.error(f"[bootstrap] db migration failed: {e}")
                logger.error(f"[bootstrap] traceback: \n{traceback.format_exc()}")
                raise RuntimeError(f"数据库迁移失败: {e}") from e
        else:
            logger.info(f"[bootstrap] db version: {current_rev} (up to date)")

        # ========== 阶段 5: 数据库初始化 ==========
        try:
            await _init_database(g_config)
            logger.info("[bootstrap] phase 3: database connection initialized")
        except Exception as e:
            logger.error(f"[bootstrap] database initialization failed: {e}")
            logger.error(f"[bootstrap] traceback: \n{traceback.format_exc()}")
            raise RuntimeError(f"数据库初始化失败: {e}") from e

        # ========== 阶段 6: uv 环境检查 ==========
        if config.skills.execution.sandbox_provider == "uv":
            _check_uv_installation()

        # ========== 阶段 6.5: 工具系统初始化 ==========
        # 必须在此处初始化 tools registry，否则 SkillAgent 无法获得任何 atomic tools
        try:
            from tools import bootstrap as tools_bootstrap

            await tools_bootstrap(mcp_config=mcp_config)
            logger.info("[bootstrap] phase 6.5: tools system initialized")
        except Exception as e:
            logger.error(f"[bootstrap] tools bootstrap failed: {e}")
            logger.error(f"[bootstrap] traceback: \n{traceback.format_exc()}")
            raise RuntimeError(f"工具系统初始化失败: {e}") from e

        # ========== 阶段 7: Skill 同步（三步同步）==========
        if background_skill_sync:
            _start_skill_sync_in_background()
        else:
            try:
                logger.info("[bootstrap] phase 7: syncing skills...")

                # 执行三步同步
                await _sync_skills()
                logger.info("[bootstrap] skill sync completed successfully")
            except Exception as e:
                logger.error(f"[bootstrap] skill sync failed: {e}")
                logger.error(f"[bootstrap] traceback: \n{traceback.format_exc()}")
                # skill 同步失败不是致命的，继续启动
                logger.warning("[bootstrap] continuing without skill sync...")

        # ========== 阶段 8: 打印启动信息 ==========
        _print_bootstrap_info(
            config_dir=config_dir,
            config_file=config_file,
            dirs=dirs,
            manager=g_config,

            mcp_manager=g_mcp_config_manager,
            config_migration=config_migration,
            db_migration=db_migration_status,
        )

        logger.info("[bootstrap] all phases completed successfully")

        # ========== 阶段 7: 启动 IM Gateway ==========
        # 注意：已禁用旧的 Bridge 模式，只用 Gateway 模式
        # _start_feishu_if_configured()  # 旧版 Bridge 模式，已禁用

        # 启动 Gateway 模式（统一处理所有 IM 平台）
        _start_gateway_if_configured()

        # ========== 阶段 9: 启动 DreamDaemon ==========
        _start_dream_daemon()

        # ========== 阶段 10: 确保 profile 文件存在 ==========
        _ensure_agent_profile_files()

        # ========== 阶段 11: 启动 AgentProfileEvolverDaemon ==========
        _start_agent_profile_evolver()

        # ========== 阶段 11: 启动 AutoConsolidationLoop ==========
        _start_auto_consolidation_loop()

        return g_config

    except Exception as e:
        # 捕获所有未处理的异常并打印详细信息
        import sys

        error_msg = f"[bootstrap] CRITICAL ERROR: {type(e).__name__}: {e}"
        print(error_msg, file=sys.stderr)
        print(f"[bootstrap] traceback: \n{traceback.format_exc()}", file=sys.stderr)
        # 如果 logger 已初始化，也记录到日志
        try:
            from utils.logger import logger

            logger.error(error_msg)
            logger.error(f"[bootstrap] traceback: \n{traceback.format_exc()}")
        except:
            pass
        raise


def _start_skill_sync_in_background() -> None:
    """在后台守护线程中执行 Skill 同步，避免阻塞主启动流程。"""
    global _skill_sync_started

    with _skill_sync_lock:
        if _skill_sync_started:
            logger.info("[bootstrap] phase 7: skill sync already running/skipped")
            return
        _skill_sync_started = True

    logger.info("[bootstrap] phase 7: scheduling skill sync in background thread...")

    def _run() -> None:
        try:
            asyncio.run(_sync_skills())
            logger.info("[bootstrap] background skill sync completed successfully")
        except Exception as e:
            logger.error(f"[bootstrap] background skill sync failed: {e}")
            logger.error(f"[bootstrap] traceback: \n{traceback.format_exc()}")
            logger.warning("[bootstrap] continuing without skill sync...")

    t = threading.Thread(target=_run, daemon=True, name="bootstrap-skill-sync")
    t.start()


def _start_feishu_if_configured() -> None:
    """如果配置了飞书 App 凭证，在后台启动 EndpointService。

    通过 EndpointService 统一管理所有 IM 渠道（替代原有的独立 Bridge 模式）。
    """
    global _feishu_bridge_started
    if _feishu_bridge_started:
        return

    # 检查凭证
    import json

    _feishu_cfg: dict = {}
    try:
        _cfg_path = Path.home() / "memento_s" / "config.json"
        with open(_cfg_path, "r", encoding="utf-8") as _f:
            _feishu_cfg = json.load(_f).get("im", {}).get("feishu", {})
    except Exception:
        pass
    _app_id = _feishu_cfg.get("app_id") or os.environ.get("FEISHU_APP_ID", "")
    _app_secret = _feishu_cfg.get("app_secret") or os.environ.get(
        "FEISHU_APP_SECRET", ""
    )

    if not (_app_id and _app_secret):
        logger.debug("[bootstrap] 未配置飞书凭证，跳过飞书自动启动")
        return

    import threading

    _ready = threading.Event()

    def _run() -> None:
        """在独立事件循环中运行 EndpointService"""
        try:
            from server.endpoint.im import EndpointService
            from middleware.im.gateway import ChannelType, ConnectionMode

            service = EndpointService.get_instance()
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                service.start_channel(
                    account_id="feishu_main",
                    channel_type=ChannelType.FEISHU,
                    credentials={
                        "app_id": _app_id,
                        "app_secret": _app_secret,
                        "encrypt_key": _feishu_cfg.get("encrypt_key", ""),
                        "verification_token": _feishu_cfg.get("verification_token", ""),
                    },
                    mode=ConnectionMode.WEBSOCKET,
                )
            )
            _ready.set()
            logger.info("[feishu] EndpointService 飞书渠道已在后台自动启动")
        except Exception as exc:
            logger.error(f"[feishu] 后台启动失败: {exc}", exc_info=True)
        finally:
            _ready.set()

    _feishu_bridge_started = True
    t = threading.Thread(target=_run, daemon=True, name="feishu-endpoint")
    t.start()
    _ready.wait(timeout=5)
    logger.info("[bootstrap] 飞书渠道已在后台自动启动（FEISHU_APP_ID 已配置）")


def _start_gateway_if_configured() -> None:
    """如果配置了 Gateway 模式或任意 IM 平台，在后台启动 EndpointService。

    通过 EndpointService 统一管理所有 IM 渠道（替代原有的 gateway_starter 模式）。
    只要有任何 IM 平台配置了凭证，就启动服务，确保 Electron GUI 中的开关能正常工作。
    """
    try:
        from server.endpoint.im import EndpointService
        from middleware.config import g_config

        config = g_config.load()

        # 检查是否需要启动 IM 服务
        should_start = False
        reason = ""

        # 1. Gateway 模式明确启用
        gateway_enabled = (
            getattr(config.gateway, "enabled", False)
            if hasattr(config, "gateway")
            else False
        )

        # 2. 检查各 IM 平台是否有有效凭证
        im = getattr(config, "im", None)
        has_wechat = False
        has_feishu = False
        has_dingtalk = False
        has_wecom = False

        if im:
            wc = getattr(im, "wechat", None)
            has_wechat = bool(wc and getattr(wc, "token", None))
            fs = getattr(im, "feishu", None)
            has_feishu = bool(fs and getattr(fs, "app_id", None))
            dt = getattr(im, "dingtalk", None)
            has_dingtalk = bool(dt and getattr(dt, "app_key", None))
            wc2 = getattr(im, "wecom", None)
            has_wecom = bool(wc2 and getattr(wc2, "corp_id", None))

        if gateway_enabled:
            should_start = True
            reason = "gateway.enabled=True"
        elif has_wechat:
            should_start = True
            reason = "wechat token configured"
        elif has_feishu:
            should_start = True
            reason = "feishu app_id configured"
        elif has_dingtalk:
            should_start = True
            reason = "dingtalk app_key configured"
        elif has_wecom:
            should_start = True
            reason = "wecom corp_id configured"

        if not should_start:
            logger.info("[EndpointService] No IM platform configured, skipping auto-start")
            return

        service = EndpointService.get_instance()
        if service.is_running:
            logger.info("[EndpointService] Already running")
            return

        service.start_in_background()
        logger.info(f"[EndpointService] Started in background ({reason})")
    except Exception as e:
        logger.warning(f"[bootstrap] IM Gateway 启动失败: {e}")


def _start_dream_daemon() -> None:
    """如果 dream.enabled=True，在后台启动 DreamDaemon。"""
    try:
        from middleware.config import g_config
        dream_cfg = g_config.load().dream
    except Exception:
        return

    if not dream_cfg.enabled:
        return

    from daemon.dream import DreamDaemon
    DreamDaemon.start(config=dream_cfg)
    logger.info("[bootstrap] DreamDaemon scheduled")


def _ensure_agent_profile_files() -> None:
    """确保 SOUL.md 和 USER.md 存在，不存在则用默认模板创建。"""
    try:
        from core.agent_profile import apm
        apm.ensure_files()
    except Exception:
        logger.warning("[bootstrap] ensure_profile_files failed", exc_info=True)


def _start_agent_profile_evolver() -> None:
    """启动 AgentProfileEvolverDaemon，定时 + 会话结束时进化 USER.md。"""
    try:
        from daemon.agent_profile import AgentProfileEvolverDaemon
        AgentProfileEvolverDaemon.start()
        logger.info("[bootstrap] AgentProfileEvolverDaemon scheduled")
    except Exception:
        logger.warning("[bootstrap] AgentProfileEvolverDaemon start failed", exc_info=True)


def _start_auto_consolidation_loop() -> None:
    """如果 memory.enabled=True，在后台启动 AutoConsolidationLoop。"""
    try:
        from middleware.config import g_config
        mem_cfg = g_config.load().memory
    except Exception:
        return

    if not mem_cfg.enabled:
        return

    from pathlib import Path

    cfg = g_config.load()
    context_dir: Path | None = cfg.paths.context_dir
    if context_dir is None:
        return

    memory_dir = context_dir / "memory"

    from infra.memory.consolidation import AutoConsolidationLoop, MemoryConsolidationEngine

    engine = MemoryConsolidationEngine(
        memory=memory_dir,
        config=mem_cfg,
    )
    loop = AutoConsolidationLoop(
        engine=engine,
        poll_interval_seconds=mem_cfg.poll_interval_seconds,
    )
    asyncio.create_task(loop.start())
    logger.info(
        "[bootstrap] AutoConsolidationLoop started: poll_interval={}s, "
        "min_sessions={}, min_bytes={}",
        mem_cfg.poll_interval_seconds,
        mem_cfg.min_staging_sessions,
        mem_cfg.min_staging_bytes,
    )


def bootstrap_sync(background_skill_sync: bool = True) -> ConfigManager:
    """同步版本的 bootstrap（用于非异步环境）。

    Args:
        background_skill_sync: 是否将 skill 同步放到后台线程执行（默认开启）

    Returns:
        配置管理器实例
    """
    return asyncio.run(bootstrap(background_skill_sync=background_skill_sync))


if __name__ == "__main__":
    manager = bootstrap_sync()
