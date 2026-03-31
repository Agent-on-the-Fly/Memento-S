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
from middleware.config.migrations import (
    ConfigMigrator,
    MigrationResult,
    merge_template_defaults,
)
from middleware.storage.core.engine import DatabaseManager, get_db_manager
from middleware.storage.migrations.db_updater import run_auto_upgrade
from middleware.storage.models import Base
from utils.logger import setup_logger, logger


def _init_directories(manager: ConfigManager, config: GlobalConfig) -> dict[str, Path]:
    """初始化所有必要的目录结构。

    Returns:
        包含所有创建目录路径的字典
    """
    config.paths.workspace_dir.mkdir(parents=True, exist_ok=True)
    config.paths.skills_dir.mkdir(parents=True, exist_ok=True)
    config.paths.db_dir.mkdir(parents=True, exist_ok=True)
    config.paths.logs_dir.mkdir(parents=True, exist_ok=True)

    return {
        "workspace": config.paths.workspace_dir,
        "skills": config.paths.skills_dir,
        "db": config.paths.db_dir,
        "logs": config.paths.logs_dir,
    }


def _init_logging(config: GlobalConfig, enable_console: bool = False) -> None:
    """初始化日志系统（Loguur）。

    Args:
        config: 全局配置
        enable_console: 是否启用控制台输出（GUI 模式下应设为 True）
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

    支持 PyInstaller（sys._MEIPASS）和 Nuitka / 普通运行三种模式。

    Returns:
        uv 可执行文件的路径，如果不存在则返回 None
    """
    import sys
    import platform

    uv_name = "uv.exe" if platform.system() == "Windows" else "uv"

    # PyInstaller one-file / one-dir 模式：资源被解压到 sys._MEIPASS
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidate = Path(meipass) / "resources" / "bin" / uv_name
        if candidate.exists():
            return candidate

    # Nuitka standalone 或开发环境：相对于当前文件所在目录
    candidate = Path(__file__).resolve().parent / "resources" / "bin" / uv_name
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

    # 解析项目根目录
    if getattr(sys, "frozen", False):
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
    run_auto_upgrade(db_url=db_url)


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
    from core.skill.config import SkillConfig

    # 从全局配置创建 SkillConfig
    config = SkillConfig.from_global_config()

    # 初始化技能系统（包含完整的 5 步初始化流程）
    await init_skill_system(config)


def _perform_config_migration(manager: ConfigManager) -> MigrationResult | None:
    """执行配置模板合并（无版本）。

    只在 bootstrap 阶段进行：
    - 模板新增字段会补充到用户配置
    - 用户已有字段保持不变
    - 标记为 x-managed-by: user 的字段完全由用户控制
    - 绝不覆盖用户的任何现有配置

    Args:
        manager: ConfigManager 实例

    Returns:
        迁移结果，如果无需变更则返回 None
    """
    from middleware.config.schema_meta import SchemaMetadata

    try:
        template = manager.load_user_template()
        user = manager.get_raw_user_config()
        schema = manager.load_schema()
    except FileNotFoundError:
        return None

    # 使用 Schema 元数据驱动的合并
    merged = SchemaMetadata.merge_respecting_metadata(template, user, schema)

    # 确保 gateway.enabled 为 True（强制启用 gateway 模式）
    if merged.get("gateway", {}).get("enabled") is not True:
        if "gateway" not in merged:
            merged["gateway"] = {}
        merged["gateway"]["enabled"] = True

    # 如果 llm.profiles 为空，从模板复制默认配置（为新用户添加默认模型）
    if not merged.get("llm", {}).get("profiles"):
        template_llm = template.get("llm", {})
        if template_llm.get("profiles"):
            if "llm" not in merged:
                merged["llm"] = {}
            merged["llm"]["profiles"] = template_llm["profiles"]
            merged["llm"]["active_profile"] = template_llm.get(
                "active_profile", "default"
            )

    # 检查是否有变更
    original_user_str = json.dumps(user, sort_keys=True)
    merged_str = json.dumps(merged, sort_keys=True)
    if merged_str == original_user_str:
        return None

    # 保存合并后的配置（使用直接写入方法，不依赖 load()）
    manager.save_user_config_direct(merged)

    return MigrationResult(
        migrated=True,
        old_version=str(user.get("version", "")),
        new_version=str(merged.get("version", "")),
        backup_path=None,
        changes=["配置模板合并完成"],
    )


def _ensure_config_version(manager: ConfigManager) -> None:
    """确保用户配置包含版本号标记（信息性，用于调试和问题排查）。

    如果用户配置缺少 version 字段，从 system_config.json 获取并写入。
    这是一个纯信息性标记，不触发版本化迁移逻辑。

    Args:
        manager: ConfigManager 实例
    """
    try:
        # 从 system_config 获取版本号
        system_config = manager.load_system_config()
        system_version = system_config.get("version", "0.2.0")

        # 获取当前用户配置
        user_config = manager.get_raw_user_config()

        # 如果缺少 version 字段，补充写入
        if "version" not in user_config:
            user_config["version"] = system_version
            manager.save_user_config_direct(user_config)
            logger.info(f"[bootstrap] 已添加版本号标记: {system_version}")
    except Exception as e:
        # 版本号补充失败不应阻塞启动
        logger.warning(f"[bootstrap] 版本号标记补充失败: {e}")


def _print_bootstrap_info(
    config_dir: Path,
    config_file: Path,
    dirs: dict[str, Path],
    manager: ConfigManager,
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
        # ========== 阶段 1: 配置系统初始化 ==========
        # 使用全局 g_config 实例
        global g_config

        # 确保配置目录和文件存在
        config_dir = g_config.ensure_user_config_dir()
        config_file = g_config.ensure_user_config_file()

        # 执行配置版本迁移
        config_migration = _perform_config_migration(g_config)

        # 补充版本号标记（从 system_config 获取，用于调试和问题排查）
        _ensure_config_version(g_config)

        # 加载并校验配置
        config: GlobalConfig = g_config.load()

        # 校验必要配置
        if config.paths.workspace_dir is None:
            raise ValueError("paths.workspace_dir 不应为空，请检查配置补全逻辑")

        # ========== 阶段 2: 目录结构初始化 ==========
        dirs = _init_directories(g_config, config)

        # ========== 阶段 3: 日志系统初始化 ==========
        # 注：setup_logger 有防重复机制，如果 GUI 已调用则跳过
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
            config_migration=config_migration,
            db_migration=db_migration_status,
        )

        logger.info("[bootstrap] all phases completed successfully")

        # ========== 阶段 7: 启动 IM Gateway ==========
        # 注意：已禁用旧的 Bridge 模式，只用 Gateway 模式
        # _start_feishu_if_configured()  # 旧版 Bridge 模式，已禁用

        # 启动 Gateway 模式（统一处理所有 IM 平台）
        _start_gateway_if_configured()

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
    """如果配置了飞书 App 凭证，在后台守护线程中启动长链接桥。

    - 优先从 ~/memento_s/config.json 的 im.feishu 节读取凭证，回退到环境变量。
    - 使用守护线程：主进程退出时自动结束。
    - 全局 _feishu_bridge_started 标记防止重复启动（由 feishu 命令检查）。
    - 使用顶层 im.feishu 模块启动桥接。
    """
    global _feishu_bridge_started
    if _feishu_bridge_started:
        return

    # 优先从 config 文件读取，回退到环境变量
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
        """在独立事件循环中运行飞书桥接。"""
        try:
            from im.feishu import start_feishu_bridge_background

            # 使用顶层 im 模块启动
            bridge = start_feishu_bridge_background()
            _ready.set()
            logger.info("[feishu-bridge] 飞书长链接已在后台自动启动")

            # 保持线程运行
            if bridge._bg_thread:
                bridge._bg_thread.join()

        except Exception as exc:
            logger.error(f"[feishu-bridge] 后台启动失败: {exc}", exc_info=True)
        finally:
            _ready.set()  # 确保主线程不会永久等待

    # 先设置标记，再启动线程，防止外部重复调用
    _feishu_bridge_started = True
    t = threading.Thread(target=_run, daemon=True, name="feishu-bridge")
    t.start()
    _ready.wait(timeout=5)  # 等待启动完成
    logger.info("[bootstrap] 飞书长链接已在后台自动启动（FEISHU_APP_ID 已配置）")


def _start_gateway_if_configured() -> None:
    """如果配置了 Gateway 模式，在后台启动 Gateway。

    Gateway 模式支持多平台同时运行（飞书、钉钉、企业微信等）。
    优先级高于 Bridge 模式，如果 Gateway 启用则使用 Gateway。
    """
    try:
        from middleware.im.gateway_starter import start_gateway_if_configured

        start_gateway_if_configured()
    except Exception as e:
        logger.warning(f"[bootstrap] Gateway 启动失败（可能未配置）: {e}")


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
