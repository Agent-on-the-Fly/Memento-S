"""Memento-S bootstrap — one-shot system initialisation.

Called before any CLI command or GUI startup to ensure the runtime
environment (config, database, builtin skills) is ready.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

_done = False


def bootstrap_sync() -> None:
    """Synchronous bootstrap: config → DB migration → async DB init → skill sync."""
    global _done
    if _done:
        return
    _done = True

    # ------------------------------------------------------------------
    # 1. Load global config (creates ~/.memento_s/ dirs + config.json)
    # ------------------------------------------------------------------
    from middleware.config import g_config

    try:
        g_config.load()
        logger.info("Config loaded successfully")
    except Exception:
        logger.exception("Failed to load config")
        raise

    # ------------------------------------------------------------------
    # 1b. Initialize logger from config
    # ------------------------------------------------------------------
    from utils.logger import setup_logger

    log_level = g_config.logging.level if g_config.logging else "INFO"
    setup_logger(console_level=log_level, file_level=log_level)

    # ------------------------------------------------------------------
    # 1c. Ensure runtime directories exist
    # ------------------------------------------------------------------
    from utils.path_manager import PathManager

    for dir_path in (
        PathManager.get_workspace_dir(),
        PathManager.get_skills_dir(),
        PathManager.get_db_dir(),
        PathManager.get_logs_dir(),
        PathManager.get_venv_dir(),
        PathManager.get_context_dir(),
    ):
        dir_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Run Alembic DB migrations (sync URL required)
    # ------------------------------------------------------------------
    try:
        db_path = g_config.get_db_path()
        sync_db_url = f"sqlite:///{db_path}"

        from middleware.storage.migrations.db_updater import run_auto_upgrade

        run_auto_upgrade(sync_db_url)
        logger.info("DB migration completed")
    except Exception:
        logger.exception("DB migration failed — continuing anyway")

    # ------------------------------------------------------------------
    # 3. Initialize async DatabaseManager singleton
    # ------------------------------------------------------------------
    try:
        async_db_url = g_config.get_db_url()

        from middleware.storage.core.engine import get_db_manager

        db_manager = get_db_manager()
        asyncio.run(db_manager.init(async_db_url))
        logger.info("Async DatabaseManager initialized")

        # Ensure all tables exist (no Alembic migration versions yet)
        from middleware.storage.models import Base

        async def _create_tables() -> None:
            async with db_manager.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

        asyncio.run(_create_tables())
        logger.info("Database tables ensured")
    except Exception:
        logger.exception("Async DB init failed — continuing anyway")

    # ------------------------------------------------------------------
    # 4. Sync builtin skills to workspace
    # ------------------------------------------------------------------
    try:
        from core.skill.initializer import sync_builtin_skills

        synced = sync_builtin_skills()
        if synced:
            logger.info("Synced builtin skills: %s", synced)
    except Exception:
        logger.exception("Builtin skill sync failed — continuing anyway")
