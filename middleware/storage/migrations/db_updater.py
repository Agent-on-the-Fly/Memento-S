from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

from alembic import command
from alembic.config import Config

logger = logging.getLogger(__name__)


def _resolve_runtime_root() -> Path:
    """Resolve runtime project root for both dev and packaged environments.

    Dev mode:
        middleware/storage/migrations/db_updater.py -> parents[3] => project root
    Packaged mode:
        use sys._MEIPASS as extraction/runtime root
    """
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", None)
        if not base:
            raise RuntimeError("Packaged runtime detected but sys._MEIPASS is missing")
        return Path(base).resolve()

    return Path(__file__).resolve().parents[3]


def run_auto_upgrade(db_url: str) -> None:
    """Programmatically run Alembic upgrade to head.

    Must be called before UI startup.
    """
    root = _resolve_runtime_root()

    # In both dev and packaged layout, files live under middleware/storage/migrations
    alembic_ini = root / "middleware" / "storage" / "migrations" / "alembic.ini"
    script_location = root / "middleware" / "storage" / "migrations"

    try:
        if not alembic_ini.exists():
            raise FileNotFoundError(f"alembic.ini not found: {alembic_ini}")
        if not script_location.exists():
            raise FileNotFoundError(f"migrations dir not found: {script_location}")

        alembic_cfg = Config(str(alembic_ini))

        # Critical for bundled mode: force absolute script path
        alembic_cfg.set_main_option("script_location", str(script_location))
        alembic_cfg.set_main_option("sqlalchemy.url", db_url)

        logger.info("Running DB migration to head")
        logger.info("Alembic INI: {}", alembic_ini)
        logger.info("Script location: {}", script_location)
        logger.info("DB URL: {}", db_url)

        command.upgrade(alembic_cfg, "head")
        logger.info("DB migration completed")

    except Exception as exc:
        logger.error("Database migration failed: {}", exc)
        logger.error("Traceback:\n{}", traceback.format_exc())
        # re-raise so caller can decide stop/start fallback policy
        raise
