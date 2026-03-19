"""Cross-platform path manager for Memento-S."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from tempfile import gettempdir

import platformdirs


class PathManager:
    """Centralized cross-platform path provider."""

    APP_NAME = "memento_s"
    APP_AUTHOR = "memento_s"

    @classmethod
    def is_packaged_runtime(cls) -> bool:
        """Return True when running from packaged/frozen binary (double-check)."""
        frozen_flag = bool(getattr(sys, "frozen", False))
        meipass_flag = bool(getattr(sys, "_MEIPASS", None))
        env_flag = os.getenv("MEMENTO_PACKAGED", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        return frozen_flag or meipass_flag or env_flag

    @classmethod
    def _resolve_packaged_mode(cls, packaged: bool | None) -> bool:
        """Resolve runtime mode; explicit argument overrides auto-detection."""
        if packaged is not None:
            return packaged
        return cls.is_packaged_runtime()

    @classmethod
    def get_home_dir(cls, packaged: bool | None = None) -> Path:
        """Return runtime-specific home/base directory."""
        _ = cls._resolve_packaged_mode(packaged)
        return Path.home()

    @classmethod
    def get_project_root_dir(cls, packaged: bool | None = None) -> Path:
        """Return application root directory.

        - Dev runtime: Path.home()/memento_s
        - Packaged runtime: platformdirs user config directory
        """
        if cls._resolve_packaged_mode(packaged):
            return Path(platformdirs.user_config_dir(cls.APP_NAME, cls.APP_AUTHOR))
        return cls.get_home_dir(packaged=False) / cls.APP_NAME

    @classmethod
    def get_config_file(cls, packaged: bool | None = None) -> Path:
        """Return config file path under application root directory."""
        return cls.get_project_root_dir(packaged=packaged) / "config.json"

    @classmethod
    def get_data_dir(cls, packaged: bool | None = None) -> Path:
        """Return the root directory for user data (workspace, skills, etc).

        - Dev runtime: ~/memento_s
        - Packaged runtime: platformdirs user data dir
        """
        if cls._resolve_packaged_mode(packaged):
            return Path(platformdirs.user_data_dir(cls.APP_NAME, cls.APP_AUTHOR))
        return cls.get_project_root_dir(packaged=False)

    @classmethod
    def get_workspace_dir(cls, packaged: bool | None = None) -> Path:
        """Return workspace directory.

        - Dev runtime: ~/memento_s/workspace
        - Packaged runtime: platformdirs user data dir + /workspace
        """
        return cls.get_data_dir(packaged=packaged) / "workspace"

    @classmethod
    def get_skills_dir(cls, packaged: bool | None = None) -> Path:
        """Return skills directory."""
        return cls.get_data_dir(packaged=packaged) / "skills"

    @classmethod
    def get_db_dir(cls, packaged: bool | None = None) -> Path:
        """Return database directory."""
        return cls.get_data_dir(packaged=packaged) / "db"

    @classmethod
    def get_logs_dir(cls, packaged: bool | None = None) -> Path:
        """Return logs directory."""
        if cls._resolve_packaged_mode(packaged):
            return Path(platformdirs.user_log_dir(cls.APP_NAME, cls.APP_AUTHOR))
        return cls.get_project_root_dir(packaged=False) / "logs"

    @classmethod
    def get_venv_dir(cls, packaged: bool | None = None) -> Path:
        """Return uv virtual environment directory.

        Located alongside workspace directory (same parent) for uv sandbox.
        - Dev runtime: ~/memento_s/.venv
        - Packaged runtime: platformdirs user data dir + /.venv
        """
        return cls.get_data_dir(packaged=packaged) / ".venv"

    @classmethod
    def get_context_dir(cls, packaged: bool | None = None) -> Path:
        """Return context data directory (memory, daily notes, scratchpad).

        - Dev runtime: ~/memento_s/context
        - Packaged runtime: platformdirs user data dir + /context
        """
        return cls.get_data_dir(packaged=packaged) / "context"


def _print_mode_paths(mode_name: str, packaged: bool) -> None:
    print(f"[{mode_name}]")
    print(f"  home_dir:         {PathManager.get_home_dir(packaged=packaged)}")
    print(f"  project_root_dir: {PathManager.get_project_root_dir(packaged=packaged)}")
    print(f"  config_file:      {PathManager.get_config_file(packaged=packaged)}")
    print(f"  workspace_dir:    {PathManager.get_workspace_dir(packaged=packaged)}")
    print(f"  skills_dir:       {PathManager.get_skills_dir(packaged=packaged)}")
    print(f"  db_dir:           {PathManager.get_db_dir(packaged=packaged)}")
    print(f"  logs_dir:         {PathManager.get_logs_dir(packaged=packaged)}")
    print(f"  venv_dir:         {PathManager.get_venv_dir(packaged=packaged)}")


def _main() -> None:
    """Self-test: print both dev-mode and packaged-mode paths."""
    print("[PathManager] self-test")
    print(f"  detected_packaged_runtime: {PathManager.is_packaged_runtime()}")
    print()
    _print_mode_paths("dev", packaged=False)
    print()
    _print_mode_paths("packaged", packaged=True)


if __name__ == "__main__":
    _main()
