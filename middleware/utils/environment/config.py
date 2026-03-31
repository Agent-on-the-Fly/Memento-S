"""Config-based environment variable loading."""

from __future__ import annotations

from typing import Any


def get_config_env_vars(config: Any = None) -> dict[str, str]:
    """Extract environment variables from config.

    Loads env vars from config.env (typically from config.json).
    Users can configure PIP_INDEX_URL and other mirror variables there.

    Args:
        config: Config instance. If None, will import g_config lazily.

    Returns:
        Dictionary of environment variable names (uppercase) to string values.
    """
    env: dict[str, str] = {}
    try:
        if config is None:
            from middleware.config import g_config
            config = g_config

        extra_env = config.env
        if isinstance(extra_env, dict):
            for key, value in extra_env.items():
                if value is None:
                    continue
                if isinstance(value, (str, int, float, bool)):
                    env[str(key).upper()] = str(value)
    except Exception:
        pass
    return env
