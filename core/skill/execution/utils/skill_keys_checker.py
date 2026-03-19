"""Skill API-key availability checker.

Provides utilities to check whether a skill's required API keys
are present in the current runtime environment (config + os.environ).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from utils.logger import get_logger

if TYPE_CHECKING:
    from core.skill.schema import Skill

logger = get_logger(__name__)


def get_available_keys() -> set[str]:
    """Collect all API key names currently available.

    Sources:
    1. Config-derived env vars (config.env)
    2. os.environ entries containing 'KEY'
    """
    keys: set[str] = set()

    # 1. From config (same logic as sandbox _config_env_vars)
    try:
        from middleware.config import g_config

        cfg = g_config
        extra_env = cfg.env
        if isinstance(extra_env, dict):
            for key, value in extra_env.items():
                if value and isinstance(value, str) and "KEY" in key.upper():
                    keys.add(str(key).upper())
    except Exception:
        pass

    # 2. From os.environ
    for k, v in os.environ.items():
        if "KEY" in k and v:
            keys.add(k)

    return keys


def check_skill_keys(skill: Skill) -> tuple[bool, list[str]]:
    """Check whether all required keys for a skill are configured.

    Returns:
        (all_satisfied, missing_keys)
    """
    if not skill.required_keys:
        return True, []
    available = get_available_keys()
    missing = [k for k in skill.required_keys if k not in available]
    return len(missing) == 0, missing
