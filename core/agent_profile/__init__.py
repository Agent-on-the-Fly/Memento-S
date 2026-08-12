# SPDX-License-Identifier: Apache-2.0
"""AgentProfile — agent identity system: SOUL.md + USER.md.

Architecture (v2):
  Facade         — AgentProfileManager (manager.py): agent 唯一入口，委托给底层两个 Manager
  SoulManager    (soul_manager.py): SOUL.md I/O + TTL cache
  UserManager    (user_manager.py): USER.md I/O + TTL cache
  models.py      — AgentProfile dataclass
  defaults.py    — SOUL.md / USER.md 默认模板
  utils.py       — SOUL.md 解析/格式化
  constants.py   — SOUL_FILE / USER_FILE / CACHE_TTL_SECONDS
"""

from core.agent_profile.constants import SOUL_FILE, USER_FILE
from core.agent_profile.manager import AgentProfileManager, apm
from core.agent_profile.models import AgentProfile
from core.agent_profile.soul_manager import SoulManager
from core.agent_profile.user_manager import UserManager

__all__ = [
    "AgentProfile",
    "AgentProfileManager",
    "apm",
    "SOUL_FILE",
    "SoulManager",
    "USER_FILE",
    "UserManager",
]
