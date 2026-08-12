# SPDX-License-Identifier: Apache-2.0
"""skill — 技能领域模型与契约导出

提供 Skill 相关的领域模型和公共 API。

使用方式：
    from core.skill import init_skill_system

    # 初始化并获取 Gateway（自动从 g_config 读取配置，包含完整的 skills 同步和注册表初始化）
    gateway = await init_skill_system()
"""

from __future__ import annotations

# 初始化函数
from .bootstrap import init_skill_system

from .gateway import SkillGateway
from .market import SkillMarket
from .registry import SkillRegistry, registry

# 核心数据模型
from .schema import (
    Skill,
)
