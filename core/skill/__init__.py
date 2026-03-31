"""skill — 技能领域模型与契约导出

提供 Skill 相关的领域模型和公共 API。

使用方式：
    from core.skill import init_skill_system

    # 初始化并获取 Gateway（自动从 g_config 读取配置，包含完整的 skills 同步和 embedding 生成）
    gateway = await init_skill_system()
"""

from __future__ import annotations

# 初始化函数
from .initializer import init_skill_system

# 配置与核心类
from .config import SkillConfig
from .gateway import SkillGateway
from .market import SkillMarket

# 核心数据模型
from .schema import (
    ExecutionMode,
    Skill,
    SkillExecutionResponse,
    SkillManifest,
    SkillStatus,
)

__all__ = [
    # 初始化
    "init_skill_system",
    # 配置
    "SkillConfig",
    # 契约接口
    "SkillGateway",
    "SkillMarket",
    # 核心数据模型
    "Skill",
    "SkillManifest",
    "ExecutionMode",
    "SkillStatus",
    "SkillExecutionResponse",
]
