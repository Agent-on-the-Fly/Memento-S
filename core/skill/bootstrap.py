# SPDX-License-Identifier: Apache-2.0
"""Skill 系统引导模块

提供 `init_skill_system()` 作为项目启动的唯一入口。

Usage:
    from core.skill.bootstrap import init_skill_system

    gateway = await init_skill_system()
"""

from __future__ import annotations

from utils.logger import get_logger
from shared.schema import SkillConfig
from .gateway import SkillGateway
from .initializer import SkillInitializer

logger = get_logger(__name__)


async def init_skill_system(config: SkillConfig | None = None) -> SkillGateway:
    """初始化技能系统并返回 Gateway 实例。

    调用者需要自行保存返回的 Gateway 实例，不要依赖全局状态。

    初始化步骤：
    1. 注册全局无状态 Hook（PolicyGateHook, PathPolicyHook, ErrorPatternSupervisionHook,
       SkillPolicyHook, LoopTelemetryHook）
    2. 通过 SkillGateway.from_config() 创建 Gateway（内部完成注册表同步）
    3. 通过 SkillInitializer.initialize() 同步内置 skills 到运行时目录

    Args:
        config: 配置对象，为 None 时自动从 g_config 读取

    Returns:
        SkillGateway 实例
    """
    if config is None:
        config = SkillConfig.from_global_config()

    # ── 全局 Hook 注册 ──────────────────────────────────────────────────
    # P0-0: 注册所有全局 Hook 到单例 HookExecutor
    from core.skill.execution.hooks import register_skill_agent_hooks

    workspace_root = config.workspace_dir.resolve()
    # policy_manager 在此传入，如果已有全局实例（通过 get_policy_manager() 获取）则复用
    # 如果没有（SkillSystem 独立使用），传入 None（PolicyGateHook 会处理）
    from core.skill.execution.hooks.registry import get_policy_manager

    register_skill_agent_hooks(
        workspace_root=workspace_root,
        policy_manager=get_policy_manager(),
    )

    gateway = await SkillGateway.from_config(config)

    initializer = SkillInitializer(config)
    init_result = await initializer.initialize(sync_builtin=True)

    logger.info(
        "Skill system initialized: builtin={}",
        len(init_result.get("builtin_synced", [])),
    )

    return gateway
