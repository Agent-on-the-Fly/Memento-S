# SPDX-License-Identifier: Apache-2.0
"""shared/schema/skill.py — Skill 核心元数据类型（跨层共享）。

从 core/skill/schema.py 迁移：
  - SkillManifest、SkillGovernanceMeta（被 gui/memento_s 共引）

ExecutionMode 在此独立定义（与 core.skill.schema.ExecutionMode 同值），
避免 shared/schema → core/skill 的循环依赖。

保留在 core/skill/schema.py 的类型：
  - ExecutionMode（Skill 类依赖它）
  - Skill（仅 skill 子系统内部使用）
  - ErrorType、SkillExecutionOutcome 等执行层内部类型
  - SkillExecOptions、DiscoverStrategy（仅 skill 内部使用）
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ExecutionMode(str, Enum):
    """Skill 执行模式。"""

    KNOWLEDGE = "knowledge"
    PLAYBOOK = "playbook"


class SkillGovernanceMeta(BaseModel):
    """Skill 治理元数据。"""

    source: Literal["local", "cloud"] = "local"


class SkillManifest(BaseModel):
    """Skill 元数据 — 用于发现和注册。

    这是 core/skill 对外的唯一跨层契约。
    """

    name: str
    description: str
    execution_mode: ExecutionMode
    parameters: dict[str, Any] | None = None
    dependencies: list[str] = Field(default_factory=list)
    governance: SkillGovernanceMeta = Field(default_factory=SkillGovernanceMeta)


__all__ = ["ExecutionMode", "SkillGovernanceMeta", "SkillManifest"]
