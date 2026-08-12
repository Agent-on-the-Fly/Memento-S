# SPDX-License-Identifier: Apache-2.0
"""shared/schema — 跨层共享 Schema 中心。

所有被 core/skill/ 之外的模块共引的 Skill 相关类型在此定义和重导出。

使用方式：
    from shared.schema import (
        SkillManifest, SkillSearchResult, SkillExecutionResponse, SkillConfig,
    )
"""

from shared.schema.skill import ExecutionMode, SkillGovernanceMeta, SkillManifest
from shared.schema.skill_search import SkillSearchResult
from shared.schema.skill_execution import SkillErrorCode, SkillExecutionResponse, SkillStatus
from shared.schema.skill_config import SkillConfig
from shared.schema.result import Result, Ok, Err

__all__ = [
    # Manifest
    "SkillManifest",
    "SkillGovernanceMeta",
    "ExecutionMode",
    # Search
    "SkillSearchResult",
    # Execution
    "SkillErrorCode",
    "SkillExecutionResponse",
    "SkillStatus",
    # Config
    "SkillConfig",
    # Result
    "Result",
    "Ok",
    "Err",
]