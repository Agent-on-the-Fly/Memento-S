# SPDX-License-Identifier: Apache-2.0
"""
server/schema/skill.py
技能相关 Pydantic 模型
"""
from pydantic import BaseModel
from typing import Optional


class SkillManifest(BaseModel):
    """技能清单"""
    name: str
    description: str
    version: str
    author: Optional[str] = None
    tags: list[str] = []


class SkillResult(BaseModel):
    """技能执行结果"""
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
