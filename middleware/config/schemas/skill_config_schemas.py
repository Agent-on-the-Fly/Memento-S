# SPDX-License-Identifier: Apache-2.0
"""
Skill Registry Configuration Pydantic Models.

These models mirror the JSON Schema in skill_config_schema.json but provide
Pydantic type-checking, validation, and IDE auto-complete.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SkillEntry(BaseModel):
    """Single skill registry entry (mirrors a value in skill.json's skills dict)."""

    model_config = ConfigDict(extra="ignore")

    location: str
    source: Literal["builtin", "local", "cloud"]
    version: int = 1
    installed_at: datetime | None = None
    status: Literal["active", "disabled"] = "active"
    tags: list[str] = Field(default_factory=list)


class SkillIndex(BaseModel):
    """Sync index metadata for the skill registry."""

    model_config = ConfigDict(extra="ignore")

    last_sync: datetime | None = None
    sync_errors: list[str] = Field(default_factory=list)


class SkillRegistryConfig(BaseModel):
    """Top-level skill registry configuration (mirrors skill.json)."""

    model_config = ConfigDict(extra="ignore")

    version: int = 1
    skills: dict[str, SkillEntry] = Field(default_factory=dict)
    index: SkillIndex = Field(default_factory=SkillIndex)


__all__ = [
    "SkillEntry",
    "SkillIndex",
    "SkillRegistryConfig",
]
