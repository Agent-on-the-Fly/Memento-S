"""Stable public API for skill-engine operations.

This module is intended as the long-term import target for upper layers.
It wraps split internal modules and avoids exposing private helper symbols.
"""

from __future__ import annotations

from typing import Any

from core.skill_engine.skill_catalog import (
    build_available_skills_xml,
    build_router_step_note,
    derive_semantic_goal,
    load_available_skills_block,
    load_available_skills_block_from,
    parse_available_skills,
    precompute_router_embedding_cache,
    select_router_top_skills,
)
from core.skill_engine.skill_executor import execute_skill_plan, execute_skill_plan_result, normalize_plan_shape


def normalize_plan(plan: Any) -> dict:
    return normalize_plan_shape(plan)


def execute_plan(skill_name: str, plan: dict[str, Any]) -> str:
    return execute_skill_plan(skill_name, plan)


def execute_plan_result(skill_name: str, plan: dict[str, Any]) -> dict[str, Any]:
    return execute_skill_plan_result(skill_name, plan)


def load_skills_block() -> str:
    return load_available_skills_block()


def load_skills_block_from(path: str) -> str:
    return load_available_skills_block_from(path)


def parse_skills(skills_xml: str) -> list[dict]:
    return parse_available_skills(skills_xml)


def build_skills_xml(skills: list[dict]) -> str:
    return build_available_skills_xml(skills)


def select_top_skills(goal_text: str, skills: list[dict], top_k: int) -> list[dict]:
    return select_router_top_skills(goal_text, skills, top_k=top_k)


def precompute_embedding_cache(
    skills: list[dict],
    *,
    methods: tuple[str, ...] = ("qwen_embedding", "memento_qwen_embedding"),
    show_progress: bool = False,
) -> list[tuple[str, str]]:
    return precompute_router_embedding_cache(skills, methods=methods, show_progress=show_progress)


def build_router_note(
    *,
    step_num: int,
    step_skill: str,
    step_instruction: str,
    step_output: str,
    original_goal: str,
) -> str:
    return build_router_step_note(
        step_num=step_num,
        step_skill=step_skill,
        step_instruction=step_instruction,
        step_output=step_output,
        original_goal=original_goal,
    )


def derive_next_goal(original_goal: str, router_context: list[str]) -> str:
    return derive_semantic_goal(original_goal, router_context)


__all__ = [
    "normalize_plan",
    "execute_plan",
    "execute_plan_result",
    "load_skills_block",
    "load_skills_block_from",
    "parse_skills",
    "build_skills_xml",
    "select_top_skills",
    "precompute_embedding_cache",
    "build_router_note",
    "derive_next_goal",
]

