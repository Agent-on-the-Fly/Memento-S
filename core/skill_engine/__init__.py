"""Public entry points for skill-engine helpers.

This package-level module provides a stable import surface for higher layers
(`cli`, `agent.py`) and for contributors extending core behaviour.
"""

from core.skill_engine.skill_runner import (
    ask_for_plan,
    validate_plan_for_skill,
    build_strict_schema_prompt,
    normalize_skill_creator_plan,
    run_one_skill,
    run_one_skill_loop,
    run_skill_once_with_plan,
    should_auto_continue_skill_result,
    _count_approx_tokens,
    summarize_step_output,
    _should_create_skill_on_miss_fallback,
    should_create_skill_on_miss,
    create_skill_on_miss,
)
from core.skill_engine.api import (
    normalize_plan,
    execute_plan,
    execute_plan_result,
    load_skills_block,
    load_skills_block_from,
    parse_skills,
    build_skills_xml,
    select_top_skills,
    precompute_embedding_cache,
    build_router_note,
    derive_next_goal,
)

__all__ = [
    "ask_for_plan",
    "validate_plan_for_skill",
    "build_strict_schema_prompt",
    "normalize_skill_creator_plan",
    "run_one_skill",
    "run_one_skill_loop",
    "run_skill_once_with_plan",
    "should_auto_continue_skill_result",
    "_count_approx_tokens",
    "summarize_step_output",
    "_should_create_skill_on_miss_fallback",
    "should_create_skill_on_miss",
    "create_skill_on_miss",
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
