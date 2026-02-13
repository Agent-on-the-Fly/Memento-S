"""Backward-compatible facade for skill-engine runner helpers.

This module keeps stable import paths (`core.skill_engine.skill_runner`) while
internally splitting the implementation into smaller focused modules:
- planning.py
- execution.py
- summarization.py
- create_on_miss.py
"""

from core.skill_engine.planning import (
    ask_for_plan,
    validate_plan_for_skill,
    build_strict_schema_prompt,
    normalize_skill_creator_plan,
)
from core.skill_engine.execution import (
    run_one_skill,
    run_one_skill_loop,
    run_skill_once_with_plan,
    should_auto_continue_skill_result,
)
from core.skill_engine.summarization import (
    _count_approx_tokens,
    summarize_step_output,
)
from core.skill_engine.create_on_miss import (
    _should_create_skill_on_miss_fallback,
    should_create_skill_on_miss,
    create_skill_on_miss,
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
]
