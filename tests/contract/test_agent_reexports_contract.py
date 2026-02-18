"""Contract tests for agent.py re-export surface."""

from __future__ import annotations

import importlib


def test_agent_reexports_key_symbols() -> None:
    agent = importlib.import_module("agent")
    expected = [
        # skill_catalog
        "load_available_skills_block",
        "parse_available_skills",
        "build_available_skills_xml",
        "select_semantic_top_skills",
        "_load_router_catalog_from_jsonl",
        "_merge_skill_catalog",
        "build_router_step_note",
        "derive_semantic_goal",
        # skill_resolver
        "_resolve_skill_dir",
        "has_local_skill_dir",
        "ensure_skill_available",
        "openskills_read",
        "install_or_update_skill",
        # skill_executor
        "normalize_plan_shape",
        "_execute_filesystem_ops",
        "_execute_terminal_ops",
        "_execute_web_ops",
        "execute_skill_plan_result",
        "execute_skill_plan",
        # router
        "explicit_skill_match",
        "route_skill",
        # skill_runner
        "ask_for_plan",
        "validate_plan_for_skill",
        "run_one_skill",
        "run_one_skill_loop",
        "run_skill_once_with_plan",
        "summarize_step_output",
        "create_skill_on_miss",
    ]
    missing = [name for name in expected if not hasattr(agent, name)]
    assert not missing, f"agent.py missing re-exports: {missing}"

