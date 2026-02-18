"""Contract tests for the new stable public API module."""

from __future__ import annotations

import importlib


def test_skill_engine_api_exports() -> None:
    api = importlib.import_module("core.skill_engine.api")
    expected = [
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
    missing = [name for name in expected if not hasattr(api, name)]
    assert not missing, f"core.skill_engine.api missing exports: {missing}"


def test_skill_engine_package_reexports_new_api() -> None:
    pkg = importlib.import_module("core.skill_engine")
    expected = [
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
    missing = [name for name in expected if not hasattr(pkg, name)]
    assert not missing, f"core.skill_engine missing API re-exports: {missing}"

