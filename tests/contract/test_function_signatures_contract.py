"""Signature-level contract tests for core skill-engine entry points."""

from __future__ import annotations

import importlib
import inspect


def _signature_params(func) -> list[inspect.Parameter]:
    return list(inspect.signature(func).parameters.values())


def _assert_param_names(func, expected_names: list[str]) -> None:
    params = _signature_params(func)
    assert [p.name for p in params] == expected_names


def test_skill_executor_signatures() -> None:
    mod = importlib.import_module("core.skill_engine.skill_executor")

    _assert_param_names(mod.normalize_plan_shape, ["plan"])
    _assert_param_names(mod.execute_skill_plan, ["skill_name", "plan"])
    _assert_param_names(mod.execute_skill_plan_result, ["skill_name", "plan"])

    params = _signature_params(mod._dispatch_bridge_op)
    assert [p.name for p in params] == ["op", "parent_plan", "caller_skill", "call_id"]
    assert params[3].kind == inspect.Parameter.KEYWORD_ONLY


def test_skill_catalog_signatures() -> None:
    mod = importlib.import_module("core.skill_engine.skill_catalog")

    _assert_param_names(mod.parse_available_skills, ["skills_xml"])
    _assert_param_names(mod.build_available_skills_xml, ["skills"])
    _assert_param_names(mod.select_router_top_skills, ["goal_text", "skills", "top_k"])

    params = _signature_params(mod.build_router_step_note)
    assert [p.name for p in params] == [
        "step_num",
        "step_skill",
        "step_instruction",
        "step_output",
        "original_goal",
    ]
    assert all(p.kind == inspect.Parameter.KEYWORD_ONLY for p in params)


def test_new_public_api_signatures() -> None:
    api = importlib.import_module("core.skill_engine.api")

    _assert_param_names(api.normalize_plan, ["plan"])
    _assert_param_names(api.execute_plan, ["skill_name", "plan"])
    _assert_param_names(api.execute_plan_result, ["skill_name", "plan"])
    _assert_param_names(api.load_skills_block, [])
    _assert_param_names(api.load_skills_block_from, ["path"])
    _assert_param_names(api.parse_skills, ["skills_xml"])
    _assert_param_names(api.build_skills_xml, ["skills"])
    _assert_param_names(api.select_top_skills, ["goal_text", "skills", "top_k"])

    params = _signature_params(api.precompute_embedding_cache)
    assert [p.name for p in params] == ["skills", "methods", "show_progress"]
    assert params[1].kind == inspect.Parameter.KEYWORD_ONLY
    assert params[2].kind == inspect.Parameter.KEYWORD_ONLY


def test_execute_result_shape_contract() -> None:
    mod = importlib.import_module("core.skill_engine.skill_executor")
    result = mod.execute_skill_plan_result("filesystem", {})
    assert isinstance(result, dict)
    for key in ("ok", "code", "skill_name", "output", "errors", "normalized_plan"):
        assert key in result

