"""Contract tests for facade/module exports after split refactors.

These tests intentionally verify public compatibility surfaces only.
They should fail when a symbol expected by downstream modules disappears.
"""

from __future__ import annotations

import importlib
from typing import Iterable


def _assert_exports(module_name: str, names: Iterable[str]) -> None:
    module = importlib.import_module(module_name)
    missing = [name for name in names if not hasattr(module, name)]
    assert not missing, f"{module_name} missing exports: {missing}"


def test_skill_catalog_facade_exports() -> None:
    expected = [
        "load_available_skills_block_from",
        "load_available_skills_block",
        "write_visible_skills_block",
        "parse_available_skills",
        "build_available_skills_xml",
        "_ROUTER_STOPWORDS",
        "_tokenize_for_semantic",
        "_catalog_signature",
        "_build_semantic_index",
        "_get_semantic_index",
        "select_semantic_top_skills",
        "_resolve_forced_skills",
        "_append_forced_skills_and_fill",
        "_tokenize_for_bm25",
        "_build_bm25_index",
        "_get_bm25_index",
        "select_bm25_top_skills",
        "_resolve_embedding_paths",
        "_resolve_embedding_cache_file",
        "_get_model_device",
        "_last_token_pool",
        "_load_embedding_runtime",
        "_encode_texts_with_embedding",
        "_load_embedding_doc_cache",
        "_save_embedding_doc_cache",
        "_get_embedding_doc_matrix",
        "_prewarm_embedding_catalog_sync",
        "_router_method_to_embedding_methods",
        "ensure_router_embedding_prewarm",
        "precompute_router_embedding_cache",
        "select_embedding_top_skills",
        "select_router_top_skills",
        "_resolve_catalog_jsonl_path",
        "_parse_int_or_zero",
        "_choose_catalog_entry",
        "parse_catalog_jsonl_text",
        "_load_router_catalog_from_jsonl",
        "_merge_skill_catalog",
        "build_router_step_note",
        "derive_semantic_goal",
    ]
    _assert_exports("core.skill_engine.skill_catalog", expected)


def test_skill_executor_facade_exports() -> None:
    expected = [
        "_normalize_op_dict",
        "_tool_call_to_op",
        "normalize_plan_shape",
        "_coerce_skill_context",
        "_extract_skill_context",
        "_execute_skill_creator_plan",
        "_filesystem_tree",
        "_execute_filesystem_op",
        "_execute_filesystem_ops",
        "_execute_terminal_ops",
        "_convert_pip_to_uv",
        "_run_uv_pip",
        "_execute_uv_pip_ops",
        "_web_google_search",
        "_fetch_async",
        "_web_fetch",
        "_execute_web_ops",
        "_dispatch_bridge_op",
        "execute_skill_plan_result",
        "execute_skill_plan",
    ]
    _assert_exports("core.skill_engine.skill_executor", expected)


def test_split_package_exports() -> None:
    _assert_exports(
        "core.skill_engine.executor",
        [
            "filesystem_tree",
            "execute_filesystem_op",
            "execute_filesystem_ops",
            "convert_pip_to_uv",
            "execute_terminal_ops",
            "execute_uv_pip_ops",
            "run_uv_pip",
            "web_google_search",
            "fetch_async",
            "web_fetch",
            "execute_web_ops",
        ],
    )
    _assert_exports(
        "core.skill_engine.catalog",
        [
            "_ROUTER_STOPWORDS",
            "_tokenize_for_semantic",
            "_catalog_signature",
            "load_available_skills_block_from",
            "load_available_skills_block",
            "write_visible_skills_block",
            "parse_available_skills",
            "build_available_skills_xml",
            "select_semantic_top_skills",
            "select_bm25_top_skills",
            "ensure_router_embedding_prewarm",
            "precompute_router_embedding_cache",
            "select_embedding_top_skills",
            "select_router_top_skills",
            "_load_router_catalog_from_jsonl",
            "_merge_skill_catalog",
            "build_router_step_note",
            "derive_semantic_goal",
        ],
    )
    _assert_exports(
        "core.skill_engine.bridge",
        [
            "TOOL_PROTOCOL_VERSION",
            "ToolSchema",
            "ToolSpec",
            "ToolCall",
            "ToolCallResult",
            "build_tool_registry",
            "coerce_call_stack",
            "dispatch_bridge_op",
        ],
    )

