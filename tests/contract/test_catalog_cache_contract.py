"""Contract tests for catalog_cache abstraction API."""

from __future__ import annotations

import importlib


def test_catalog_cache_api_surface() -> None:
    mod = importlib.import_module("core.skill_engine.catalog.catalog_cache")
    expected = [
        "get_last_visible_agents_sig",
        "set_last_visible_agents_sig",
        "get_or_build_semantic_index",
        "get_or_build_bm25_index",
        "get_jsonl_catalog_cache",
        "put_jsonl_catalog_cache",
        "get_embedding_runtime_cache",
        "put_embedding_runtime_cache",
        "get_embedding_doc_cache",
        "put_embedding_doc_cache",
        "begin_embedding_prewarm",
        "finish_embedding_prewarm",
        "router_method",
        "router_embed_max_length",
        "router_embed_batch_size",
        "router_embed_query_instruction",
        "router_embed_prewarm_enabled",
        "router_embed_cache_dir",
        "env_str",
        "tokenize_for_semantic",
        "catalog_signature",
    ]
    missing = [name for name in expected if not hasattr(mod, name)]
    assert not missing, f"catalog_cache missing abstraction APIs: {missing}"

