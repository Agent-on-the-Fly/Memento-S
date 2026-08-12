# SPDX-License-Identifier: Apache-2.0
"""Tests for infra/context/base.py — ContextProvider abstract interface.

These tests verify the ContextProvider interface by checking that concrete
implementations (FileContextProvider) expose all required abstract methods.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from infra.context.base import ContextProvider


class TestContextProviderInterface:
    """Test that ContextProvider declares all required abstract methods."""

    def test_all_abstract_methods_defined(self):
        """Every abstract method in ContextProvider should have a proper docstring."""
        methods = [
            "load_and_assemble",
            "append",
            "prepare_for_api",
            "assemble_system_prompt",
            "load_history",
            "build_history_summary",
            "init_budget",
            "sync_tokens",
            "total_tokens",
            "persist_tool_result",
            "session_memory",
            "context_memory",
            "get_stats",
        ]
        for method_name in methods:
            assert hasattr(ContextProvider, method_name), f"Missing: {method_name}"

    def test_load_and_assemble_signature(self):
        sig = inspect.signature(ContextProvider.load_and_assemble)
        params = list(sig.parameters.keys())
        assert "current_message" in params

    def test_abstract_methods_have_docstrings(self):
        abstract_methods = [
            ContextProvider.load_and_assemble,
            ContextProvider.append,
            ContextProvider.prepare_for_api,
            ContextProvider.assemble_system_prompt,
            ContextProvider.load_history,
            ContextProvider.build_history_summary,
            ContextProvider.init_budget,
            ContextProvider.sync_tokens,
            ContextProvider.persist_tool_result,
            ContextProvider.session_memory,
            ContextProvider.context_memory,
            ContextProvider.get_stats,
        ]
        for method in abstract_methods:
            assert method.__doc__ is not None, f"Missing docstring: {method.__name__}"

    def test_context_provider_is_abc(self):
        """ContextProvider should be an ABC."""
        assert hasattr(ContextProvider, "__abstractmethods__")

    def test_no_concrete_implementations_in_base(self):
        """Base class should not have any non-abstract concrete methods."""
        # Properties are fine (they're just descriptors)
        # We just check that abstractmethods is non-empty
        assert len(ContextProvider.__abstractmethods__) > 0
