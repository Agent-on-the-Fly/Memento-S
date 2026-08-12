# SPDX-License-Identifier: Apache-2.0
"""Tests for tools/mcp/client.py"""

from __future__ import annotations

import pytest

from tools.mcp.client import _resolve_env_vars


class TestResolveEnvVars:
    """Test environment variable resolution."""

    def test_resolve_simple_string(self, monkeypatch):
        monkeypatch.setenv("MY_VAR", "my_value")
        assert _resolve_env_vars("hello ${MY_VAR} world") == "hello my_value world"

    def test_resolve_missing_var(self, monkeypatch):
        monkeypatch.delenv("UNDEFINED_VAR", raising=False)
        assert _resolve_env_vars("hello ${UNDEFINED_VAR} world") == "hello  world"

    def test_resolve_dict(self, monkeypatch):
        monkeypatch.setenv("API_KEY", "secret123")
        input_dict = {
            "command": "npx",
            "args": ["-y", "@server"],
            "env": {
                "TOKEN": "${API_KEY}",
                "OTHER": "static",
            }
        }
        result = _resolve_env_vars(input_dict)
        assert result["env"]["TOKEN"] == "secret123"
        assert result["env"]["OTHER"] == "static"

    def test_resolve_list(self, monkeypatch):
        monkeypatch.setenv("PORT", "8080")
        input_list = ["node", "server.js", "--port", "${PORT}"]
        result = _resolve_env_vars(input_list)
        assert result == ["node", "server.js", "--port", "8080"]

    def test_resolve_no_substitution(self):
        assert _resolve_env_vars("no vars here") == "no vars here"
        assert _resolve_env_vars(123) == 123
        assert _resolve_env_vars(None) is None

    def test_resolve_multiple_vars(self, monkeypatch):
        monkeypatch.setenv("VAR1", "val1")
        monkeypatch.setenv("VAR2", "val2")
        assert _resolve_env_vars("${VAR1} and ${VAR2}") == "val1 and val2"
