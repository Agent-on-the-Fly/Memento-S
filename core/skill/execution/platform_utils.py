# SPDX-License-Identifier: Apache-2.0
"""Backward-compatible platform helpers for the skill execution package.

The implementation lives in :mod:`middleware.utils.platform`.  Keeping this
module as a thin re-export preserves the public import path used by existing
skills and by older integrations.
"""

from middleware.utils.platform import (
    SCRIPT_EXTENSIONS,
    SUBPROCESS_TEXT_KWARGS,
    background_hint,
    chmod_executable,
    filter_env_by_whitelist,
    has_bash,
    has_node,
    has_powershell,
    is_path_within,
    node_install_hint,
    pip_shim_content,
    pip_shim_path,
    python_executable,
    temp_dir,
    uv_install_hint,
    venv_bin_dir,
    venv_python,
)

__all__ = [
    "SCRIPT_EXTENSIONS",
    "SUBPROCESS_TEXT_KWARGS",
    "background_hint",
    "chmod_executable",
    "filter_env_by_whitelist",
    "has_bash",
    "has_node",
    "has_powershell",
    "is_path_within",
    "node_install_hint",
    "pip_shim_content",
    "pip_shim_path",
    "python_executable",
    "temp_dir",
    "uv_install_hint",
    "venv_bin_dir",
    "venv_python",
]
