# SPDX-License-Identifier: Apache-2.0
"""Shared tool security utilities (utility layer).

Provides IGNORE_DIRS for file traversal. All path boundary logic lives in
shared.tools.path_boundary.
"""

from __future__ import annotations

from shared.tools.path_boundary import PathBoundary

IGNORE_DIRS = PathBoundary.IGNORE_DIRS
