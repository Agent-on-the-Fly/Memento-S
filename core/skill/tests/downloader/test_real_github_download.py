# SPDX-License-Identifier: Apache-2.0
"""Downloader real GitHub integration test."""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.schema import SkillConfig
from core.skill.downloader.factory import create_default_download_manager


@pytest.mark.integration
def test_download_real_skill_from_github(tmp_path):
    """Real GitHub download for a published skill."""
    url = "https://github.com/ruvnet/ruflo/tree/main/.agents/skills/agentdb-learning"
    manager = create_default_download_manager()

    result = manager.download(url, tmp_path / "skills", "agentdb-learning")

    assert result is not None
    assert result.exists()
    assert result.is_dir()
    assert (result / "SKILL.md").exists()
