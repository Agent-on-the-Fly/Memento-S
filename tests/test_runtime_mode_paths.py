# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from utils.runtime_mode import RuntimeMode


def test_memento_home_overrides_all_runtime_roots(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "isolated-memento"
    monkeypatch.setenv("MEMENTO_HOME", str(root))

    for mode in RuntimeMode:
        assert mode.data_dir == root
        assert mode.config_dir == root
        assert mode.logs_dir == root / "logs"
        assert mode.skills_dir == root / "skills"
        assert mode.workspace_dir == root / "workspace"
        assert mode.db_dir == root / "db"
        assert mode.venv_dir == root / ".venv"
        assert mode.context_dir == root / "context"


def test_default_dev_root_uses_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MEMENTO_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    assert RuntimeMode.DEV.data_dir == tmp_path / "memento_s"
    assert RuntimeMode.DEV.config_dir == tmp_path / "memento_s"
