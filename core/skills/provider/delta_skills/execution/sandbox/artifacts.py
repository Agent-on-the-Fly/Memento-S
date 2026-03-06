
from __future__ import annotations

from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path

from core.config import g_settings
from core.config.logging import get_logger

logger = get_logger(__name__)

ARTIFACT_IGNORE: set[str] = {
    "__runner__.py", "__init__.py", "__params__.json",
    ".pyc", "__pycache__",
}


def should_ignore_artifact(rel_path: str) -> bool:
    name = Path(rel_path).name
    for pattern in ARTIFACT_IGNORE:
        if name == pattern or name.endswith(pattern):
            return True
    return False


def get_output_dir(skill_name: str) -> Path:
    output_root = g_settings.workspace_path / "outputs"
    ts = time.strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:6]
    output_dir = output_root / f"{skill_name}_{ts}_{short_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def collect_local_artifacts(
    work_dir: Path,
    pre_files: set[str],
    skill_name: str,
) -> list[str]:
    post_files = snapshot_files(work_dir)
    new_files = post_files - pre_files
    new_files = {f for f in new_files if not should_ignore_artifact(f)}

    if not new_files:
        return []

    output_dir = get_output_dir(skill_name)
    local_artifacts: list[str] = []

    for rel in sorted(new_files):
        src = work_dir / rel
        dst = output_dir / rel
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            local_artifacts.append(str(dst))
            logger.debug("Collected artifact: %s → %s", rel, dst)
        except Exception as e:
            logger.warning("Failed to collect artifact '%s': %s", rel, e)

    if local_artifacts:
        logger.info(
            "Collected %d artifacts for '%s' to %s",
            len(local_artifacts), skill_name, output_dir,
        )

    return local_artifacts


def snapshot_files(work_dir: Path) -> set[str]:
    return {
        str(f.relative_to(work_dir))
        for f in work_dir.rglob("*")
        if f.is_file()
    }
