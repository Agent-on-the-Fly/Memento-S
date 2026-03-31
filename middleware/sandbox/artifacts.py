"""Artifact management for sandbox execution.

Provides utilities for managing sandbox working directories and
collecting generated artifacts.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from middleware.config import g_config


class ArtifactManager:
    """Manages sandbox artifacts and working directories."""

    @staticmethod
    def get_sandbox_dir(skill_name: str, session_id: str) -> Path:
        """Get the sandbox working directory for a skill execution.

        Args:
            skill_name: Name of the skill
            session_id: Session identifier

        Returns:
            Path to the sandbox directory
        """
        base_dir = Path(g_config.paths.workspace_dir) / ".sandbox"
        # Create a unique directory based on skill and session
        dir_hash = hashlib.md5(f"{skill_name}:{session_id}".encode()).hexdigest()[:8]
        sandbox_dir = base_dir / f"{skill_name}_{session_id}_{dir_hash}"
        return sandbox_dir

    @staticmethod
    def get_output_dir(skill_name: str, session_id: str) -> Path:
        """Get the output directory for artifacts.

        Args:
            skill_name: Name of the skill
            session_id: Session identifier

        Returns:
            Path to the output directory
        """
        base_dir = Path(g_config.paths.workspace_dir) / ".output"
        dir_hash = hashlib.md5(f"{skill_name}:{session_id}".encode()).hexdigest()[:8]
        output_dir = base_dir / f"{skill_name}_{session_id}_{dir_hash}"
        return output_dir

    @staticmethod
    def snapshot_files(directory: Path) -> dict[str, Any]:
        """Take a snapshot of files in a directory.

        Args:
            directory: Directory to snapshot

        Returns:
            Dict mapping file paths to file metadata (mtime, size)
        """
        if not directory.exists():
            return {}

        snapshot = {}
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                snapshot[str(file_path.relative_to(directory))] = {
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                }
        return snapshot

    @staticmethod
    def collect_local_artifacts(
        work_dir: Path,
        pre_snapshot: dict[str, Any],
        skill_name: str,
        session_id: str = "",
    ) -> list[str]:
        """Collect artifacts generated during execution.

        Compares current files with pre-execution snapshot to identify new/modified files.

        Args:
            work_dir: Working directory
            pre_snapshot: File snapshot before execution
            skill_name: Name of the skill
            session_id: Session identifier

        Returns:
            List of artifact file paths
        """
        if not work_dir.exists():
            return []

        artifacts = []
        current_snapshot = ArtifactManager.snapshot_files(work_dir)

        # Find new or modified files
        for rel_path, current_info in current_snapshot.items():
            if rel_path not in pre_snapshot:
                # New file
                artifacts.append(str(work_dir / rel_path))
            elif current_info["mtime"] != pre_snapshot[rel_path]["mtime"]:
                # Modified file
                artifacts.append(str(work_dir / rel_path))

        return artifacts

    @staticmethod
    def cleanup_sandbox(skill_name: str, session_id: str) -> None:
        """Clean up sandbox directory after execution.

        Args:
            skill_name: Name of the skill
            session_id: Session identifier
        """
        sandbox_dir = ArtifactManager.get_sandbox_dir(skill_name, session_id)
        if sandbox_dir.exists():
            import shutil

            shutil.rmtree(sandbox_dir)


__all__ = ["ArtifactManager"]
