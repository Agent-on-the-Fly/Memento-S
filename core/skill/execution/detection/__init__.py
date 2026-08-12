# SPDX-License-Identifier: Apache-2.0
"""File change detection module.

Provides unified file change detection, lifecycle classification,
and lifecycle policy enforcement for skill tool executions.

Note: Core snapshot types have been migrated to shared/fs/.
This module re-exports them for backward compatibility.
"""

from __future__ import annotations

from core.skill.execution.detection.config import LifecyclePolicy, DetectionConfig
from core.skill.execution.detection.types import FileLifecycle
from shared.fs.types import (
    # Re-export from shared.fs for backward compatibility
    ChangeType,
    FsSnapshotEntry,
    FsSnapshot,
    FsChange,
    DiffResult,
)
# Aliases for backward compatibility
from shared.fs.types import FsSnapshot as FileSnapshot
from shared.fs.types import FsChange as FileChange

# ExecutionRecord is also available from shared.fs.types
# Alias for backward compatibility
from core.skill.execution.detection.execution_tracker import ExecutionRecord

# For backward compatibility, define a minimal ChangeSet equivalent
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChangeSet:
    """Collection of file changes from a single tool execution.

    This is a compatibility shim. The preferred type is now shared.fs.types.FsSnapshot.diff().
    """
    changes: list[FsChange] = field(default_factory=list)
    tool_name: str = ""
    execution_id: str = ""

    def get_significant_changes(self) -> list[FsChange]:
        """Get only significant changes."""
        return [c for c in self.changes if c.is_significant]

    def get_created(self) -> list[FsChange]:
        return [c for c in self.changes if c.change_type == ChangeType.CREATED]

    def get_modified(self) -> list[FsChange]:
        return [c for c in self.changes if c.change_type == ChangeType.MODIFIED]

    def get_deleted(self) -> list[FsChange]:
        return [c for c in self.changes if c.change_type == ChangeType.DELETED]


from core.skill.execution.detection.lifecycle_classifier import (
    FileLifecycleClassifier,
    LifecycleClassification,
)
from core.skill.execution.detection.lifecycle_policy import (
    LifecyclePolicyEngine,
    LifecycleDecision,
    PolicyResult,
    PolicyReport,
)
from core.skill.execution.detection.execution_tracker import (
    ExecutionFileTracker,
)

__all__ = [
    # Config
    "LifecyclePolicy",
    "DetectionConfig",
    # Types
    "FileLifecycle",
    # Re-exported from shared.fs (for backward compatibility)
    "ChangeType",
    "FsSnapshotEntry",
    "FileSnapshot",
    "FileSnapshotEntry",
    "FsSnapshot",
    "FsChange",
    "FileChange",
    "ChangeSet",
    "DiffResult",
    # Classifier
    "FileLifecycleClassifier",
    "LifecycleClassification",
    # Policy
    "LifecyclePolicyEngine",
    "LifecycleDecision",
    "PolicyResult",
    "PolicyReport",
    # Tracker
    "ExecutionFileTracker",
    "ExecutionRecord",
]
