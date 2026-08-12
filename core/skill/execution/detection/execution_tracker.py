# SPDX-License-Identifier: Apache-2.0
"""Execution-level file tracker for tool execution.

Aggregates file changes, classifies lifecycle, applies policies,
and reports results for each tool execution.

Note: Core snapshot functionality has been migrated to shared.fs.
This module now uses FsSnapshotManager from shared.fs.snapshot.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, replace as _dataclass_replace
from pathlib import Path
from typing import Any

from core.skill.execution.detection.config import DetectionConfig, LifecyclePolicy
from shared.fs.snapshot import FsSnapshotManager, SnapshotConfig
from shared.fs.types import (
    ChangeType,
    DiffResult,
    FsChange,
    FsSnapshot,
)
from core.skill.execution.detection.lifecycle_classifier import FileLifecycleClassifier
from core.skill.execution.detection.lifecycle_policy import (
    LifecyclePolicyEngine,
    PolicyReport,
)
from core.skill.execution.detection.types import FileLifecycle
from utils.logger import get_logger

logger = get_logger(__name__)


# Backward compatibility alias
ChangeSet = Any  # Will be defined below


@dataclass
class ExecutionRecord:
    """Record of a single tool execution's file changes."""
    execution_id: str
    tool_name: str
    timestamp: float
    input_paths: list[Path]
    change_set: "ChangeSetCompat" | None = None  # Use compat class
    policy_report: PolicyReport | None = None
    artifact_paths: list[Path] = field(default_factory=list)
    temporary_paths: list[Path] = field(default_factory=list)
    audit_paths: list[Path] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "execution_id": self.execution_id,
            "tool_name": self.tool_name,
            "timestamp": self.timestamp,
            "input_paths": [str(p) for p in self.input_paths],
            "artifact_paths": [str(p) for p in self.artifact_paths],
            "temporary_paths": [str(p) for p in self.temporary_paths],
            "audit_paths": [str(p) for p in self.audit_paths],
            "error": self.error,
            "change_count": len(self.change_set.changes) if self.change_set else 0,
        }


class ChangeSetCompat:
    """Compatibility shim for ChangeSet.

    Provides the same interface as the old ChangeSet class
    but uses FsChange from shared.fs.types.
    """
    changes: list[FsChange]
    tool_name: str
    execution_id: str

    def __init__(
        self,
        changes: list[FsChange] | None = None,
        tool_name: str = "",
        execution_id: str = "",
    ):
        self.changes = changes or []
        self.tool_name = tool_name
        self.execution_id = execution_id

    def get_significant_changes(self) -> list[FsChange]:
        """Get only significant changes."""
        return [c for c in self.changes if c.is_significant]

    def get_created(self) -> list[FsChange]:
        return [c for c in self.changes if c.change_type == ChangeType.CREATED]

    def get_modified(self) -> list[FsChange]:
        return [c for c in self.changes if c.change_type == ChangeType.MODIFIED]

    def get_deleted(self) -> list[FsChange]:
        return [c for c in self.changes if c.change_type == ChangeType.DELETED]


# Update type hints
ChangeSet = ChangeSetCompat


class ExecutionFileTracker:
    """Tracks file changes across tool executions.

    Provides a high-level interface for:
    1. Taking snapshots before/after tool execution
    2. Detecting and classifying file changes
    3. Applying lifecycle policies
    4. Reporting artifacts and temporary files

    Now uses FsSnapshotManager from shared.fs.snapshot.
    """

    def __init__(
        self,
        workspace_root: Path,
        config: DetectionConfig | None = None,
        policy: LifecyclePolicy | None = None,
    ):
        self.workspace_root = workspace_root.resolve()
        self.config = config or DetectionConfig(workspace_root=self.workspace_root)
        self.policy = policy or self.config.lifecycle_policy

        # Use FsSnapshotManager from shared.fs
        snapshot_config = SnapshotConfig(
            workspace_root=self.workspace_root,
            compute_hash=False,  # Default to fast mode
            ignore_dirs=frozenset(self.config.ignore_patterns) if hasattr(self.config, 'ignore_patterns') else None,
        )
        self._snapshot_manager = FsSnapshotManager(self.workspace_root, snapshot_config)

        self._classifier = FileLifecycleClassifier()
        self._policy_engine = LifecyclePolicyEngine(self.policy)

        # Execution records for history
        self._records: dict[str, ExecutionRecord] = {}
        self._history: list[ExecutionRecord] = []

        # Track current execution（栈式管理，同一 turn 内多个 tool_call 各自配对）
        self._execution_stack: list[str] = []

        logger.info(
            "[ExecutionFileTracker] Initialized for workspace: {}",
            self.workspace_root
        )

    @property
    def _current_execution_id(self) -> str | None:
        """栈顶（最近一个）execution_id。"""
        return self._execution_stack[-1] if self._execution_stack else None

    @property
    def before_execution_id(self) -> str:
        """Get the before snapshot ID for current execution."""
        cur = self._current_execution_id
        return f"{cur}_before" if cur else "unknown_before"

    @property
    def after_execution_id(self) -> str:
        """Get the after snapshot ID for current execution."""
        cur = self._current_execution_id
        return f"{cur}_after" if cur else "unknown_after"

    async def before_execute(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        input_paths: list[Path] | None = None,
        execution_id: str | None = None,
    ) -> str:
        """Called before tool execution to take a before snapshot.

        Args:
            tool_name: Name of the tool being executed.
            args: Tool arguments.
            input_paths: Explicit list of input file paths.
            execution_id: Optional execution ID (generated if not provided).

        Returns:
            The execution_id for this execution.
        """
        exec_id = execution_id or self._generate_execution_id()
        self._execution_stack.append(exec_id)

        logger.debug(
            "[ExecutionFileTracker] before_execute: tool={}, execution_id={}, stack_size={}",
            tool_name, exec_id, len(self._execution_stack)
        )

        # Take before snapshot using FsSnapshotManager
        self._snapshot_manager.take_full_snapshot(self.before_execution_id)

        # Create execution record
        self._records[exec_id] = ExecutionRecord(
            execution_id=exec_id,
            tool_name=tool_name,
            timestamp=time.time(),
            input_paths=input_paths or [],
        )

        return exec_id

    async def after_execute(
        self,
        execution_id: str | None = None,
        tool_name: str = "",
        args: dict[str, Any] | None = None,
        result: Any = None,
    ) -> ExecutionRecord:
        """Called after tool execution to detect changes and apply policies.

        Args:
            execution_id: Execution ID (uses current if not provided).
            tool_name: Name of the tool (for record keeping).
            args: Tool arguments.
            result: Tool execution result.

        Returns:
            ExecutionRecord with all file changes and policy decisions.
        """
        exec_id = execution_id
        if not exec_id:
            if not self._execution_stack:
                logger.warning(
                    "[ExecutionFileTracker] after_execute called without execution_id and empty stack"
                )
                return ExecutionRecord(
                    execution_id="unknown",
                    tool_name=tool_name,
                    timestamp=time.time(),
                )
            exec_id = self._execution_stack.pop(0)
            # 弹栈后 _current_execution_id 已不再是弹出的 ID，
            # 必须用弹出的 exec_id 直接构建快照 key，避免属性返回栈顶的错误 ID
            _snap_before = f"{exec_id}_before"
            _snap_after = f"{exec_id}_after"
        else:
            _snap_before = f"{exec_id}_before"
            _snap_after = f"{exec_id}_after"

        logger.debug(
            "[ExecutionFileTracker] after_execute: execution_id={}, tool={}",
            exec_id, tool_name
        )

        # Take after snapshot using FsSnapshotManager
        self._snapshot_manager.take_full_snapshot(_snap_after)

        # Get before snapshot
        before_snapshot = self._snapshot_manager.get_snapshot(_snap_before)
        if not before_snapshot:
            logger.warning(
                "[ExecutionFileTracker] No before snapshot for execution_id={}",
                exec_id
            )
            return self._records.get(exec_id, ExecutionRecord(
                execution_id=exec_id,
                tool_name=tool_name,
                timestamp=time.time(),
            ))

        # Compare snapshots to get changes
        changes = self._snapshot_manager.compare(
            _snap_before,
            _snap_after,
        )

        # Wrap in compatibility ChangeSet
        change_set = ChangeSetCompat(
            changes=changes,
            tool_name=tool_name,
            execution_id=exec_id,
        )

        # Extract artifacts from tool result (e.g., python_repl JSON payload)
        # 过滤 __runner__*.py：它是 python_repl 执行时的内部 runner 脚本，不是用户产出
        tool_artifacts: list[str] = []
        if result and isinstance(result, str):
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict):
                    raw_artifacts = parsed.get("artifacts")
                    if isinstance(raw_artifacts, list):
                        tool_artifacts = [
                            str(p) for p in raw_artifacts
                            if isinstance(p, str)
                            and not p.rsplit("/", 1)[-1].startswith("__runner__")
                        ]
                        logger.info(
                            "[ANALYSIS-LOG] ExecutionFileTracker.after_execute: "
                            "extracted tool_artifacts={} from result",
                            tool_artifacts
                        )
            except Exception:
                pass

        # Classify lifecycle for each change
        # 过滤 __runner__*.py：它是 python_repl 执行时的内部 runner 脚本，不是用户产出
        if self.config.enable_lifecycle_classification and change_set.changes:
            filtered_for_policy = [
                c for c in change_set.changes
                if not c.path.name.startswith("__runner__")
            ]
            if len(filtered_for_policy) < len(change_set.changes):
                logger.debug(
                    "[ExecutionFileTracker] Filtered {} __runner__*.py from policy (kept in change_set)",
                    len(change_set.changes) - len(filtered_for_policy)
                )
            # 仅对 policy 决策使用 filtered_for_policy，change_set.changes 保持原样
            classifications = self._classifier.classify_batch(
                filtered_for_policy,
                tool_name=tool_name,
                tool_args=args,
            )

            # Apply policy — pass tool_artifacts so explicitly-reported artifacts are preserved
            policy_report = self._policy_engine.apply(
                ChangeSetCompat(changes=filtered_for_policy, tool_name=tool_name, execution_id=exec_id),
                classifications,
                artifacts=tool_artifacts if tool_artifacts else None,
            )

            # Extract categorized paths
            artifact_paths = self._policy_engine.get_artifact_paths(policy_report)
            temporary_paths = self._policy_engine.get_temporary_paths(policy_report)
            audit_paths = self._policy_engine.get_audit_paths(policy_report)
        else:
            # No classification, treat all as UNKNOWN
            policy_report = None
            artifact_paths = [
                c.path for c in change_set.changes
                if c.change_type != ChangeType.DELETED and c.path.exists()
                and not c.path.name.startswith("__runner__")
            ]
            temporary_paths = []
            audit_paths = []

        # Update execution record
        record = self._records.get(exec_id)
        if record:
            record.change_set = change_set
            record.policy_report = policy_report
            record.artifact_paths = artifact_paths
            record.temporary_paths = temporary_paths
            record.audit_paths = audit_paths
        else:
            record = ExecutionRecord(
                execution_id=exec_id,
                tool_name=tool_name,
                timestamp=time.time(),
                change_set=change_set,
                policy_report=policy_report,
                artifact_paths=artifact_paths,
                temporary_paths=temporary_paths,
                audit_paths=audit_paths,
            )
            self._records[exec_id] = record

        # Add to history
        self._history.append(record)

        # Clean up snapshots to save memory
        self._snapshot_manager.release(_snap_before)
        self._snapshot_manager.release(_snap_after)

        logger.info(
            "[ExecutionFileTracker] Execution complete: id={}, tool={}, "
            "artifacts={}, temporary={}, audit={}",
            exec_id, tool_name,
            len(artifact_paths),
            len(temporary_paths),
            len(audit_paths),
        )

        return record

    def get_record(self, execution_id: str) -> ExecutionRecord | None:
        """Get the record for a specific execution."""
        return self._records.get(execution_id)

    def get_history(self) -> list[ExecutionRecord]:
        """Get all execution records in chronological order."""
        return list(self._history)

    def get_all_artifacts(self) -> list[Path]:
        """Get all artifact paths from all executions."""
        artifacts: set[Path] = set()
        for record in self._history:
            artifacts.update(record.artifact_paths)
        return list(artifacts)

    def get_all_temporary_files(self) -> list[Path]:
        """Get all temporary file paths from all executions."""
        temporary: set[Path] = set()
        for record in self._history:
            temporary.update(record.temporary_paths)
        return list(temporary)

    def cleanup_temporary(
        self,
        execution_id: str | None = None,
    ) -> int:
        """Clean up temporary files.

        Args:
            execution_id: If provided, only cleanup files from that execution.
                        If None, cleanup all temporary files.

        Returns:
            Number of files cleaned up.
        """
        if execution_id:
            record = self._records.get(execution_id)
            if not record:
                return 0
            paths_to_cleanup = record.temporary_paths
        else:
            paths_to_cleanup = self.get_all_temporary_files()

        cleaned = 0
        for path in paths_to_cleanup:
            try:
                if path.exists():
                    path.unlink()
                    cleaned += 1
                    logger.debug(
                        "[ExecutionFileTracker] Cleaned up temporary: {}",
                        path
                    )
            except OSError as e:
                logger.warning(
                    "[ExecutionFileTracker] Failed to cleanup {}: {}",
                    path, e
                )

        logger.info(
            "[ExecutionFileTracker] Cleaned up {} temporary files",
            cleaned
        )
        return cleaned

    def get_significant_changes(
        self,
        execution_id: str | None = None,
    ) -> list[FsChange]:
        """Get significant file changes.

        Args:
            execution_id: If provided, get changes for that execution only.

        Returns:
            List of significant FsChange objects.
        """
        if execution_id:
            record = self._records.get(execution_id)
            if not record or not record.change_set:
                return []
            return record.change_set.get_significant_changes()

        # Aggregate from all executions
        all_changes: list[FsChange] = []
        for record in self._history:
            if record.change_set:
                all_changes.extend(record.change_set.get_significant_changes())
        return all_changes

    def get_summary(self) -> dict:
        """Get a summary of all tracked file changes."""
        total_artifacts = 0
        total_temporary = 0
        total_audit = 0
        total_executions = len(self._history)

        for record in self._history:
            total_artifacts += len(record.artifact_paths)
            total_temporary += len(record.temporary_paths)
            total_audit += len(record.audit_paths)

        return {
            "total_executions": total_executions,
            "total_artifacts": total_artifacts,
            "total_temporary": total_temporary,
            "total_audit": total_audit,
            "workspace_root": str(self.workspace_root),
        }

    def clear_history(self) -> None:
        """Clear all history and records."""
        self._records.clear()
        self._history.clear()
        self._snapshot_manager.clear()
        logger.info("[ExecutionFileTracker] History cleared")

    def _generate_execution_id(self) -> str:
        """Generate a unique execution ID."""
        return f"exec_{uuid.uuid4().hex[:12]}"
