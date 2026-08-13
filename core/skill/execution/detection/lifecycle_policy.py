# SPDX-License-Identifier: Apache-2.0
"""Lifecycle policy engine for file change detection.

Decides what to do with files based on their lifecycle classification
(CLEANUP, PERSIST, or AUDIT).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from core.skill.execution.detection.config import LifecyclePolicy
from shared.fs.types import ChangeType
from core.skill.execution.detection.lifecycle_classifier import (
    FileLifecycleClassifier,
    LifecycleClassification,
)
from core.skill.execution.detection.types import FileLifecycle
from utils.logger import get_logger

logger = get_logger(__name__)


class LifecycleDecision(Enum):
    """Decision made by the policy engine for a file."""
    CLEANUP = "cleanup"          # Delete the file
    CLEANUP_BACKUP = "cleanup_backup"  # Backup then delete
    PERSIST = "persist"          # Keep the file
    REGISTER_ARTIFACT = "register_artifact"  # Keep and register to ArtifactRegistry
    AUDIT = "audit"              # Keep for manual review


@dataclass
class PolicyResult:
    """Result of applying policy to a single file."""
    path: Path
    decision: LifecycleDecision
    lifecycle: FileLifecycle
    classification: LifecycleClassification | None
    action_taken: str = ""  # Description of action taken
    error: str | None = None


@dataclass
class PolicyReport:
    """Report of all policy decisions for a ChangeSet."""
    results: list[PolicyResult] = field(default_factory=list)
    cleanup_count: int = 0
    persist_count: int = 0
    artifact_count: int = 0
    audit_count: int = 0
    errors: list[str] = field(default_factory=list)

    def add_result(self, result: PolicyResult) -> None:
        """Add a policy result."""
        self.results.append(result)
        if result.decision in {LifecycleDecision.CLEANUP, LifecycleDecision.CLEANUP_BACKUP}:
            self.cleanup_count += 1
        elif result.decision == LifecycleDecision.PERSIST:
            self.persist_count += 1
        elif result.decision == LifecycleDecision.REGISTER_ARTIFACT:
            self.artifact_count += 1
        elif result.decision == LifecycleDecision.AUDIT:
            self.audit_count += 1
        if result.error:
            self.errors.append(result.error)


class LifecyclePolicyEngine:
    """Enforces lifecycle policies on file changes.

    Applies rules based on:
    1. File lifecycle classification
    2. Policy configuration
    3. Resource constraints
    """

    def __init__(self, policy: LifecyclePolicy | None = None):
        self.policy = policy or LifecyclePolicy()
        self._classifier = FileLifecycleClassifier()

    def decide(
        self,
        change: "FileChange",  # noqa: F821
        classification: LifecycleClassification | None = None,
        artifacts: frozenset[Path] | None = None,
    ) -> LifecycleDecision:
        """Decide what to do with a file based on its lifecycle.

        Args:
            change: The file change to evaluate.
            classification: Optional pre-computed classification.
            artifacts: Set of paths explicitly reported as artifacts by the tool.

        Returns:
            LifecycleDecision for this file.
        """
        lifecycle = getattr(change, "lifecycle", None)
        if classification is None and lifecycle is None:
            classification = self._classifier.classify(change.path, "")
        if classification is not None:
            lifecycle = classification.lifecycle

        # [ANALYSIS-LOG] Always log at INFO
        logger.info(
            "[ANALYSIS-LOG] LifecyclePolicyEngine.decide: path={}, lifecycle={}, "
            "artifacts_set={}, in_artifacts={}",
            change.path, lifecycle.value if lifecycle else "NONE",
            artifacts is not None, artifacts is not None and change.path in artifacts
        )

        # If tool explicitly reported this file as an artifact, always keep it
        if artifacts and change.path in artifacts:
            logger.info(
                "[ANALYSIS-LOG] LifecyclePolicyEngine.decide: path={} in artifacts set → PERSIST (override)",
                change.path
            )
            return LifecycleDecision.REGISTER_ARTIFACT

        # Decision table
        if lifecycle == FileLifecycle.TEMPORARY:
            if self.policy.backup_before_cleanup and self.policy.backup_dir:
                return LifecycleDecision.CLEANUP_BACKUP
            return LifecycleDecision.CLEANUP

        elif lifecycle == FileLifecycle.ARTIFACT:
            if self.policy.auto_register_artifacts:
                return LifecycleDecision.REGISTER_ARTIFACT
            return LifecycleDecision.PERSIST

        elif lifecycle == FileLifecycle.UNKNOWN:
            # For unknown files, always keep (audit)
            return LifecycleDecision.AUDIT

        # Default fallback
        return LifecycleDecision.AUDIT

    def apply(
        self,
        change_set: "ChangeSet",  # noqa: F821
        classifications: list[LifecycleClassification] | None = None,
        artifacts: list[str] | None = None,
    ) -> PolicyReport:
        """Apply policy to all changes in a ChangeSet.

        Args:
            change_set: The set of file changes.
            classifications: Optional pre-computed classifications.
            artifacts: Optional list of paths explicitly reported as artifacts by the tool
                       (e.g., from python_repl's artifacts JSON field). These paths will
                       never be deleted even if classified as TEMPORARY.

        Returns:
            PolicyReport with results of all decisions and actions.
        """
        # Build frozenset for fast lookup
        artifacts_set: frozenset[Path] | None = None
        if artifacts:
            artifacts_set = frozenset(Path(p) for p in artifacts)
            logger.info(
                "[ANALYSIS-LOG] LifecyclePolicyEngine.apply: artifacts_set={}, count={}",
                [str(p) for p in artifacts_set], len(artifacts_set)
            )

        report = PolicyReport()

        # Track resource usage for UNKNOWN files
        unknown_count = 0
        unknown_size_bytes = 0

        for i, change in enumerate(change_set.changes):
            classification = None
            if classifications and i < len(classifications):
                classification = classifications[i]
            if classification is None:
                classification = self._classifier.classify(
                    change.path,
                    getattr(change_set, "tool_name", ""),
                    change_type=change.change_type,
                )

            decision = self.decide(change, classification, artifacts_set)
            lifecycle = classification.lifecycle

            result = PolicyResult(
                path=change.path,
                decision=decision,
                lifecycle=lifecycle,
                classification=classification,
            )

            # Check resource limits for AUDIT files
            if decision == LifecycleDecision.AUDIT:
                unknown_count += 1
                try:
                    if change.path.exists():
                        unknown_size_bytes += change.path.stat().st_size
                except OSError:
                    pass

                if unknown_count > self.policy.max_unknown_files:
                    logger.warning(
                        "[LifecyclePolicyEngine] Unknown file limit reached: {} > {}",
                        unknown_count, self.policy.max_unknown_files
                    )
                    # Downgrade to CLEANUP if limit exceeded
                    if self.policy.backup_before_cleanup and self.policy.backup_dir:
                        decision = LifecycleDecision.CLEANUP_BACKUP
                    else:
                        decision = LifecycleDecision.CLEANUP
                    result.decision = decision

                unknown_size_mb = unknown_size_bytes / (1024 * 1024)
                if unknown_size_mb > self.policy.max_unknown_size_mb:
                    logger.warning(
                        "[LifecyclePolicyEngine] Unknown file size limit: {:.1f}MB > {}MB",
                        unknown_size_mb, self.policy.max_unknown_size_mb
                    )
                    # Downgrade to CLEANUP if size limit exceeded
                    if self.policy.backup_before_cleanup and self.policy.backup_dir:
                        decision = LifecycleDecision.CLEANUP_BACKUP
                    else:
                        decision = LifecycleDecision.CLEANUP
                    result.decision = decision

            # Execute the decision
            self._execute_decision(result, change)

            report.add_result(result)

        logger.info(
            "[LifecyclePolicyEngine] Policy applied: "
            "cleanup={}, persist={}, artifacts={}, audit={}, errors={}",
            report.cleanup_count,
            report.persist_count,
            report.artifact_count,
            report.audit_count,
            len(report.errors),
        )

        return report

    def _execute_decision(
        self,
        result: PolicyResult,
        change: "FileChange",  # noqa: F821
    ) -> None:
        """Execute a policy decision on a file."""
        decision = result.decision
        path = result.path

        if decision in {LifecycleDecision.CLEANUP, LifecycleDecision.CLEANUP_BACKUP}:
            # Check if file exists before cleanup
            if not path.exists():
                result.action_taken = "skipped: file does not exist"
                logger.debug(
                    "[LifecyclePolicyEngine] Cleanup skipped (not exists): {}",
                    path
                )
                return

            if decision == LifecycleDecision.CLEANUP_BACKUP:
                success = self._backup_file(path)
                if success:
                    result.action_taken = f"backed up to {self.policy.backup_dir}"
                else:
                    result.error = "backup failed, cleanup aborted"
                    result.decision = LifecycleDecision.AUDIT
                    result.action_taken = "cleanup aborted due to backup failure"
                    return

            # Perform cleanup
            try:
                path.unlink()
                result.action_taken = "deleted"
                logger.debug("[LifecyclePolicyEngine] Deleted: {}", path)
            except OSError as e:
                result.error = f"delete failed: {e}"
                logger.warning(
                    "[LifecyclePolicyEngine] Failed to delete {}: {}",
                    path, e
                )

        elif decision == LifecycleDecision.PERSIST:
            result.action_taken = "kept (no registration)"

        elif decision == LifecycleDecision.REGISTER_ARTIFACT:
            result.action_taken = "registered to ArtifactRegistry"

        elif decision == LifecycleDecision.AUDIT:
            result.action_taken = "kept for manual review"

    def _backup_file(self, path: Path) -> bool:
        """Backup a file before cleanup.

        Args:
            path: Path to the file to backup.

        Returns:
            True if backup succeeded, False otherwise.
        """
        if not self.policy.backup_dir:
            return False

        try:
            self.policy.backup_dir.mkdir(parents=True, exist_ok=True)

            # Generate backup filename with timestamp
            backup_name = f"{path.stem}_{path.stat().st_mtime:.0f}{path.suffix}"
            backup_path = self.policy.backup_dir / backup_name

            shutil.copy2(path, backup_path)
            logger.debug(
                "[LifecyclePolicyEngine] Backed up {} -> {}",
                path, backup_path
            )
            return True

        except (OSError, shutil.Error) as e:
            logger.warning(
                "[LifecyclePolicyEngine] Backup failed for {}: {}",
                path, e
            )
            return False

    def get_artifact_paths(self, report: PolicyReport) -> list[Path]:
        """Get list of artifact paths from a policy report.

        Args:
            report: The policy report to extract from.

        Returns:
            List of paths that should be registered as artifacts.
        """
        return [
            r.path for r in report.results
            if r.decision == LifecycleDecision.REGISTER_ARTIFACT
            and r.path.exists()
        ]

    def get_temporary_paths(self, report: PolicyReport) -> list[Path]:
        """Get list of temporary file paths from a policy report.

        Args:
            report: The policy report to extract from.

        Returns:
            List of paths marked for cleanup.
        """
        return [
            r.path for r in report.results
            if r.decision in {LifecycleDecision.CLEANUP, LifecycleDecision.CLEANUP_BACKUP}
        ]

    def get_audit_paths(self, report: PolicyReport) -> list[Path]:
        """Get list of audit (unknown) paths from a policy report.

        Args:
            report: The policy report to extract from.

        Returns:
            List of paths kept for manual review.
        """
        return [
            r.path for r in report.results
            if r.decision == LifecycleDecision.AUDIT
            and r.path.exists()
        ]
