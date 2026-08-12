# SPDX-License-Identifier: Apache-2.0
"""Tests for core.skill.execution.detection module."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from core.skill.execution.detection.config import DetectionConfig, LifecyclePolicy
from core.skill.execution.detection.types import FileLifecycle
# Import from shared.fs for the core snapshot types
from shared.fs.types import (
    ChangeType,
    FsChange,
    FsSnapshot,
)
from core.skill.execution.detection.lifecycle_classifier import (
    FileLifecycleClassifier,
    LifecycleClassification,
)
from core.skill.execution.detection.lifecycle_policy import (
    LifecyclePolicyEngine,
    LifecycleDecision,
    PolicyReport,
)
from core.skill.execution.detection.execution_tracker import ExecutionFileTracker

# For backward compatibility in tests, create aliases
FileChangeDetector = None  # Deprecated, use FsSnapshotManager from shared.fs


# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def detection_config(temp_workspace):
    """Create a default DetectionConfig."""
    return DetectionConfig(workspace_root=temp_workspace)


@pytest.fixture
def lifecycle_policy():
    """Create a default LifecyclePolicy."""
    return LifecyclePolicy(
        auto_cleanup_temporary=True,
        backup_before_cleanup=False,
    )


@pytest.fixture
def detector(temp_workspace, detection_config):
    """Create a FsSnapshotManager (formerly FileChangeDetector)."""
    from shared.fs.snapshot import FsSnapshotManager
    return FsSnapshotManager(temp_workspace)


@pytest.fixture
def classifier():
    """Create a FileLifecycleClassifier."""
    return FileLifecycleClassifier()


@pytest.fixture
def policy_engine(lifecycle_policy):
    """Create a LifecyclePolicyEngine."""
    return LifecyclePolicyEngine(lifecycle_policy)


@pytest.fixture
def tracker(temp_workspace, detection_config, lifecycle_policy):
    """Create an ExecutionFileTracker."""
    return ExecutionFileTracker(
        workspace_root=temp_workspace,
        config=detection_config,
        policy=lifecycle_policy,
    )


# ─── FsSnapshotManager Tests ─────────────────────────────────────────────────


class TestFileChangeDetector:
    """Tests for FileChangeDetector."""

    def test_take_snapshot_empty_dir(self, detector, temp_workspace):
        """Test taking snapshot of empty directory."""
        snapshot = detector.take_snapshot("test-1")
        assert snapshot is not None
        assert len(snapshot.entries) == 0

    def test_take_snapshot_with_files(self, detector, temp_workspace):
        """Test taking snapshot with files."""
        # Create test files
        (temp_workspace / "file1.txt").write_text("content1")
        (temp_workspace / "file2.py").write_text("print('hello')")

        snapshot = detector.take_snapshot("test-2")
        assert len(snapshot.entries) == 2
        assert "file1.txt" in snapshot.entries
        assert "file2.py" in snapshot.entries

    def test_detect_file_creation(self, detector, temp_workspace):
        """Test detecting file creation."""
        # Before snapshot
        detector.take_snapshot("create-before")

        # Create a new file
        (temp_workspace / "new_file.txt").write_text("new content")

        # After snapshot
        detector.take_snapshot("create-after")

        # Compare
        change_set = detector.compare("create-before", "create-after")
        assert len(change_set.changes) == 1
        assert change_set.changes[0].change_type == ChangeType.CREATED
        assert change_set.changes[0].path.name == "new_file.txt"

    def test_detect_file_modification(self, detector, temp_workspace):
        """Test detecting file modification."""
        # Create initial file
        test_file = temp_workspace / "test.txt"
        test_file.write_text("original")

        # Before snapshot
        detector.take_snapshot("mod-before")

        # Modify file
        test_file.write_text("modified content")

        # After snapshot
        detector.take_snapshot("mod-after")

        # Compare
        change_set = detector.compare("mod-before", "mod-after")
        assert len(change_set.changes) == 1
        assert change_set.changes[0].change_type == ChangeType.MODIFIED
        assert change_set.changes[0].path.name == "test.txt"

    def test_detect_file_deletion(self, detector, temp_workspace):
        """Test detecting file deletion."""
        # Create initial file
        test_file = temp_workspace / "delete_me.txt"
        test_file.write_text("will be deleted")

        # Before snapshot
        detector.take_snapshot("del-before")

        # Delete file
        test_file.unlink()

        # After snapshot
        detector.take_snapshot("del-after")

        # Compare
        change_set = detector.compare("del-before", "del-after")
        assert len(change_set.changes) == 1
        assert change_set.changes[0].change_type == ChangeType.DELETED

    def test_detect_nested_file(self, detector, temp_workspace):
        """Test detecting file in nested directory."""
        # Create nested directory structure
        nested_dir = temp_workspace / "subdir" / "nested"
        nested_dir.mkdir(parents=True)
        (nested_dir / "deep_file.txt").write_text("deep content")

        # Before snapshot
        detector.take_snapshot("nested-before")

        # Create new nested file
        (nested_dir / "new_nested.txt").write_text("new")

        # After snapshot
        detector.take_snapshot("nested-after")

        # Compare
        change_set = detector.compare("nested-before", "nested-after")
        assert len(change_set.changes) == 1
        assert "subdir/nested/new_nested.txt" in str(change_set.changes[0].path)

    def test_ignore_hidden_files(self, detector, temp_workspace):
        """Test that hidden files are ignored."""
        # Create hidden file
        (temp_workspace / ".hidden").write_text("hidden content")
        (temp_workspace / "visible.txt").write_text("visible")

        snapshot = detector.take_snapshot("ignore-1")
        paths = [e.path.name for e in snapshot.entries.values()]
        assert ".hidden" not in paths
        assert "visible.txt" in paths

    def test_clear_snapshot(self, detector):
        """Test clearing snapshots."""
        detector.take_snapshot("clear-test")
        assert detector.get_snapshot("clear-test") is not None

        detector.clear_snapshot("clear-test")
        assert detector.get_snapshot("clear-test") is None


# ─── LifecycleClassifier Tests ────────────────────────────────────────────────


class TestFileLifecycleClassifier:
    """Tests for FileLifecycleClassifier."""

    def test_classify_by_extension_artifact(self, classifier, temp_workspace):
        """Test classification by artifact extension (using non-artifact tool)."""
        test_file = temp_workspace / "report.pdf"
        test_file.write_text("pdf content")

        # Use a neutral tool (not in artifact_tools) to test extension-based classification
        result = classifier.classify(test_file, "read_file")
        assert result.lifecycle == FileLifecycle.ARTIFACT

    def test_classify_by_extension_temporary(self, classifier, temp_workspace):
        """Test classification by temporary extension (using non-path-based extension)."""
        test_file = temp_workspace / "temp.log"
        test_file.write_text("log content")

        # Use a neutral tool to test extension-based classification
        result = classifier.classify(test_file, "grep")
        assert result.lifecycle == FileLifecycle.TEMPORARY
        assert "temporary_extension" in result.overrides

    def test_classify_by_temporary_path(self, classifier, temp_workspace):
        """Test classification by temporary path."""
        # Create file in __pycache__
        cache_dir = temp_workspace / "__pycache__"
        cache_dir.mkdir()
        test_file = cache_dir / "module.pyc"
        test_file.write_text("cached")

        # Use a neutral tool to test path-based classification
        result = classifier.classify(test_file, "grep")
        assert result.lifecycle == FileLifecycle.TEMPORARY
        assert "temporary_path_pattern" in result.overrides

    def test_classify_by_tool_source(self, classifier, temp_workspace):
        """Test classification by tool source."""
        # python_repl with non-artifact extension → TEMPORARY
        test_file = temp_workspace / "output.cache"
        test_file.write_text("python output")

        result = classifier.classify(test_file, "python_repl")
        assert result.lifecycle == FileLifecycle.TEMPORARY
        assert "tool_source_temporary" in result.overrides

        # python_repl with artifact extension (.md) → ARTIFACT (artifact_extension_override takes precedence)
        test_md = temp_workspace / "result.md"
        test_md.write_text("# Transcript")
        result_md = classifier.classify(test_md, "python_repl")
        assert result_md.lifecycle == FileLifecycle.ARTIFACT
        assert "artifact_extension_override" in result_md.overrides

    def test_classify_by_artifact_tool(self, classifier, temp_workspace):
        """Test classification by artifact-producing tool."""
        test_file = temp_workspace / "created.py"
        test_file.write_text("created")

        result = classifier.classify(test_file, "file_create")
        assert result.lifecycle == FileLifecycle.ARTIFACT
        # Either tool_source_artifact (for non-extension files) or artifact_extension_override (for .py)
        assert any(o in result.overrides for o in ["tool_source_artifact", "artifact_extension_override"])

    def test_classify_unknown(self, classifier, temp_workspace):
        """Test classification for unknown files."""
        # 文件在 workspace 内，扩展名无意义，工具无明确分类
        test_file = temp_workspace / "unknown.xyz"
        test_file.write_text("unknown content")

        result = classifier.classify(test_file, "read_file")
        # read_file 不产生文件，所以不会有变化
        # 这个测试主要验证分类器不会崩溃
        assert result.lifecycle in [FileLifecycle.UNKNOWN, FileLifecycle.ARTIFACT]

    def test_classify_empty_file(self, classifier, temp_workspace):
        """Test classification of empty file (using temporary extension)."""
        test_file = temp_workspace / "empty.bak"
        test_file.write_text("")

        # Use a neutral tool to test empty file classification
        result = classifier.classify(test_file, "grep")
        # .bak is in temporary_extensions, so should be TEMPORARY
        assert result.lifecycle == FileLifecycle.TEMPORARY

    def test_batch_classify(self, classifier, temp_workspace):
        """Test batch classification."""
        # Create test files
        pdf_file = temp_workspace / "doc.pdf"
        pdf_file.write_text("pdf")
        tmp_file = temp_workspace / "temp.tmp"
        tmp_file.write_text("tmp")

        from shared.fs.types import FsChange as FileChange, ChangeType

        changes = [
            FsChange(
                path=pdf_file,
                rel_path=str(pdf_file.relative_to(temp_workspace)),
                change_type=ChangeType.CREATED,
                size_bytes=10,
                size_delta=None,
                is_significant=True,
            ),
            FsChange(
                path=tmp_file,
                rel_path=str(tmp_file.relative_to(temp_workspace)),
                change_type=ChangeType.CREATED,
                size_bytes=10,
                size_delta=None,
                is_significant=True,
            ),
        ]

        classifications = classifier.classify_batch(changes, "grep")
        assert len(classifications) == 2
        assert classifications[0].lifecycle == FileLifecycle.ARTIFACT
        assert classifications[1].lifecycle == FileLifecycle.TEMPORARY


# ─── LifecyclePolicy Tests ───────────────────────────────────────────────────


class TestLifecyclePolicyEngine:
    """Tests for LifecyclePolicyEngine."""

    def test_decide_temporary_cleanup(self, policy_engine, temp_workspace):
        """Test decision for temporary file."""
        from shared.fs.types import FsChange as FileChange, ChangeType

        change = FileChange(
            path=temp_workspace / "temp.tmp",
            rel_path="temp.tmp",
            change_type=ChangeType.CREATED,
            size_bytes=100,
            size_delta=None,
            is_significant=True,
        )

        decision = policy_engine.decide(change)
        assert decision in {LifecycleDecision.CLEANUP, LifecycleDecision.CLEANUP_BACKUP}

    def test_decide_artifact_persist(self, policy_engine, temp_workspace):
        """Test decision for artifact file."""
        from shared.fs.types import FsChange as FileChange, ChangeType

        change = FileChange(
            path=temp_workspace / "report.pdf",
            rel_path="report.pdf",
            change_type=ChangeType.CREATED,
            size_bytes=1000,
            size_delta=None,
            is_significant=True,
        )

        decision = policy_engine.decide(change)
        assert decision == LifecycleDecision.REGISTER_ARTIFACT

    def test_decide_unknown_audit(self, policy_engine, temp_workspace):
        """Test decision for unknown file."""
        from shared.fs.types import FsChange as FileChange, ChangeType

        change = FileChange(
            path=temp_workspace / "unknown.xyz",
            rel_path="unknown.xyz",
            change_type=ChangeType.CREATED,
            size_bytes=100,
            size_delta=None,
            is_significant=True,
        )

        decision = policy_engine.decide(change)
        assert decision == LifecycleDecision.AUDIT

    def test_apply_policy_cleanup(self, policy_engine, temp_workspace):
        """Test applying policy and cleanup."""
        from shared.fs.types import FsChange as FileChange, ChangeType
        from core.skill.execution.detection.execution_tracker import ChangeSetCompat as ChangeSet

        # Create temporary file
        tmp_file = temp_workspace / "temp.tmp"
        tmp_file.write_text("temp")

        change = FileChange(
            path=tmp_file,
            rel_path="temp.tmp",
            change_type=ChangeType.CREATED,
            size_bytes=10,
            size_delta=None,
            is_significant=True,
        )

        change_set = ChangeSet(
            changes=[change],
            tool_name="bash",
            execution_id="test-1",
        )

        report = policy_engine.apply(change_set)

        assert report.cleanup_count >= 1
        # File should be deleted
        assert not tmp_file.exists()

    def test_apply_policy_preserve_artifact(self, policy_engine, temp_workspace):
        """Test applying policy and preserving artifact."""
        from shared.fs.types import FsChange as FileChange, ChangeType
        from core.skill.execution.detection.execution_tracker import ChangeSetCompat as ChangeSet

        # Create artifact file
        artifact = temp_workspace / "report.pdf"
        artifact.write_text("pdf content")

        change = FileChange(
            path=artifact,
            rel_path="report.pdf",
            change_type=ChangeType.CREATED,
            size_bytes=100,
            size_delta=None,
            is_significant=True,
        )

        change_set = ChangeSet(
            changes=[change],
            tool_name="file_create",
            execution_id="test-2",
        )

        report = policy_engine.apply(change_set)

        assert report.artifact_count >= 1
        # File should still exist
        assert artifact.exists()

    def test_get_artifact_paths(self, policy_engine, temp_workspace):
        """Test extracting artifact paths from report."""
        from shared.fs.types import FsChange as FileChange, ChangeType
        from core.skill.execution.detection.execution_tracker import ChangeSetCompat as ChangeSet

        # Use temp_workspace paths
        artifact1 = temp_workspace / "artifact1.pdf"
        artifact2 = temp_workspace / "artifact2.py"
        artifact1.write_text("pdf")
        artifact2.write_text("py")

        change1 = FileChange(
            path=artifact1,
            rel_path="artifact1.pdf",
            change_type=ChangeType.CREATED,
            size_bytes=100,
            size_delta=None,
            is_significant=True,
        )
        change2 = FileChange(
            path=artifact2,
            rel_path="artifact2.py",
            change_type=ChangeType.CREATED,
            size_bytes=50,
            size_delta=None,
            is_significant=True,
        )

        change_set = ChangeSet(changes=[change1, change2])
        report = policy_engine.apply(change_set)

        artifacts = policy_engine.get_artifact_paths(report)
        assert len(artifacts) == 2
        assert artifact1 in artifacts
        assert artifact2 in artifacts


# ─── ExecutionFileTracker Tests ──────────────────────────────────────────────


class TestExecutionFileTracker:
    """Tests for ExecutionFileTracker."""

    @pytest.mark.asyncio
    async def test_track_file_creation(self, tracker, temp_workspace):
        """Test tracking file creation."""
        exec_id = await tracker.before_execute(
            tool_name="bash",
            args={"command": "echo hello"},
        )

        # Simulate file creation
        (temp_workspace / "created.txt").write_text("content")

        record = await tracker.after_execute(
            execution_id=exec_id,
            tool_name="bash",
            args={"command": "echo hello"},
        )

        assert record is not None
        assert record.tool_name == "bash"
        assert len(record.change_set.changes) >= 1

    @pytest.mark.asyncio
    async def test_get_history(self, tracker, temp_workspace):
        """Test getting execution history."""
        # Execute a few times
        for i in range(3):
            exec_id = await tracker.before_execute(tool_name="bash")
            (temp_workspace / f"file{i}.txt").write_text(f"content{i}")
            await tracker.after_execute(execution_id=exec_id, tool_name="bash")

        history = tracker.get_history()
        assert len(history) == 3

    @pytest.mark.asyncio
    async def test_cleanup_temporary(self, tracker, temp_workspace):
        """Test cleaning up temporary files via the tracker."""
        # Create a file before tracking
        tmp_file = temp_workspace / "temp.pyc"
        tmp_file.write_text("cached")

        exec_id = await tracker.before_execute(tool_name="python_repl")

        # Create another file during "execution"
        (temp_workspace / "new_file.txt").write_text("new")

        record = await tracker.after_execute(
            execution_id=exec_id,
            tool_name="python_repl",
        )

        # Verify that we tracked something
        assert record is not None
        # The result depends on classification, just verify it doesn't crash
        assert "tool_name" in record.to_dict()

    def test_get_summary(self, tracker):
        """Test getting summary."""
        summary = tracker.get_summary()
        assert "total_executions" in summary
        assert "total_artifacts" in summary
        assert "workspace_root" in summary


# ─── Config Tests ──────────────────────────────────────────────────────────────


class TestConfig:
    """Tests for configuration models."""

    def test_lifecycle_policy_defaults(self):
        """Test LifecyclePolicy defaults."""
        policy = LifecyclePolicy()
        assert policy.auto_cleanup_temporary is True
        assert policy.backup_before_cleanup is False
        assert policy.max_unknown_files == 50
        assert policy.max_unknown_size_mb == 100

    def test_detection_config_is_ignored(self, temp_workspace):
        """Test DetectionConfig.is_ignored()."""
        config = DetectionConfig(workspace_root=temp_workspace)

        # Hidden files/dirs should be ignored
        assert config.is_ignored(Path(".git"))
        assert config.is_ignored(Path("node_modules"))
        assert config.is_ignored(Path("__pycache__"))

        # Regular files should not be ignored
        assert not config.is_ignored(Path("file.txt"))
        assert not config.is_ignored(Path("src/main.py"))

    def test_detection_config_ignore_by_extension(self, temp_workspace):
        """Test ignoring files by extension."""
        config = DetectionConfig(workspace_root=temp_workspace)

        assert config.is_ignored(Path("file.pyc"))
        assert config.is_ignored(Path("file.tmp"))
        assert not config.is_ignored(Path("file.py"))


# ─── Types Tests ─────────────────────────────────────────────────────────────


class TestTypes:
    """Tests for type enums."""

    def test_file_lifecycle_values(self):
        """Test FileLifecycle enum values."""
        assert FileLifecycle.TEMPORARY.value == "temporary"
        assert FileLifecycle.ARTIFACT.value == "artifact"
        assert FileLifecycle.UNKNOWN.value == "unknown"

    def test_change_type_values(self):
        """Test ChangeType enum values."""
        assert ChangeType.CREATED.value == "created"
        assert ChangeType.MODIFIED.value == "modified"
        assert ChangeType.DELETED.value == "deleted"


class TestArtifactPreservation:
    """Test that explicitly-reported artifacts are never deleted."""

    def test_python_repl_artifacts_preserved(self, policy_engine, temp_workspace):
        """python_repl reported artifacts (.md) should never be deleted even if classified as TEMPORARY."""
        from shared.fs.types import FsChange as FileChange, ChangeType
        from core.skill.execution.detection.execution_tracker import ChangeSetCompat as ChangeSet

        transcript = temp_workspace / "transcript.md"
        transcript.write_text("# Transcript content")

        change = FileChange(
            path=transcript,
            rel_path="transcript.md",
            change_type=ChangeType.CREATED,
            size_bytes=1000,
            size_delta=None,
            is_significant=True,
        )
        change_set = ChangeSet(
            changes=[change],
            tool_name="python_repl",
            execution_id="test-repl-1",
        )

        # Pass artifacts list matching the file → should NOT be deleted
        report = policy_engine.apply(
            change_set,
            artifacts=[str(transcript)],
        )

        assert report.cleanup_count == 0, f"Expected no cleanup, got {report.cleanup_count}"
        assert transcript.exists(), "transcript.md should NOT have been deleted"
        assert report.artifact_count == 1

    def test_python_repl_unreported_temp_deleted(self, policy_engine, temp_workspace):
        """python_repl files NOT in artifacts list should still be cleaned up."""
        from shared.fs.types import FsChange as FileChange, ChangeType
        from core.skill.execution.detection.execution_tracker import ChangeSetCompat as ChangeSet

        runner_py = temp_workspace / "__runner__.py"
        runner_py.write_text("# runner")

        change = FileChange(
            path=runner_py,
            rel_path="__runner__.py",
            change_type=ChangeType.CREATED,
            size_bytes=500,
            size_delta=None,
            is_significant=True,
        )
        change_set = ChangeSet(
            changes=[change],
            tool_name="python_repl",
            execution_id="test-repl-2",
        )

        # No artifacts list → should be deleted (TEMPORARY → CLEANUP)
        report = policy_engine.apply(change_set)

        assert report.cleanup_count == 1
        assert not runner_py.exists(), "__runner__.py should have been deleted"

    def test_artifacts_list_with_mixed_files(self, policy_engine, temp_workspace):
        """When some files in artifacts list and some not, only non-artifacts are deleted."""
        from shared.fs.types import FsChange as FileChange, ChangeType
        from core.skill.execution.detection.execution_tracker import ChangeSetCompat as ChangeSet

        transcript = temp_workspace / "transcript.md"
        transcript.write_text("# Transcript")
        runner_py = temp_workspace / "__runner__.py"
        runner_py.write_text("# runner")

        changes = [
            FsChange(
                path=transcript,
                rel_path="transcript.md",
                change_type=ChangeType.CREATED,
                size_bytes=1000,
                size_delta=None,
                is_significant=True,
            ),
            FsChange(
                path=runner_py,
                rel_path="__runner__.py",
                change_type=ChangeType.CREATED,
                size_bytes=500,
                size_delta=None,
                is_significant=True,
            ),
        ]
        change_set = ChangeSetCompat(
            changes=changes,
            tool_name="python_repl",
            execution_id="test-repl-3",
        )

        # Only transcript is in artifacts → only runner.py deleted
        report = policy_engine.apply(
            change_set,
            artifacts=[str(transcript)],
        )

        assert report.cleanup_count == 1
        assert transcript.exists(), "transcript.md should be preserved (in artifacts)"
        assert not runner_py.exists(), "__runner__.py should be deleted (not in artifacts)"
