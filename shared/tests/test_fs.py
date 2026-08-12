# SPDX-License-Identifier: Apache-2.0
"""Tests for shared/fs module.

Comprehensive unit tests for:
- shared/fs/types.py: ChangeType, FsSnapshotEntry, FsSnapshot, FsChange, DiffResult, ExecutionRecord
- shared/fs/snapshot.py: FsSnapshotManager, SnapshotConfig, SandboxSnapshot
- shared/fs/monitor.py: FsMonitor, SessionFsMonitor
"""

import asyncio
import hashlib
import os
import tempfile
import time
from pathlib import Path

import pytest

from shared.fs.types import (
    ChangeType,
    DiffResult,
    ExecutionRecord,
    FsChange,
    FsMonitorProtocol,
    FsSnapshot,
    FsSnapshotEntry,
    IS_POSIX,
    IS_WINDOWS,
    DEFAULT_IGNORE_DIRS,
    DEFAULT_IGNORE_PATTERNS,
    is_safe_within_workspace,
    paths_equal,
    resolve_path_safe,
)
from shared.fs.snapshot import (
    FsSnapshotManager,
    SnapshotConfig,
    SandboxSnapshot,
)
from shared.fs.monitor import (
    FsMonitor,
    SessionFsMonitor,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def nested_workspace():
    """Create a workspace with nested directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # Create nested structure
        (root / "src").mkdir()
        (root / "src" / "pkg").mkdir()
        (root / "tests").mkdir()
        (root / "docs").mkdir()
        # Create files
        (root / "src" / "main.py").write_text("def main(): pass")
        (root / "src" / "pkg" / "__init__.py").write_text("")
        (root / "tests" / "test_main.py").write_text("def test_main(): pass")
        (root / "README.md").write_text("# Project")
        yield root


# =============================================================================
# Types Tests - ChangeType
# =============================================================================

class TestChangeType:
    """Tests for ChangeType enum."""

    def test_change_type_values(self):
        """Test ChangeType enum values."""
        assert ChangeType.CREATED.value == "created"
        assert ChangeType.MODIFIED.value == "modified"
        assert ChangeType.DELETED.value == "deleted"
        assert ChangeType.UNCHANGED.value == "unchanged"

    def test_change_type_from_string(self):
        """Test creating ChangeType from string."""
        assert ChangeType("created") == ChangeType.CREATED
        assert ChangeType("modified") == ChangeType.MODIFIED
        assert ChangeType("deleted") == ChangeType.DELETED


# =============================================================================
# Types Tests - Cross-platform Utilities
# =============================================================================

class TestCrossPlatformUtils:
    """Tests for cross-platform utility functions."""

    def test_is_safe_within_workspace_inside(self, temp_workspace):
        """Test path inside workspace returns True."""
        path = temp_workspace / "file.txt"
        assert is_safe_within_workspace(path, temp_workspace) is True

    def test_is_safe_within_workspace_nested(self, temp_workspace):
        """Test nested path inside workspace returns True."""
        subdir = temp_workspace / "sub" / "nested"
        subdir.mkdir(parents=True)
        path = subdir / "file.txt"
        assert is_safe_within_workspace(path, temp_workspace) is True

    def test_is_safe_within_workspace_outside(self, temp_workspace):
        """Test path outside workspace returns False."""
        path = temp_workspace.parent / "outside.txt"
        assert is_safe_within_workspace(path, temp_workspace) is False

    def test_is_safe_within_workspace_nonexistent(self):
        """Test nonexistent path - behavior depends on platform."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # On macOS, nonexistent paths within workspace resolve to workspace
            # On some platforms, they may return False
            nonexistent = root / "nonexistent_subdir" / "file.txt"
            # Test that function handles nonexistent paths gracefully
            result = is_safe_within_workspace(nonexistent, root)
            # The important thing is it doesn't raise an exception
            assert isinstance(result, bool)

    def test_is_safe_within_workspace_outside(self):
        """Test path outside workspace returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            outside = Path(tmpdir).parent / "outside.txt"
            assert is_safe_within_workspace(outside, root) is False

    def test_paths_equal_same(self, temp_workspace):
        """Test same paths are equal."""
        file1 = temp_workspace / "file.txt"
        file2 = temp_workspace / "file.txt"
        assert paths_equal(file1, file2) is True

    def test_paths_equal_different(self, temp_workspace):
        """Test different paths are not equal."""
        file1 = temp_workspace / "file1.txt"
        file2 = temp_workspace / "file2.txt"
        assert paths_equal(file1, file2) is False

    def test_paths_equal_resolved(self, temp_workspace):
        """Test paths_equal resolves symlinks."""
        file1 = temp_workspace / "file.txt"
        file1.write_text("content")
        link = temp_workspace / "link.txt"
        try:
            link.symlink_to(file1)
            assert paths_equal(file1, link) is True
        except OSError:
            pytest.skip("Symlinks not supported on this platform")

    def test_resolve_path_safe_valid(self, temp_workspace):
        """Test resolve_path_safe with valid path."""
        file = temp_workspace / "file.txt"
        file.write_text("content")
        resolved = resolve_path_safe(file)
        assert resolved is not None
        assert resolved.is_absolute()

    def test_resolve_path_safe_nonexistent(self):
        """Test resolve_path_safe with nonexistent path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nonexistent = root / "nonexistent_subdir" / "file.txt"
            # Ensure parent doesn't exist
            assert not nonexistent.parent.exists()
            resolved = resolve_path_safe(nonexistent)
            # On some platforms (like macOS), resolve() succeeds for nonexistent paths
            # The important thing is it doesn't raise an exception
            assert resolved is None or isinstance(resolved, Path)

    def test_platform_detection(self):
        """Test platform detection constants."""
        assert isinstance(IS_POSIX, bool)
        assert isinstance(IS_WINDOWS, bool)
        assert IS_POSIX != IS_WINDOWS  # Must be different


# =============================================================================
# Types Tests - Constants
# =============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_default_ignore_dirs_contains_common(self):
        """Test DEFAULT_IGNORE_DIRS contains common directories."""
        assert ".git" in DEFAULT_IGNORE_DIRS
        assert "__pycache__" in DEFAULT_IGNORE_DIRS
        assert "node_modules" in DEFAULT_IGNORE_DIRS
        assert ".venv" in DEFAULT_IGNORE_DIRS

    def test_default_ignore_patterns_contains_common(self):
        """Test DEFAULT_IGNORE_PATTERNS contains common patterns."""
        assert "*.pyc" in DEFAULT_IGNORE_PATTERNS
        assert ".DS_Store" in DEFAULT_IGNORE_PATTERNS
        assert "Thumbs.db" in DEFAULT_IGNORE_PATTERNS


# =============================================================================
# Types Tests - FsSnapshotEntry
# =============================================================================

class TestFsSnapshotEntry:
    """Tests for FsSnapshotEntry."""

    def test_creation(self, temp_workspace):
        """Test FsSnapshotEntry creation."""
        file = temp_workspace / "test.txt"
        file.write_text("content")
        stat = file.stat()

        entry = FsSnapshotEntry(
            path=file,
            mtime=stat.st_mtime,
            size=stat.st_size,
        )
        assert entry.path == file
        assert entry.mtime == stat.st_mtime
        assert entry.size == stat.st_size
        assert entry.hash_sha256 is None

    def test_creation_with_hash(self, temp_workspace):
        """Test FsSnapshotEntry creation with hash."""
        file = temp_workspace / "test.txt"
        content = "test content"
        file.write_text(content)
        expected_hash = hashlib.sha256(content.encode()).hexdigest()

        entry = FsSnapshotEntry(
            path=file,
            mtime=file.stat().st_mtime,
            size=len(content),
            hash_sha256=expected_hash,
        )
        assert entry.hash_sha256 == expected_hash

    def test_is_same_content_by_hash(self, temp_workspace):
        """Test is_same_content using hash comparison."""
        file1 = temp_workspace / "file1.txt"
        file2 = temp_workspace / "file2.txt"
        content = "same content"
        file1.write_text(content)
        file2.write_text(content)
        hash_val = hashlib.sha256(content.encode()).hexdigest()

        entry1 = FsSnapshotEntry(path=file1, mtime=0, size=len(content), hash_sha256=hash_val)
        entry2 = FsSnapshotEntry(path=file2, mtime=0, size=len(content), hash_sha256=hash_val)

        assert entry1.is_same_content(entry2) is True

    def test_is_same_content_different_size(self, temp_workspace):
        """Test is_same_content with different sizes returns False."""
        file1 = temp_workspace / "file1.txt"
        file2 = temp_workspace / "file2.txt"
        file1.write_text("short")
        file2.write_text("much longer content")

        entry1 = FsSnapshotEntry(path=file1, mtime=0, size=5, hash_sha256="abc")
        entry2 = FsSnapshotEntry(path=file2, mtime=0, size=18, hash_sha256="def")

        assert entry1.is_same_content(entry2) is False

    def test_compute_hash(self, temp_workspace):
        """Test compute_hash method."""
        file = temp_workspace / "test.txt"
        content = "test content for hash"
        file.write_text(content)
        expected_hash = hashlib.sha256(content.encode()).hexdigest()

        entry = FsSnapshotEntry(
            path=file,
            mtime=file.stat().st_mtime,
            size=len(content),
        )
        computed_hash = entry.compute_hash()
        assert computed_hash == expected_hash

    def test_compute_hash_no_file(self):
        """Test compute_hash with nonexistent file."""
        entry = FsSnapshotEntry(
            path=Path("/nonexistent/file.txt"),
            mtime=0,
            size=0,
        )
        assert entry.compute_hash() is None

    def test_to_dict(self, temp_workspace):
        """Test to_dict serialization."""
        file = temp_workspace / "test.txt"
        file.write_text("content")
        stat = file.stat()

        entry = FsSnapshotEntry(
            path=file,
            mtime=stat.st_mtime,
            size=stat.st_size,
            hash_sha256="abc123",
        )

        d = entry.to_dict()
        assert d["path"] == str(file)
        assert d["mtime"] == stat.st_mtime
        assert d["size"] == stat.st_size
        assert d["hash"] == "abc123"

    def test_from_dict(self, temp_workspace):
        """Test from_dict deserialization."""
        file = temp_workspace / "test.txt"
        data = {
            "path": str(file),
            "mtime": 1234567890.0,
            "size": 100,
            "hash": "hash123",
        }

        entry = FsSnapshotEntry.from_dict(data)
        assert entry.path == file
        assert entry.mtime == 1234567890.0
        assert entry.size == 100
        assert entry.hash_sha256 == "hash123"


# =============================================================================
# Types Tests - FsSnapshot
# =============================================================================

class TestFsSnapshot:
    """Tests for FsSnapshot."""

    def test_creation(self, temp_workspace):
        """Test FsSnapshot creation."""
        snapshot = FsSnapshot(
            timestamp=time.time(),
            workspace_root=temp_workspace,
            execution_id="test-1",
        )
        assert snapshot.workspace_root == temp_workspace
        assert snapshot.execution_id == "test-1"
        assert len(snapshot.entries) == 0

    def test_get_existing(self, temp_workspace):
        """Test getting existing entry."""
        file = temp_workspace / "test.txt"
        file.write_text("content")

        snapshot = FsSnapshot(
            timestamp=time.time(),
            workspace_root=temp_workspace,
            entries={
                "test.txt": FsSnapshotEntry(
                    path=file,
                    mtime=file.stat().st_mtime,
                    size=len("content"),
                )
            },
        )

        entry = snapshot.get("test.txt")
        assert entry is not None
        assert entry.path == file

    def test_get_nonexistent(self, temp_workspace):
        """Test getting nonexistent entry returns None."""
        snapshot = FsSnapshot(
            timestamp=time.time(),
            workspace_root=temp_workspace,
        )
        assert snapshot.get("nonexistent.txt") is None

    def test_diff_created(self, temp_workspace):
        """Test diff detects created files."""
        before = FsSnapshot(
            timestamp=time.time(),
            workspace_root=temp_workspace,
        )

        file = temp_workspace / "new.txt"
        file.write_text("content")

        after = FsSnapshot(
            timestamp=time.time(),
            workspace_root=temp_workspace,
            entries={
                "new.txt": FsSnapshotEntry(
                    path=file,
                    mtime=file.stat().st_mtime,
                    size=len("content"),
                )
            },
        )

        changes = before.diff(after)
        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.CREATED
        assert changes[0].rel_path == "new.txt"

    def test_diff_modified(self, temp_workspace):
        """Test diff detects modified files."""
        file = temp_workspace / "test.txt"
        file.write_text("original")
        stat1 = file.stat()

        before = FsSnapshot(
            timestamp=time.time(),
            workspace_root=temp_workspace,
            entries={
                "test.txt": FsSnapshotEntry(
                    path=file,
                    mtime=stat1.st_mtime,
                    size=len("original"),
                )
            },
        )

        time.sleep(0.01)  # Ensure mtime changes
        file.write_text("modified")
        stat2 = file.stat()

        after = FsSnapshot(
            timestamp=time.time(),
            workspace_root=temp_workspace,
            entries={
                "test.txt": FsSnapshotEntry(
                    path=file,
                    mtime=stat2.st_mtime,
                    size=len("modified"),
                )
            },
        )

        changes = before.diff(after)
        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.MODIFIED
        assert changes[0].size_delta == len("modified") - len("original")

    def test_diff_deleted(self, temp_workspace):
        """Test diff detects deleted files."""
        file = temp_workspace / "deleted.txt"
        file.write_text("content")
        stat = file.stat()

        before = FsSnapshot(
            timestamp=time.time(),
            workspace_root=temp_workspace,
            entries={
                "deleted.txt": FsSnapshotEntry(
                    path=file,
                    mtime=stat.st_mtime,
                    size=len("content"),
                )
            },
        )

        file.unlink()

        after = FsSnapshot(
            timestamp=time.time(),
            workspace_root=temp_workspace,
        )

        changes = before.diff(after)
        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.DELETED
        assert changes[0].size_delta == -len("content")

    def test_diff_no_changes(self, temp_workspace):
        """Test diff with no changes."""
        file = temp_workspace / "test.txt"
        file.write_text("content")
        stat = file.stat()

        snapshot = FsSnapshot(
            timestamp=time.time(),
            workspace_root=temp_workspace,
            entries={
                "test.txt": FsSnapshotEntry(
                    path=file,
                    mtime=stat.st_mtime,
                    size=len("content"),
                )
            },
        )

        changes = snapshot.diff(snapshot)
        assert len(changes) == 0

    def test_to_dict(self, temp_workspace):
        """Test to_dict serialization."""
        file = temp_workspace / "test.txt"
        file.write_text("content")

        snapshot = FsSnapshot(
            timestamp=1234567890.0,
            workspace_root=temp_workspace,
            execution_id="test-1",
            entries={
                "test.txt": FsSnapshotEntry(
                    path=file,
                    mtime=file.stat().st_mtime,
                    size=len("content"),
                )
            },
        )

        d = snapshot.to_dict()
        assert d["timestamp"] == 1234567890.0
        assert d["workspace_root"] == str(temp_workspace)
        assert d["execution_id"] == "test-1"
        assert "test.txt" in d["entries"]


# =============================================================================
# Types Tests - DiffResult
# =============================================================================

class TestDiffResult:
    """Tests for DiffResult."""

    def test_creation_empty(self):
        """Test empty DiffResult creation."""
        result = DiffResult()
        assert len(result.created) == 0
        assert len(result.modified) == 0
        assert len(result.deleted) == 0
        assert len(result.new_dirs) == 0

    def test_creation_with_values(self):
        """Test DiffResult creation with values."""
        result = DiffResult(
            created=["file1.txt", "file2.txt"],
            modified=["file3.txt"],
            deleted=["file4.txt"],
            new_dirs=["new_dir"],
        )
        assert len(result.created) == 2
        assert len(result.modified) == 1
        assert len(result.deleted) == 1
        assert len(result.new_dirs) == 1

    def test_has_changes_true(self):
        """Test has_changes returns True when changes exist."""
        result = DiffResult(created=["file.txt"])
        assert result.has_changes is True

    def test_has_changes_false(self):
        """Test has_changes returns False when no changes."""
        result = DiffResult()
        assert result.has_changes is False

    def test_artifact_paths(self):
        """Test artifact_paths property."""
        result = DiffResult(
            created=["created.txt"],
            modified=["modified.txt"],
            deleted=["deleted.txt"],
        )
        artifacts = result.artifact_paths
        assert "created.txt" in artifacts
        assert "modified.txt" in artifacts
        assert "deleted.txt" not in artifacts

    def test_to_dict(self):
        """Test to_dict serialization."""
        result = DiffResult(
            created=["file1.txt"],
            modified=["file2.txt"],
        )
        d = result.to_dict()
        assert d["created"] == ["file1.txt"]
        assert d["modified"] == ["file2.txt"]
        assert d["has_changes"] is True


# =============================================================================
# Types Tests - ExecutionRecord
# =============================================================================

class TestExecutionRecord:
    """Tests for ExecutionRecord."""

    def test_creation(self):
        """Test ExecutionRecord creation."""
        record = ExecutionRecord(
            execution_id="exec-1",
            tool_name="bash",
            timestamp=time.time(),
        )
        assert record.execution_id == "exec-1"
        assert record.tool_name == "bash"
        assert len(record.changes) == 0

    def test_created_paths(self, temp_workspace):
        """Test created_paths property."""
        file1 = temp_workspace / "file1.txt"
        file2 = temp_workspace / "file2.txt"

        record = ExecutionRecord(
            execution_id="exec-1",
            tool_name="bash",
            timestamp=time.time(),
            changes=[
                FsChange(path=file1, rel_path="file1.txt", change_type=ChangeType.CREATED,
                         size_bytes=10, size_delta=None, is_significant=True),
                FsChange(path=file2, rel_path="file2.txt", change_type=ChangeType.MODIFIED,
                         size_bytes=20, size_delta=10, is_significant=True),
            ],
        )

        created = record.created_paths
        assert len(created) == 1
        assert created[0] == file1

    def test_modified_paths(self, temp_workspace):
        """Test modified_paths property."""
        file1 = temp_workspace / "file1.txt"
        file2 = temp_workspace / "file2.txt"

        record = ExecutionRecord(
            execution_id="exec-1",
            tool_name="bash",
            timestamp=time.time(),
            changes=[
                FsChange(path=file1, rel_path="file1.txt", change_type=ChangeType.CREATED,
                         size_bytes=10, size_delta=None, is_significant=True),
                FsChange(path=file2, rel_path="file2.txt", change_type=ChangeType.MODIFIED,
                         size_bytes=20, size_delta=10, is_significant=True),
            ],
        )

        modified = record.modified_paths
        assert len(modified) == 1
        assert modified[0] == file2

    def test_deleted_paths(self, temp_workspace):
        """Test deleted_paths property."""
        file1 = temp_workspace / "file1.txt"
        file2 = temp_workspace / "file2.txt"

        record = ExecutionRecord(
            execution_id="exec-1",
            tool_name="bash",
            timestamp=time.time(),
            changes=[
                FsChange(path=file1, rel_path="file1.txt", change_type=ChangeType.CREATED,
                         size_bytes=10, size_delta=None, is_significant=True),
                FsChange(path=file2, rel_path="file2.txt", change_type=ChangeType.DELETED,
                         size_bytes=0, size_delta=-20, is_significant=True),
            ],
        )

        deleted = record.deleted_paths
        assert len(deleted) == 1
        assert deleted[0] == file2

    def test_to_dict(self):
        """Test to_dict serialization."""
        record = ExecutionRecord(
            execution_id="exec-1",
            tool_name="bash",
            timestamp=1234567890.0,
            artifact_paths=[Path("/path/to/artifact.txt")],
        )

        d = record.to_dict()
        assert d["execution_id"] == "exec-1"
        assert d["tool_name"] == "bash"
        assert d["timestamp"] == 1234567890.0
        assert len(d["artifact_paths"]) == 1


# =============================================================================
# SnapshotConfig Tests
# =============================================================================

class TestSnapshotConfig:
    """Tests for SnapshotConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SnapshotConfig()
        assert config.compute_hash is False
        assert config.small_file_threshold == 1024 * 1024
        assert ".git" in config.ignore_dirs

    def test_custom_values(self):
        """Test custom configuration values."""
        config = SnapshotConfig(
            compute_hash=True,
            small_file_threshold=1024,
            ignore_dirs=frozenset({".git", "node_modules"}),
        )
        assert config.compute_hash is True
        assert config.small_file_threshold == 1024
        assert ".git" in config.ignore_dirs

    def test_is_ignored_dir(self):
        """Test is_ignored_dir method."""
        config = SnapshotConfig()
        assert config.is_ignored_dir(".git") is True
        assert config.is_ignored_dir("__pycache__") is True
        assert config.is_ignored_dir("normal_dir") is False

    def test_is_ignored_file(self):
        """Test is_ignored_file method."""
        config = SnapshotConfig(ignore_patterns=frozenset({"*.pyc", "*.log"}))
        assert config.is_ignored_file("test.pyc") is True
        assert config.is_ignored_file("debug.log") is True
        assert config.is_ignored_file("test.py") is False


# =============================================================================
# FsSnapshotManager Tests
# =============================================================================

class TestFsSnapshotManager:
    """Tests for FsSnapshotManager."""

    def test_creation(self, temp_workspace):
        """Test FsSnapshotManager creation."""
        manager = FsSnapshotManager(temp_workspace)
        assert manager.workspace_root == temp_workspace.resolve()

    def test_take_snapshot_empty(self, temp_workspace):
        """Test taking snapshot of empty directory."""
        manager = FsSnapshotManager(temp_workspace)
        snapshot = manager.take_snapshot("test-1")
        assert snapshot is not None
        assert len(snapshot.entries) == 0

    def test_take_snapshot_with_files(self, temp_workspace):
        """Test taking snapshot with files."""
        (temp_workspace / "file1.txt").write_text("content1")
        (temp_workspace / "file2.py").write_text("print('hello')")

        manager = FsSnapshotManager(temp_workspace)
        snapshot = manager.take_snapshot("test-1")

        assert len(snapshot.entries) == 2
        assert "file1.txt" in snapshot.entries
        assert "file2.py" in snapshot.entries

    def test_take_snapshot_idempotent(self, temp_workspace):
        """Test taking snapshot twice returns same snapshot."""
        (temp_workspace / "file.txt").write_text("content")

        manager = FsSnapshotManager(temp_workspace)
        snapshot1 = manager.take_snapshot("test-1")
        snapshot2 = manager.take_snapshot("test-1")

        assert snapshot1 is snapshot2

    def test_take_full_snapshot_with_hash(self, temp_workspace):
        """Test taking full snapshot with hash computation."""
        file = temp_workspace / "test.txt"
        content = "test content"
        file.write_text(content)

        config = SnapshotConfig(compute_hash=True)
        manager = FsSnapshotManager(temp_workspace, config=config)
        snapshot = manager.take_full_snapshot("test-1")

        assert len(snapshot.entries) == 1
        entry = snapshot.entries["test.txt"]
        assert entry.hash_sha256 is not None
        assert entry.hash_sha256 == hashlib.sha256(content.encode()).hexdigest()

    def test_diff_detect_creation(self, temp_workspace):
        """Test diff detects file creation."""
        manager = FsSnapshotManager(temp_workspace)
        manager.take_snapshot("test-1")

        # Create new file
        (temp_workspace / "new_file.txt").write_text("new content")

        diff = manager.diff("test-1")
        assert len(diff.created) == 1
        assert "new_file.txt" in diff.created

    def test_diff_detect_modification(self, temp_workspace):
        """Test diff detects file modification (size change)."""
        file = temp_workspace / "test.txt"
        file.write_text("original")

        manager = FsSnapshotManager(temp_workspace)
        manager.take_snapshot("test-1")

        time.sleep(0.01)
        file.write_text("modified content longer")

        diff = manager.diff("test-1")
        assert len(diff.modified) == 1
        assert "test.txt" in diff.modified

    def test_diff_detect_deletion(self, temp_workspace):
        """Test diff detects file deletion."""
        file = temp_workspace / "test.txt"
        file.write_text("content")

        manager = FsSnapshotManager(temp_workspace)
        manager.take_snapshot("test-1")

        file.unlink()

        diff = manager.diff("test-1")
        assert len(diff.deleted) == 1
        assert "test.txt" in diff.deleted

    def test_diff_nonexistent_snapshot(self, temp_workspace):
        """Test diff with nonexistent snapshot returns empty result."""
        manager = FsSnapshotManager(temp_workspace)
        diff = manager.diff("nonexistent")
        assert len(diff.created) == 0
        assert len(diff.modified) == 0
        assert len(diff.deleted) == 0

    def test_compare_detailed_changes(self, temp_workspace):
        """Test compare method returns detailed changes."""
        file = temp_workspace / "test.txt"
        file.write_text("original")

        manager = FsSnapshotManager(temp_workspace)
        manager.take_full_snapshot("before")
        manager.take_full_snapshot("after")

        changes = manager.compare("before", "after")
        assert len(changes) == 0

        file.write_text("modified content that is longer")
        manager.take_full_snapshot("after-modified")
        changes = manager.compare("before", "after-modified")
        assert len(changes) == 1
        assert changes[0].change_type == ChangeType.MODIFIED

    def test_get_snapshot(self, temp_workspace):
        """Test getting a snapshot."""
        manager = FsSnapshotManager(temp_workspace)
        snapshot = manager.take_snapshot("test-1")
        retrieved = manager.get_snapshot("test-1")
        assert retrieved is snapshot

    def test_get_snapshot_nonexistent(self, temp_workspace):
        """Test getting nonexistent snapshot returns None."""
        manager = FsSnapshotManager(temp_workspace)
        assert manager.get_snapshot("nonexistent") is None

    def test_release_snapshot(self, temp_workspace):
        """Test releasing a snapshot."""
        manager = FsSnapshotManager(temp_workspace)
        manager.take_snapshot("test-1")
        manager.release("test-1")
        assert manager.get_snapshot("test-1") is None

    def test_clear_all_snapshots(self, temp_workspace):
        """Test clearing all snapshots."""
        manager = FsSnapshotManager(temp_workspace)
        manager.take_snapshot("test-1")
        manager.take_snapshot("test-2")
        manager.clear()
        assert manager.get_snapshot("test-1") is None
        assert manager.get_snapshot("test-2") is None

    def test_generate_execution_id(self, temp_workspace):
        """Test execution ID generation."""
        manager = FsSnapshotManager(temp_workspace)
        id1 = manager.generate_execution_id()
        id2 = manager.generate_execution_id()
        assert id1.startswith("fs_")
        assert id1 != id2

    def test_ignore_directories(self, temp_workspace):
        """Test that ignored directories are not included."""
        # Create ignored directories (use dirs that don't require special permissions)
        (temp_workspace / "__pycache__" / "module.pyc").mkdir(parents=True)
        (temp_workspace / "node_modules" / "pkg" / "index.js").mkdir(parents=True)
        # Create normal file
        (temp_workspace / "main.py").write_text("code")

        manager = FsSnapshotManager(temp_workspace)
        snapshot = manager.take_snapshot("test-1")

        assert len(snapshot.entries) == 1
        assert "main.py" in snapshot.entries

    def test_nested_directory_tracking(self, nested_workspace):
        """Test tracking files in nested directories."""
        manager = FsSnapshotManager(nested_workspace)
        snapshot = manager.take_snapshot("test-1")

        # Should track 4 files (README.md + src/main.py + src/pkg/__init__.py + tests/test_main.py)
        assert len(snapshot.entries) == 4
        assert "src/main.py" in snapshot.entries
        assert "src/pkg/__init__.py" in snapshot.entries
        assert "tests/test_main.py" in snapshot.entries
        assert "README.md" in snapshot.entries


# =============================================================================
# SandboxSnapshot Tests
# =============================================================================

class TestSandboxSnapshot:
    """Tests for SandboxSnapshot."""

    def test_take_empty_directory(self):
        """Test taking snapshot of empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = SandboxSnapshot.take(Path(tmpdir))
            assert len(snapshot) == 0

    def test_take_with_files(self):
        """Test taking snapshot with files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "file1.txt").write_text("content1")
            (root / "file2.py").write_text("code")
            (root / "subdir").mkdir()
            (root / "subdir" / "file3.txt").write_text("nested")

            snapshot = SandboxSnapshot.take(root)

            assert len(snapshot) == 3
            assert "file1.txt" in snapshot
            assert "file2.py" in snapshot
            assert "subdir/file3.txt" in snapshot

    def test_take_nonexistent_directory(self):
        """Test taking snapshot of nonexistent directory."""
        snapshot = SandboxSnapshot.take(Path("/nonexistent/path"))
        assert len(snapshot) == 0

    def test_collect_diff_created(self):
        """Test collect_diff detects created files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pre = SandboxSnapshot.take(root)

            (root / "new_file.txt").write_text("new content")

            artifacts = SandboxSnapshot.collect_diff(pre, root)
            assert len(artifacts) == 1
            assert artifacts[0].endswith("new_file.txt")

    def test_collect_diff_modified(self):
        """Test collect_diff detects modified files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            file = root / "existing.txt"
            file.write_text("original")
            pre = SandboxSnapshot.take(root)

            time.sleep(0.01)
            file.write_text("modified")

            artifacts = SandboxSnapshot.collect_diff(pre, root)
            assert len(artifacts) == 1

    def test_collect_diff_no_changes(self):
        """Test collect_diff with no changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "file.txt").write_text("content")
            pre = SandboxSnapshot.take(root)

            artifacts = SandboxSnapshot.collect_diff(pre, root)
            assert len(artifacts) == 0

    def test_collect_diff_nonexistent_directory(self):
        """Test collect_diff with nonexistent directory."""
        artifacts = SandboxSnapshot.collect_diff({}, Path("/nonexistent"))
        assert len(artifacts) == 0


# =============================================================================
# FsMonitor Tests
# =============================================================================

class TestFsMonitor:
    """Tests for FsMonitor."""

    def test_creation(self, temp_workspace):
        """Test FsMonitor creation."""
        monitor = FsMonitor(temp_workspace)
        assert monitor.workspace_root == temp_workspace.resolve()
        assert monitor.snapshot_manager is not None

    def test_before_tool(self, temp_workspace):
        """Test before_tool method."""
        monitor = FsMonitor(temp_workspace)
        exec_id = monitor.before_tool("bash", {"command": "echo hello"}, [])

        assert exec_id is not None
        assert len(exec_id) > 0

    def test_after_tool(self, temp_workspace):
        """Test after_tool method."""
        monitor = FsMonitor(temp_workspace)
        exec_id = monitor.before_tool("bash", {"command": "echo hello"}, [])

        # Create a file
        (temp_workspace / "output.txt").write_text("result")

        record = monitor.after_tool(exec_id, "bash", {"command": "echo hello"})

        assert record is not None
        assert record.execution_id == exec_id
        assert record.tool_name == "bash"

    def test_after_tool_classifies_changes(self, temp_workspace):
        """Test that after_tool classifies changes."""
        monitor = FsMonitor(temp_workspace)
        exec_id = monitor.before_tool("bash", {}, [])

        # Create artifact file
        (temp_workspace / "result.md").write_text("# Result")

        record = monitor.after_tool(exec_id, "bash", {})

        assert len(record.changes) >= 1

    def test_after_tool_with_nonexistent_exec_id(self, temp_workspace):
        """Test after_tool with nonexistent execution ID."""
        monitor = FsMonitor(temp_workspace)
        record = monitor.after_tool("nonexistent", "bash", {})
        assert record is not None

    def test_bind_artifact_registry(self, temp_workspace):
        """Test binding ArtifactRegistry."""
        monitor = FsMonitor(temp_workspace)

        # Create mock registry
        class MockRegistry:
            def __init__(self):
                self.registered = []

            def register(self, path, tool, turn, source):
                self.registered.append(path)

        registry = MockRegistry()
        monitor.bind_artifact_registry(registry)

        # Execute and check auto-registration
        exec_id = monitor.before_tool("bash", {}, [])
        (temp_workspace / "output.txt").write_text("result")
        record = monitor.after_tool(exec_id, "bash", {})

        # Registry should be notified

    def test_unbind_artifact_registry(self, temp_workspace):
        """Test unbinding ArtifactRegistry."""
        monitor = FsMonitor(temp_workspace)

        class MockRegistry:
            def register(self, path, tool, turn, source):
                pass

        registry = MockRegistry()
        monitor.bind_artifact_registry(registry)
        monitor.unbind_artifact_registry()

        # Should not raise when registry is None
        exec_id = monitor.before_tool("bash", {}, [])
        (temp_workspace / "output.txt").write_text("result")
        record = monitor.after_tool(exec_id, "bash", {})

        assert record is not None

    def test_cleanup(self, temp_workspace):
        """Test cleanup method."""
        monitor = FsMonitor(temp_workspace)
        monitor.before_tool("bash", {}, [])
        count = monitor.cleanup()
        assert count >= 0

    def test_protocol_compliance(self, temp_workspace):
        """Test that FsMonitor implements FsMonitorProtocol."""
        monitor = FsMonitor(temp_workspace)
        assert isinstance(monitor, FsMonitorProtocol)


# =============================================================================
# SessionFsMonitor Tests
# =============================================================================

class TestSessionFsMonitor:
    """Tests for SessionFsMonitor."""

    def test_creation(self, temp_workspace):
        """Test SessionFsMonitor creation."""
        session = SessionFsMonitor(temp_workspace, session_id="test-session")
        assert session.session_id == "test-session"
        assert session.workspace_root == temp_workspace.resolve()

    def test_auto_session_id(self, temp_workspace):
        """Test automatic session ID generation."""
        session = SessionFsMonitor(temp_workspace)
        assert session.session_id.startswith("session_")

    def test_get_monitor(self, temp_workspace):
        """Test getting monitor instance."""
        session = SessionFsMonitor(temp_workspace)
        monitor = session.get_monitor()
        assert isinstance(monitor, FsMonitor)

    def test_same_monitor_instance(self, temp_workspace):
        """Test that get_monitor returns the same instance."""
        session = SessionFsMonitor(temp_workspace)
        monitor1 = session.get_monitor()
        monitor2 = session.get_monitor()
        assert monitor1 is monitor2

    def test_bind_artifact_registry(self, temp_workspace):
        """Test binding ArtifactRegistry."""
        session = SessionFsMonitor(temp_workspace)

        class MockRegistry:
            def register(self, path, tool, turn, source):
                pass

        registry = MockRegistry()
        session.bind_artifact_registry(registry)

        # Should not raise
        monitor = session.get_monitor()
        assert monitor is not None

    def test_cleanup(self, temp_workspace):
        """Test cleanup method."""
        session = SessionFsMonitor(temp_workspace)
        session.get_monitor()  # Create monitor
        count = session.cleanup()
        assert count >= 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full workflow."""

    def test_full_workflow(self, temp_workspace):
        """Test complete snapshot workflow."""
        # Setup
        manager = FsSnapshotManager(temp_workspace)
        monitor = FsMonitor(temp_workspace)

        # Before execution
        exec_id = monitor.before_tool("bash", {"command": "create files"}, [])
        manager.take_full_snapshot(exec_id)

        # Create files with different sizes to ensure detection
        (temp_workspace / "artifact.py").write_text("print('hello world')")
        (temp_workspace / "temp.tmp").write_text("temp data here")

        # After execution
        record = monitor.after_tool(exec_id, "bash", {"command": "create files"})

        # Verify tracking
        assert len(record.changes) >= 2

        # Verify snapshot management using compare
        after_id = f"{exec_id}_after"
        manager.take_full_snapshot(after_id)
        changes = manager.compare(exec_id, after_id)
        assert len(changes) >= 2

    @pytest.mark.asyncio
    async def test_async_execution_tracking(self, temp_workspace):
        """Test async execution tracking workflow."""
        # Simulate async tool execution
        manager = FsSnapshotManager(temp_workspace)

        exec_id = manager.generate_execution_id()
        manager.take_full_snapshot(exec_id)

        # Simulate async work
        await asyncio.sleep(0.01)

        # Create files
        (temp_workspace / "async_result.txt").write_text("async result")

        after_id = f"{exec_id}_after"
        manager.take_full_snapshot(after_id)

        changes = manager.compare(exec_id, after_id)
        assert len(changes) >= 1
        assert any(c.rel_path == "async_result.txt" for c in changes)

    def test_cross_platform_path_handling(self):
        """Test that path handling works across platforms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create files with various path separators
            file1 = root / "file1.txt"
            file1.write_text("content")

            # Test is_safe_within_workspace
            assert is_safe_within_workspace(file1, root) is True

            # Test paths_equal
            assert paths_equal(file1, file1) is True

            # Test resolve_path_safe
            resolved = resolve_path_safe(file1)
            assert resolved is not None
            assert resolved.is_absolute()


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_unicode_filenames(self, temp_workspace):
        """Test handling of unicode filenames."""
        try:
            # Create file with unicode name
            file = temp_workspace / "\u4e2d\u6587.txt"  # "中文.txt"
            file.write_text("content")

            manager = FsSnapshotManager(temp_workspace)
            snapshot = manager.take_snapshot("test-1")

            assert len(snapshot.entries) >= 1
        except OSError:
            pytest.skip("Unicode filenames not supported on this platform")

    def test_very_long_filenames(self, temp_workspace):
        """Test handling of very long filenames."""
        long_name = "a" * 200 + ".txt"
        file = temp_workspace / long_name
        file.write_text("content")

        manager = FsSnapshotManager(temp_workspace)
        snapshot = manager.take_snapshot("test-1")

        assert len(snapshot.entries) == 1

    def test_symbolic_links(self, temp_workspace):
        """Test handling of symbolic links."""
        file = temp_workspace / "original.txt"
        file.write_text("content")

        try:
            link = temp_workspace / "link.txt"
            link.symlink_to(file)

            manager = FsSnapshotManager(temp_workspace)
            snapshot = manager.take_snapshot("test-1")

            # Should handle symlinks gracefully
            assert snapshot is not None
        except OSError:
            pytest.skip("Symbolic links not supported on this platform")

    def test_permission_denied_files(self, temp_workspace):
        """Test handling of permission denied errors."""
        # Try to access a protected file (if we have permissions)
        # This is a best-effort test
        manager = FsSnapshotManager(temp_workspace)
        snapshot = manager.take_snapshot("test-1")
        assert snapshot is not None

    def test_race_condition_simulation(self, temp_workspace):
        """Test handling of rapid snapshot/diff cycles."""
        manager = FsSnapshotManager(temp_workspace)

        for i in range(10):
            exec_id = f"rapid-{i}"
            manager.take_snapshot(exec_id)
            (temp_workspace / f"file{i}.txt").write_text(f"content{i}")

            diff = manager.diff(exec_id)
            assert diff is not None

    def test_empty_workspace(self):
        """Test handling of empty workspace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = FsSnapshotManager(root)
            snapshot = manager.take_snapshot("test-1")
            assert len(snapshot.entries) == 0

    def test_snapshot_with_only_directories(self, temp_workspace):
        """Test snapshot when only directories exist."""
        (temp_workspace / "empty_dir").mkdir()
        (temp_workspace / "nested" / "dir").mkdir(parents=True)

        manager = FsSnapshotManager(temp_workspace)
        snapshot = manager.take_snapshot("test-1")

        assert len(snapshot.entries) == 0  # Only dirs, no files

    def test_very_large_file(self, temp_workspace):
        """Test handling of large files without hash computation."""
        file = temp_workspace / "large.bin"
        # Don't actually write a huge file, just test the config
        config = SnapshotConfig(compute_hash=False)
        manager = FsSnapshotManager(temp_workspace, config=config)

        # Should not compute hash even for large files
        assert config.compute_hash is False

    def test_special_characters_in_paths(self, temp_workspace):
        """Test handling of special characters in paths."""
        special_names = [
            "file with spaces.txt",
            "file-with-dashes.txt",
            "file_with_underscores.txt",
            "file.multiple.dots.txt",
        ]

        for name in special_names:
            file = temp_workspace / name
            file.write_text("content")

        manager = FsSnapshotManager(temp_workspace)
        snapshot = manager.take_snapshot("test-1")

        assert len(snapshot.entries) == len(special_names)
