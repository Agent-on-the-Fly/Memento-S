# SPDX-License-Identifier: Apache-2.0
"""集成测试 — infra/compact/storage.py: ArtifactStore 落盘 + preview 替换。"""

from __future__ import annotations

import json
import pytest
from unittest.mock import patch
from pathlib import Path

from infra.compact.storage import ArtifactStore, FileStorageBackend


class TestArtifactStore:
    """验证 ArtifactStore 的落盘、preview 替换、读取逻辑。"""

    @pytest.mark.asyncio
    async def test_small_result_not_persisted(self, tmp_path: Path) -> None:
        """小结果（< threshold）不落盘。"""
        store = ArtifactStore(session_dir=tmp_path)
        result = await store.process_tool_result(
            "s1", "read", "hi", remaining_budget_tokens=5000,
        )
        assert result["_persisted"] is False

    @pytest.mark.asyncio
    async def test_large_skill_result_persisted(self, tmp_path: Path) -> None:
        """大 skill 结果落盘，内容被 preview 替换。"""
        store = ArtifactStore(session_dir=tmp_path)
        large = "x" * 20000
        result = await store.process_tool_result(
            "s2", "execute_skill", large, remaining_budget_tokens=100,
        )
        assert result["_persisted"] is True
        assert result["content"] != large
        assert "[Output persisted" in result["content"]
        assert (tmp_path / "artifacts" / "s2.txt").exists()

    @pytest.mark.asyncio
    async def test_backend_has_after_persist(self, tmp_path: Path) -> None:
        """FileStorageBackend.has() 在 persist 后返回 True。"""
        backend = FileStorageBackend(session_dir=tmp_path)
        # 直接 persist 一个文件
        await backend.persist("hello world", {"tool_call_id": "test1"})
        assert await backend.has("test1") is True
        assert (await backend.has("nonexistent")) is False

    @pytest.mark.asyncio
    async def test_backend_recall_returns_content(self, tmp_path: Path) -> None:
        """FileStorageBackend.recall() 返回完整内容。"""
        backend = FileStorageBackend(session_dir=tmp_path)
        original = "important " * 200
        await backend.persist(original, {"tool_call_id": "test2"})
        recalled = await backend.recall("test2")
        assert recalled == original

    @pytest.mark.asyncio
    async def test_backend_recall_nonexistent(self, tmp_path: Path) -> None:
        """FileStorageBackend.recall() 对不存在文件返回 None。"""
        backend = FileStorageBackend(session_dir=tmp_path)
        result = await backend.recall("does-not-exist")
        assert result is None

    @pytest.mark.asyncio
    async def test_read_nonexistent_returns_none(self, tmp_path: Path) -> None:
        """ArtifactStore.read_artifact() 对不存在文件返回 None。"""
        store = ArtifactStore(session_dir=tmp_path)
        result = await store.read_artifact("does-not-exist")
        assert result is None

    @pytest.mark.asyncio
    async def test_persisted_file_content(self, tmp_path: Path) -> None:
        """验证落盘文件内容与原始一致。"""
        store = ArtifactStore(session_dir=tmp_path)
        content = "test content " * 100
        await store.process_tool_result(
            "s5", "write", content, remaining_budget_tokens=50,
        )
        artifact_path = tmp_path / "artifacts" / "s5.txt"
        assert artifact_path.exists()
        file_content = artifact_path.read_text(encoding="utf-8")
        assert file_content == content


class TestFileStorageBackend:
    """直接测试 FileStorageBackend。"""

    @pytest.mark.asyncio
    async def test_persist_and_has(self, tmp_path: Path) -> None:
        """persist 后 has 应返回 True。"""
        backend = FileStorageBackend(session_dir=tmp_path)
        await backend.persist("data", {"tool_call_id": "fid1"})
        assert await backend.has("fid1") is True

    @pytest.mark.asyncio
    async def test_persist_without_tool_call_id_raises(self, tmp_path: Path) -> None:
        """persist 时没有 tool_call_id 应抛出 ValueError。"""
        backend = FileStorageBackend(session_dir=tmp_path)
        with pytest.raises(ValueError, match="tool_call_id"):
            await backend.persist("data", {})

    @pytest.mark.asyncio
    async def test_recall_after_persist(self, tmp_path: Path) -> None:
        """persist 后 recall 应返回原始内容。"""
        backend = FileStorageBackend(session_dir=tmp_path)
        content = "hello world content"
        await backend.persist(content, {"tool_call_id": "fid2"})
        recalled = await backend.recall("fid2")
        assert recalled == content

    @pytest.mark.asyncio
    async def test_has_not_exists(self, tmp_path: Path) -> None:
        """不存在的文件 has 应返回 False。"""
        backend = FileStorageBackend(session_dir=tmp_path)
        assert (await backend.has("xyz")) is False
