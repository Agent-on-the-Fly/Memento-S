# SPDX-License-Identifier: Apache-2.0
"""Tests for LongTermMemory — indexed topic files.

Verifies: accumulate_session, write_topic, list_topics, apply_consolidation_result.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_memory_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def ltm(tmp_memory_dir):
    from infra.memory.impl.long_term_memory import LongTermMemory

    return LongTermMemory(memory_dir=tmp_memory_dir, model="")


class TestLongTermMemory:
    """LongTermMemory 测试。"""

    @pytest.mark.asyncio
    async def test_accumulate_session(self, ltm, tmp_memory_dir):
        await ltm.accumulate_session("# Test session\nContent here.")
        staging = ltm.get_staging_content()
        assert "Test session" in staging
        assert "Content here" in staging

    @pytest.mark.asyncio
    async def test_write_topic(self, ltm, tmp_memory_dir):
        await ltm.write_topic(
            slug="test-topic",
            title="Test Topic",
            content="This is the test topic content.",
            hook="Hook line for the topic",
        )
        content = ltm.read_topic("test-topic")
        assert "Test Topic" in content
        assert "test topic content" in content

        # write_topic 不再直接更新索引，索引由 apply_consolidation_result 统一管理

    def test_list_topics(self, ltm, tmp_memory_dir):
        import asyncio
        asyncio.run(ltm.write_topic("topic-a", "Topic A", "Content A", "Hook A"))
        asyncio.run(ltm.write_topic("topic-b", "Topic B", "Content B", "Hook B"))

        topics = ltm.list_topics()
        slugs = {t["slug"] for t in topics}
        assert "topic-a" in slugs
        assert "topic-b" in slugs

    def test_list_topics_with_content(self, ltm, tmp_memory_dir):
        import asyncio
        asyncio.run(ltm.write_topic("topic-x", "Topic X", "Content X", "Hook X"))

        topics = ltm.list_topics_with_content()
        assert any(t["slug"] == "topic-x" and "Content X" in t["content"] for t in topics)

    def test_apply_consolidation_result(self, ltm, tmp_memory_dir):
        import asyncio
        asyncio.run(ltm.write_topic("old-topic", "Old Topic", "Old content", "Old hook"))

        result = {
            "updated_topics": [
                {"slug": "old-topic", "title": "Updated Topic", "content": "New content", "hook": "New hook"}
            ],
            "new_topics": [
                {"slug": "new-topic", "title": "New Topic", "content": "Brand new", "hook": "New hook"}
            ],
            "deleted_topics": [],
            "index_content": "# Project Memory Index\n- [Updated Topic](old-topic.md) — New hook\n- [New Topic](new-topic.md) — New hook\n",
        }
        ltm.apply_consolidation_result(result)

        assert "New content" in ltm.read_topic("old-topic")
        assert "Brand new" in ltm.read_topic("new-topic")

    def test_apply_consolidation_delete_topic(self, ltm, tmp_memory_dir):
        import asyncio
        asyncio.run(ltm.write_topic("to-delete", "To Delete", "Will be removed", "Gone"))

        result = {
            "deleted_topics": ["to-delete"],
            "updated_topics": [],
            "new_topics": [],
            "index_content": "# Project Memory Index\n",
        }
        ltm.apply_consolidation_result(result)
        assert ltm.read_topic("to-delete") == ""

    def test_load_memory_prompt_truncates(self, ltm, tmp_memory_dir):
        import asyncio
        # 写入 topic 文件（不更新 MEMORY.md）
        asyncio.run(ltm.write_topic("large", "Large Topic", "large content", "Large"))
        # 通过 apply_consolidation_result 设置 MEMORY.md（与实际 consolidation 行为一致）
        ltm.apply_consolidation_result({
            "updated_topics": [],
            "new_topics": [],
            "deleted_topics": [],
            "index_content": "# Memory Index\n- [Large Topic](large.md) — Large\n",
        })

        section = ltm.load_memory_prompt()
        assert "## Long-term Memory Index" in section
        assert "large" in section

    def test_clear_staging(self, ltm, tmp_memory_dir):
        import asyncio
        asyncio.run(ltm.accumulate_session("Test content"))
        assert ltm.get_staging_content()

        ltm.clear_staging()
        assert not ltm.get_staging_content().strip()
