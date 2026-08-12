# SPDX-License-Identifier: Apache-2.0
"""单元测试 — core/prompts/prompt_builder.py: PromptBuilder 优先级/拼接逻辑。"""

from __future__ import annotations

import pytest

from core.prompts.prompt_builder import PromptBuilder


class TestPromptBuilder:
    """验证 PromptBuilder 按 priority 升序拼接、空内容跳过、链式调用。"""

    def test_priority_order(self) -> None:
        """priority 越小越靠前。"""
        pb = PromptBuilder()
        pb.add("low", priority=100)
        pb.add("high", priority=10)
        pb.add("mid", priority=50)
        result = pb.build()
        assert result.index("high") < result.index("mid") < result.index("low")

    def test_empty_skipped(self) -> None:
        """空字符串和纯空白内容不添加。"""
        pb = PromptBuilder()
        pb.add("", priority=10)
        pb.add("   ", priority=20)
        pb.add("actual", priority=30)
        assert pb.section_count == 1

    def test_chain_api(self) -> None:
        """add() 返回 self，支持链式调用。"""
        result = (
            PromptBuilder()
            .add("a", priority=10)
            .add("b", priority=20)
            .build()
        )
        assert "a" in result
        assert "b" in result

    def test_labels_tracking(self) -> None:
        """labels() 返回注册过的 label 列表。"""
        pb = PromptBuilder()
        pb.add("content", priority=10, label="my_section")
        pb.add("other", priority=20, label="other_section")
        assert "my_section" in pb.labels()
        assert "other_section" in pb.labels()

    def test_labels_ordered_by_priority(self) -> None:
        """labels() 按 priority 升序返回。"""
        pb = PromptBuilder()
        pb.add("c", priority=30, label="third")
        pb.add("a", priority=10, label="first")
        pb.add("b", priority=20, label="second")
        assert pb.labels() == ["first", "second", "third"]

    def test_separator_between_sections(self) -> None:
        """build() 在各 section 之间插入 separator。"""
        pb = PromptBuilder()
        pb.add("sec1", priority=10)
        pb.add("sec2", priority=20)
        result = pb.build()
        assert pb._separator in result

    def test_custom_separator(self) -> None:
        """支持自定义 separator。"""
        pb = PromptBuilder(separator="\n---\n")
        pb.add("a", priority=10)
        pb.add("b", priority=20)
        result = pb.build()
        assert "\n---\n" in result

    def test_section_count(self) -> None:
        """section_count 正确计数。"""
        pb = PromptBuilder()
        assert pb.section_count == 0
        pb.add("a", priority=10)
        assert pb.section_count == 1
        pb.add("b", priority=20)
        assert pb.section_count == 2
        pb.add("", priority=30)
        assert pb.section_count == 2  # 空内容不计入

    def test_build_empty(self) -> None:
        """没有任何 section 时 build() 返回空字符串。"""
        pb = PromptBuilder()
        assert pb.build() == ""

    def test_labels_empty_when_no_sections(self) -> None:
        """无 section 时 labels() 返回空列表。"""
        pb = PromptBuilder()
        assert pb.labels() == []
