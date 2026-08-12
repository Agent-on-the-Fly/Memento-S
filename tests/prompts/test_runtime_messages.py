# SPDX-License-Identifier: Apache-2.0
"""单元测试 — core/prompts/templates.py: 运行时消息模板格式化。"""

from __future__ import annotations

import pytest

from core.prompts import templates as T


class TestRuntimeMessages:
    """验证运行时消息模板（STEP_GOAL_HINT、POST_COMPACTION_STATE 等）格式化正确。"""

    def test_step_goal_hint_format(self) -> None:
        result = T.STEP_GOAL_HINT.format(
            skill_catalog="## Available Local Skills\n- pdf: PDF skill",
            step_id="2.1",
            action="Read configuration file",
            expected_output="config.yaml content",
            skill_name="code",
            skill_request="read config",
            input_summary="none",
        )
        assert "Step 2.1" in result
        assert "Read configuration file" in result
        assert "config.yaml content" in result
        assert "code" in result

    def test_step_completed_msg_format(self) -> None:
        result = T.STEP_COMPLETED_MSG.format(step_id="1", results="Success")
        assert "Step 1" in result
        assert "Success" in result

    def test_post_compaction_state_format(self) -> None:
        result = T.POST_COMPACTION_STATE.format(
            goal="Deploy the app",
            completed_steps="Step 1: Build\nStep 2: Test",
            current_step="Step 3: Deploy",
            remaining_steps="Step 4: Verify",
        )
        assert "Deploy the app" in result
        assert "Step 1" in result
        assert "Step 3" in result

    def test_finalize_instruction_has_final_answer(self) -> None:
        assert (
            "Final Answer" in T.FINALIZE_INSTRUCTION
            or "final answer" in T.FINALIZE_INSTRUCTION
        )

    def test_finalize_instruction_no_placeholders(self) -> None:
        assert "{" not in T.FINALIZE_INSTRUCTION
        assert "}" not in T.FINALIZE_INSTRUCTION

    def test_exec_failures_exceeded_format(self) -> None:
        result = T.EXEC_FAILURES_EXCEEDED_MSG.format(last_error="timeout")
        assert "timeout" in result

    def test_max_iterations_msg_no_placeholders(self) -> None:
        assert "{" not in T.MAX_ITERATIONS_MSG
        assert "}" not in T.MAX_ITERATIONS_MSG

    def test_no_tool_no_final_answer(self) -> None:
        assert len(T.NO_TOOL_NO_FINAL_ANSWER_MSG) > 0
        assert "{" not in T.NO_TOOL_NO_FINAL_ANSWER_MSG

    def test_error_policy_msg_format(self) -> None:
        result = T.ERROR_POLICY_MSG.format(action="read_file", reason="file not found")
        assert "read_file" in result
        assert "file not found" in result

    def test_skill_check_hint_msg_format(self) -> None:
        result = T.SKILL_CHECK_HINT_MSG.format(reason="no data returned")
        assert "no data returned" in result

    def test_step_reflection_hint_format(self) -> None:
        result = T.STEP_REFLECTION_HINT.format(reason="step completed successfully")
        assert "step completed successfully" in result
