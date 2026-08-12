# SPDX-License-Identifier: Apache-2.0
"""单元测试 — core/prompts/templates.py: 所有模板常量格式验证。"""

from __future__ import annotations

import re
import sys

import pytest

from core.prompts import templates as T

# 模板名称到必需变量的映射（基于实际模板分析）
REQUIRED_VARS: dict[str, set[str]] = {
    "INTENT_PROMPT": {"user_message", "history_summary", "session_context"},
    "PLAN_GENERATION_PROMPT": {
        "goal", "context", "skill_catalog", "current_datetime", "current_year"
    },
    "REFLECTION_PROMPT": {
        "plan", "current_step", "step_result", "remaining_steps", "execution_state"
    },
    "IDENTITY_SECTION": {"identity_opening", "current_time", "runtime"},
    "STEP_GOAL_HINT": {
        "step_id", "action", "expected_output", "skill_name", "skill_request", "input_summary"
    },
    "POST_COMPACTION_STATE": {"goal", "completed_steps", "current_step", "remaining_steps"},
}


class TestPromptTemplateVars:
    """验证模板包含所有必需的 {变量} 占位符。"""

    @pytest.mark.parametrize("name,expected", list(REQUIRED_VARS.items()))
    def test_template_has_required_vars(self, name: str, expected: set[str]) -> None:
        template = getattr(T, name, None)
        assert template is not None, f"Template {name!r} not found in core.prompts.templates"
        found = set(re.findall(r"\{(\w+)\}", template))
        missing = expected - found
        assert not missing, f"{name} missing vars: {missing}"

    def test_intent_output_choices(self) -> None:
        """验证 INTENT_PROMPT 包含所有 4 种 intent 输出选项。"""
        choices = re.findall(r'"(\w+)"', T.INTENT_PROMPT)
        assert {"direct", "agentic", "confirm", "interrupt"}.issubset(set(choices))

    def test_reflection_output_choices(self) -> None:
        """验证 REFLECTION_PROMPT 包含所有 5 种 decision 选项。"""
        choices = re.findall(r'"(\w+)"', T.REFLECTION_PROMPT)
        assert {"continue", "in_progress", "replan", "finalize", "ask_user"}.issubset(set(choices))

    def test_finalize_instruction_has_final_answer_section(self) -> None:
        """FINALIZE_INSTRUCTION 必须包含 Final Answer 规则说明。"""
        assert "Final Answer" in T.FINALIZE_INSTRUCTION or "final answer" in T.FINALIZE_INSTRUCTION.lower()

    def test_finalize_instruction_no_placeholders(self) -> None:
        """FINALIZE_INSTRUCTION 是纯文本，无占位符。"""
        assert "{" not in T.FINALIZE_INSTRUCTION
        assert "}" not in T.FINALIZE_INSTRUCTION

    def test_exec_failures_exceeded_has_placeholder(self) -> None:
        """EXEC_FAILURES_EXCEEDED_MSG 必须有 {last_error} 占位符。"""
        assert "{last_error}" in T.EXEC_FAILURES_EXCEEDED_MSG

    def test_max_iterations_msg_no_placeholders(self) -> None:
        """MAX_ITERATIONS_MSG 是纯文本，无占位符。"""
        assert "{" not in T.MAX_ITERATIONS_MSG
        assert "}" not in T.MAX_ITERATIONS_MSG

    def test_no_tool_no_final_answer_no_placeholders(self) -> None:
        """NO_TOOL_NO_FINAL_ANSWER_MSG 是纯文本，无占位符。"""
        assert "{" not in T.NO_TOOL_NO_FINAL_ANSWER_MSG
        assert "}" not in T.NO_TOOL_NO_FINAL_ANSWER_MSG

    def test_step_completed_msg_has_placeholder(self) -> None:
        """STEP_COMPLETED_MSG 必须有 {step_id} 和 {results} 占位符。"""
        assert "{step_id}" in T.STEP_COMPLETED_MSG
        assert "{results}" in T.STEP_COMPLETED_MSG

    def test_step_goal_hint_has_all_vars(self) -> None:
        """STEP_GOAL_HINT.format() 应能正确格式化。"""
        result = T.STEP_GOAL_HINT.format(
            skill_catalog="## Available Local Skills\n- pdf: PDF skill",
            step_id="1.2",
            action="Read configuration file",
            expected_output="config.yaml content",
            skill_name="code",
            skill_request="read config",
            input_summary="none",
        )
        assert "Step 1.2" in result
        assert "Read configuration file" in result
        assert "config.yaml content" in result
        assert "code" in result
        # format 不抛异常即通过

    def test_post_compaction_state_format(self) -> None:
        """POST_COMPACTION_STATE.format() 应能正确格式化。"""
        result = T.POST_COMPACTION_STATE.format(
            goal="Deploy app",
            completed_steps="Step 1: Build\nStep 2: Test",
            current_step="Step 3: Deploy",
            remaining_steps="Step 4: Verify",
        )
        assert "Deploy app" in result
        assert "Step 1" in result
        assert "Step 3" in result

    def test_plan_generation_prompt_has_goal_and_context(self) -> None:
        """PLAN_GENERATION_PROMPT 必须有 {goal} 和 {context}。"""
        assert "{goal}" in T.PLAN_GENERATION_PROMPT
        assert "{context}" in T.PLAN_GENERATION_PROMPT
        assert "{skill_catalog}" in T.PLAN_GENERATION_PROMPT

    def test_reflection_prompt_has_all_vars(self) -> None:
        """REFLECTION_PROMPT 必须有所有必需的变量。"""
        assert "{plan}" in T.REFLECTION_PROMPT
        assert "{current_step}" in T.REFLECTION_PROMPT
        assert "{step_result}" in T.REFLECTION_PROMPT
        assert "{remaining_steps}" in T.REFLECTION_PROMPT
        assert "{execution_state}" in T.REFLECTION_PROMPT

    def test_summarize_conversation_prompt_format(self) -> None:
        """SUMMARIZE_CONVERSATION_PROMPT 包含 {context} 和 {max_tokens}。"""
        assert "{context}" in T.SUMMARIZE_CONVERSATION_PROMPT
        assert "{max_tokens}" in T.SUMMARIZE_CONVERSATION_PROMPT
        assert "{plan_status}" in T.SUMMARIZE_CONVERSATION_PROMPT

    def test_error_policy_msg_placeholders(self) -> None:
        """ERROR_POLICY_MSG 包含 {action} 和 {reason}。"""
        assert "{action}" in T.ERROR_POLICY_MSG
        assert "{reason}" in T.ERROR_POLICY_MSG

    def test_skill_check_hint_msg_has_placeholder(self) -> None:
        """SKILL_CHECK_HINT_MSG 包含 {reason}。"""
        assert "{reason}" in T.SKILL_CHECK_HINT_MSG
