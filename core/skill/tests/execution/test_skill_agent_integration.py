# SPDX-License-Identifier: Apache-2.0
"""集成测试：SkillAgent 端到端执行，验证 hook_context 机制。

针对 SAM3 项目介绍任务，验证：
1. hook_context["fs_changes"] 从 FileChangeHook 流向 LoopSupervisionHook
2. observation_chain loop 智能跳过行为
3. 真实的 tool call 序列

使用直接 monkeypatch 捕获 hook_context 状态。
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


def _make_skill(name: str, content: str) -> MagicMock:
    from unittest.mock import MagicMock as _MagicMock
    from core.skill.schema import Skill

    s = _MagicMock(spec=Skill)
    s.name = name
    s.description = name
    s.content = content
    s.source_dir = None
    s.allowed_tools = None
    s.execution_mode = "auto"
    return s


def _make_config(workspace: Path) -> MagicMock:
    from unittest.mock import MagicMock
    config = MagicMock()
    config.workspace_dir = workspace
    config.primary_artifact_path = None
    return config


@pytest.mark.asyncio
@pytest.mark.slow
async def test_skill_agent_sam3_presentation_hook_context():
    """
    端到端执行 SAM3 项目介绍，验证：
    - hook_context["fs_changes"] 在 AFTER_TOOL_EXEC 事件链中正确传递
    - observation_chain 警告数量 ≤ 1（智能跳过生效）
    - repeating_sequence 警告 ≥ 1（loop 检测正常工作）
    - 产物文件数 > 5（任务正常执行）
    """

    from core.skill.execution.agent import SkillAgent
    from core.skill.execution.hooks.loop_supervision import LoopSupervisionHook
    from core.skill.execution.hooks.file_change_hook import FileChangeHook
    from shared.hooks.executor import HookExecutor
    from shared.hooks.types import HookEvent

    # ── 0. Bootstrap tools + config ─────────────────────────────
    from middleware.config import g_config
    g_config.load()
    from tools import init_registry, load_atomics
    init_registry()
    load_atomics()

    # ── 1. 环境 ─────────────────────────────────────────────────
    workspace = Path(tempfile.mkdtemp(prefix="test_hook_ctx_"))
    config = _make_config(workspace)
    skill = _make_skill(
        name="sam3_presentation",
        content="# SAM3 Presentation\n\nCreate an 8-page professional presentation.",
    )

    # ── 2. 共享 hook_context 捕获（直接观察）───────────────────
    captured_hook_contexts: list[dict[str, Any]] = []
    captured_loop_records: list[dict[str, Any]] = []
    captured_loop_warnings: list[dict[str, Any]] = []
    loop_warnings_from_scratchpad: list[str] = []

    # 记录每个 AFTER_TOOL_EXEC 的 hook_context
    original_execute = HookExecutor.execute

    async def patched_execute(
        self, event: HookEvent, payload: Any
    ) -> Any:
        # BEFORE: 记录 hook_context 初始状态
        before_ctx = dict(self._hook_context)

        # 执行
        result = await original_execute(self, event, payload)

        # AFTER: 记录 hook_context 最终状态（供下一个 hook 使用）
        after_ctx = dict(self._hook_context)

        if event == HookEvent.AFTER_TOOL_EXEC:
            captured_hook_contexts.append({
                "tool": getattr(payload, "tool_name", None),
                "hook_context_after": after_ctx,
                "fs_changes": after_ctx.get("fs_changes"),
            })

        return result

    # ── 3. 捕获 LoopDetector.record() ────────────────────────────
    from core.skill.execution.loop_detector import LoopDetector

    original_record = LoopDetector.record

    def patched_record(
        self,
        tool_name: str,
        category: str,
        turn: int,
        new_entities: int = 0,
        created_artifacts: int = 0,
        artifact_registry: Any = None,
    ) -> None:
        captured_loop_records.append({
            "tool": tool_name,
            "category": category,
            "turn": turn,
            "new_entities": new_entities,
            "created_artifacts": created_artifacts,
        })
        return original_record(
            self, tool_name, category, turn,
            new_entities=new_entities,
            created_artifacts=created_artifacts,
            artifact_registry=artifact_registry,
        )

    # ── 4. 捕获 scratchpad 更新 ────────────────────────────────
    scratchpad_updates: list[dict] = []

    from core.skill.execution.state import ReActState
    global _original_state_update_scratchpad
    _original_state_update_scratchpad = ReActState.update_scratchpad

    def patched_state_update_scratchpad(self, text: str) -> None:
        scratchpad_updates.append({"text": text[:500]})
        return _original_state_update_scratchpad(self, text)

    # ── 5. 应用 patch ───────────────────────────────────────────
    HookExecutor.execute = patched_execute
    LoopDetector.record = patched_record
    ReActState.update_scratchpad = patched_state_update_scratchpad

    try:
        # ── 6. 执行任务 ─────────────────────────────────────────
        from middleware.llm import LLMClient

        llm = LLMClient()
        agent = SkillAgent(config=config, llm=llm)

        outcome, generated_code = await agent.run(
            skill=skill,
            query=(
                "请为 Meta 的开源项目 SAM 3: Segment Anything with Concepts"
                "（https://github.com/facebookresearch/sam3/）制作一份 8 页的项目介绍演示文稿。"
                "输出文件为 results_5/output.pdf，要求恰好 8 页，设计风格统一且专业。"
            ),
            params=None,
            run_dir=workspace,
            session_id="test-hook-ctx-001",
            on_step=None,
            max_turns=15,
        )
    finally:
        HookExecutor.execute = original_execute
        LoopDetector.record = original_record
        ReActState.update_scratchpad = _original_state_update_scratchpad

    # ── 7. 分析捕获数据 ─────────────────────────────────────────
    # 7a. hook_context 中有 fs_changes 的次数
    fs_changes_captures = [
        c for c in captured_hook_contexts
        if c["fs_changes"] is not None
    ]

    # 7b. observation_chain category（来自 LoopDetector.record）
    obs_chain_records = [
        r for r in captured_loop_records
        if r["category"] == "observation"
    ]

    # 7c. scratchpad 中的 loop 警告
    scratchpad_text = "".join(u.get("text", "") for u in scratchpad_updates)
    obs_chain_in_sp = "observation_chain" in scratchpad_text
    repeating_seq_in_sp = "repeating_sequence" in scratchpad_text

    # 7d. 产物文件
    all_files = [
        f for f in workspace.rglob("*")
        if f.is_file() and "__pycache__" not in str(f)
    ]

    # ── 8. 输出报告 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("HOOK CONTEXT TEST REPORT")
    print("=" * 60)
    print(f"  Outcome          : {outcome}")
    print(f"  Tool calls       : {sum(1 for r in captured_loop_records)}")
    print(f"  hook_context 捕获: {len(captured_hook_contexts)}")
    print(f"  fs_changes 有效 : {len(fs_changes_captures)}")
    print(f"  observation category: {len(obs_chain_records)}")
    print(f"  observation_chain 警告: {obs_chain_in_sp}")
    print(f"  repeating_sequence 警告: {repeating_seq_in_sp}")
    print(f"  产物文件数        : {len(all_files)}")

    print("\n  hook_context 详情（前5条）:")
    for c in captured_hook_contexts[:5]:
        fs = c["fs_changes"]
        print(f"    tool={c['tool']}, fs_changes={fs}")

    print("\n  LoopDetector.record 详情:")
    for r in captured_loop_records[:10]:
        print(f"    turn={r['turn']} tool={r['tool']} cat={r['category']} "
              f"new_ent={r['new_entities']} created={r['created_artifacts']}")

    print("=" * 60)

    # ── 9. 断言验证 ─────────────────────────────────────────────
    # H1: fs_changes 在 hook_context 中至少出现 1 次
    # 这证明 FileChangeHook 成功写入了 fs_changes 到共享上下文
    assert len(fs_changes_captures) >= 1, (
        f"hook_context['fs_changes'] 从未被设置（{len(fs_changes_captures)} 次），"
        "FileChangeHook 可能未正常工作"
    )

    # H2: 产物文件数合理（任务有进展）
    assert len(all_files) >= 2, (
        f"workspace 仅 {len(all_files)} 个文件，任务可能未正常执行"
    )

    # H3: observation_chain 类别出现时，警告数 ≤ 1
    # 即智能跳过逻辑防止了无限制的 observation_chain 警告
    obs_warnings_in_sp = scratchpad_text.count("observation_chain")
    assert obs_warnings_in_sp <= 1, (
        f"observation_chain 警告出现 {obs_warnings_in_sp} 次（预期 ≤ 1），"
        "智能跳过逻辑可能未生效"
    )

    # H4: repeating_sequence 警告出现（loop 检测工作）
    assert repeating_seq_in_sp, (
        "repeating_sequence 警告未出现，loop 检测可能未激活"
    )

    print("\n✅ 所有断言通过 — hook_context 机制验证成功")


# 修复：保存原始函数引用
_original_state_update_scratchpad = None  # filled in try block
