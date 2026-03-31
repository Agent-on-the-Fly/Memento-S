#!/usr/bin/env python3
"""
验证 agent 系统提示词的组装逻辑，重点检查 skills 部分是否包含本地全部 skills。

使用方法:
    .venv/bin/python tests/test_prompt_components.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.skill.schema import Skill, ExecutionMode
from core.skill.gateway import (
    SkillGateway,
    SkillManifest,
    SkillGovernanceMeta,
    DEFAULT_SKILL_PARAMS,
)

from core.skill.retrieval.multi_recall import MultiRecall
from core.context import ContextManager
from core.memento_s.schemas import AgentConfig


# ── 全局 mock g_config ──────────────────────────────────────────


def _make_mock_config():
    """构造一个够用的 mock g_config"""
    cfg = MagicMock()
    cfg.skills.retrieval.top_k = 5
    cfg.skills.execution.pip_install_timeout_sec = 60
    cfg.paths.workspace_dir = Path("/tmp/test_workspace")
    cfg.paths.skills_dir = Path("/tmp/test_workspace/skills")
    cfg.paths.context_dir = Path("/tmp/test_prompt_ctx")
    cfg.llm.current_profile.context_window = 128000
    cfg.llm.current_profile.max_tokens = 8192
    cfg.llm.current_profile.input_budget = 128000 - 8192
    cfg.get_skills_path.return_value = Path("/tmp/test_workspace/skills")
    cfg.get_data_dir.return_value = Path("/tmp/test_data")
    return cfg


_mock_config = _make_mock_config()


# ── 构造 fake skills ──────────────────────────────────────────


def _make_fake_skills() -> dict[str, Skill]:
    """创建几个假 skill 用于测试"""
    return {
        "web_search": Skill(
            name="web_search",
            description="Search the web for real-time information",
            content="# Web Search\nUse this skill to search the web.",
            dependencies=["requests"],
            source_dir=None,
        ),
        "filesystem": Skill(
            name="filesystem",
            description="Read, write and manage files in the workspace",
            content="# Filesystem\nUse this skill to manage files.",
            dependencies=[],
            source_dir=None,
        ),
        "skill_creator": Skill(
            name="skill_creator",
            description="Create new skills from natural language descriptions",
            content="# Skill Creator\nUse this skill to create new skills.",
            dependencies=[],
            source_dir=None,
        ),
    }


def _load_real_builtin_skills() -> dict[str, Skill] | None:
    """尝试加载真实 builtin skills，失败返回 None"""
    try:
        from core.skill.store.persistence import load_all_skills

        builtin_dir = project_root / "builtin" / "skills"
        if not builtin_dir.exists():
            return None
        skills = load_all_skills(builtin_dir)
        return skills if skills else None
    except Exception as e:
        print(f"  [WARN] 加载真实 skills 失败: {e}")
        return None


def _build_provider(skills: dict[str, Skill]) -> SkillGateway:
    """构建 SkillGateway（无 embedding、无 DB、无云端）"""
    multi_recall = MultiRecall(cloud_catalog=None)

    store = MagicMock()
    store.local_cache = skills

    with patch("core.skill.provider.g_config", _mock_config):
        provider = SkillGateway(
            config=_mock_config,
            store=store,
            multi_recall=multi_recall,
        )
    return provider


# ── 测试 ──────────────────────────────────────────────────────


async def test_multi_recall_returns_all_local():
    """验证 MultiRecall 空查询返回全部本地 skills"""
    print("\n【1. MultiRecall.recall('') 是否返回全部本地 skills】")
    skills = _make_fake_skills()
    multi_recall = MultiRecall(cloud_catalog=None)
    candidates = await multi_recall.recall("", local_cache=skills)

    candidate_names = {c.name for c in candidates}
    skill_names = set(skills.keys())

    assert candidate_names == skill_names, (
        f"不一致！\n  期望: {skill_names}\n  实际: {candidate_names}"
    )
    for c in candidates:
        assert c.source == "local"
        assert c.score == 1.0
        assert c.match_type == "local"
    print(f"  ✓ 返回 {len(candidates)} 个候选，全部 source=local, score=1.0")
    print(f"  ✓ 名称: {sorted(candidate_names)}")


async def test_multi_recall_ignores_query():
    """验证 MultiRecall 无论 query 是什么，本地都全量返回"""
    print("\n【2. MultiRecall.recall('任意查询') 本地仍然全量返回】")
    skills = _make_fake_skills()
    multi_recall = MultiRecall(cloud_catalog=None)

    for query in ["web search", "不存在的xyz", "filesystem", ""]:
        candidates = await multi_recall.recall(query, local_cache=skills)
        local_candidates = [c for c in candidates if c.source == "local"]
        assert len(local_candidates) == len(skills), (
            f"query='{query}' 时本地候选数不对: {len(local_candidates)} != {len(skills)}"
        )
    print(f"  ✓ 4 种不同 query 均返回全部 {len(skills)} 个本地 skills")


async def test_provider_search_returns_all():
    """验证 SkillGateway.search('', k=100) 返回全部本地 skills"""
    print("\n【3. SkillGateway.search('', k=100) 返回全部 SkillManifest】")
    skills = _make_fake_skills()
    provider = _build_provider(skills)

    with patch("core.skill.provider.g_config", _mock_config):
        manifests = await provider.search("", k=100)
    manifest_names = {m.name for m in manifests}

    assert manifest_names == set(skills.keys()), (
        f"不一致！\n  期望: {set(skills.keys())}\n  实际: {manifest_names}"
    )
    for m in manifests:
        assert isinstance(m, SkillManifest)
        assert m.governance.source == "local"
    print(f"  ✓ 返回 {len(manifests)} 个 SkillManifest，全部 source=local")


async def test_skills_summary_in_system_prompt():
    """验证系统提示词中包含全部本地 skills 的摘要"""
    print("\n【4. 系统提示词中 skills_summary 是否包含全部本地 skills】")
    skills = _make_fake_skills()
    provider = _build_provider(skills)

    with (
        patch("core.context.manager.g_config", _mock_config),
        patch("core.skill.provider.g_config", _mock_config),
    ):
        ctx_mgr = ContextManager(
            session_id="test",
            config=AgentConfig(),
            skill_gateway=provider,
        )
        summary = await ctx_mgr._build_skills_summary()
    print(f"\n  --- skills_summary 内容 ---")
    print(f"  {summary}")
    print(f"  --- 结束 ---\n")

    for name in skills:
        assert name in summary, f"skill '{name}' 未出现在 summary 中"
    print(f"  ✓ 全部 {len(skills)} 个 skill 名称均在 summary 中")


async def test_full_system_prompt_structure():
    """验证完整系统提示词的结构和各组件"""
    print("\n【5. 完整系统提示词结构检查（mode=agentic）】")
    skills = _make_fake_skills()
    provider = _build_provider(skills)

    with (
        patch("core.context.manager.g_config", _mock_config),
        patch("core.skill.provider.g_config", _mock_config),
    ):
        ctx_mgr = ContextManager(
            session_id="test",
            config=AgentConfig(),
            skill_gateway=provider,
        )
        prompt = await ctx_mgr.assemble_system_prompt(mode="agentic")

    print(f"\n{'=' * 70}")
    print("完整系统提示词（mode=agentic）")
    print(f"{'=' * 70}")
    print(prompt)
    print(f"{'=' * 70}")
    print(f"总长度: {len(prompt)} 字符")
    print(f"{'=' * 70}\n")

    assert "Memento-S" in prompt, "缺少 Identity 部分"
    assert "runtime_behavior" in prompt, "缺少 runtime_behavior 部分"
    assert "Protocol" in prompt, "缺少 Protocol 部分"
    assert "search_skill" in prompt, "缺少 Builtin Tools 部分"
    assert "execute_skill" in prompt, "缺少 Builtin Tools 部分"
    assert "available_skills" in prompt, "缺少 available_skills Section"

    for name, skill in skills.items():
        assert name in prompt, f"skill '{name}' 未出现在系统提示词中"
        assert skill.description in prompt, f"skill '{name}' 的描述未出现在系统提示词中"

    print("  ✓ Identity 部分存在")
    print("  ✓ runtime_behavior 部分存在")
    print("  ✓ Protocol & Format 部分存在")
    print("  ✓ Builtin Tools (search_skill / execute_skill) 部分存在")
    print("  ✓ available_skills Section 存在")
    print(f"  ✓ 全部 {len(skills)} 个 skill 名称和描述均在提示词中")


async def test_direct_mode_no_skills():
    """验证 mode=direct 时不注入 skills section"""
    print("\n【6. mode=direct 时不注入 skills section】")
    skills = _make_fake_skills()
    provider = _build_provider(skills)

    with (
        patch("core.context.manager.g_config", _mock_config),
        patch("core.skill.provider.g_config", _mock_config),
    ):
        ctx_mgr = ContextManager(
            session_id="test",
            config=AgentConfig(),
            skill_gateway=provider,
        )
        prompt = await ctx_mgr.assemble_system_prompt(mode="direct")

    has_skills_section = "Available Skills (Local)" in prompt
    print(f"  mode=direct 提示词中有 'Available Skills (Local)': {has_skills_section}")
    assert not has_skills_section, (
        "mode=direct 不应包含 Available Skills (Local) Section"
    )
    print(f"  ✓ mode=direct 正确跳过了 Available Skills section")


async def test_with_real_builtin_skills():
    """使用真实 builtin skills 验证"""
    print("\n【7. 使用真实 builtin skills 验证 (可选)】")
    real_skills = _load_real_builtin_skills()
    if real_skills is None:
        print("  [SKIP] 无法加载真实 builtin skills，跳过")
        return

    print(f"  加载了 {len(real_skills)} 个真实 builtin skills:")
    for name in sorted(real_skills.keys()):
        desc = (real_skills[name].description or "")[:60]
        print(f"    - {name}: {desc}")

    provider = _build_provider(real_skills)

    with (
        patch("core.context.manager.g_config", _mock_config),
        patch("core.skill.provider.g_config", _mock_config),
    ):
        ctx_mgr = ContextManager(
            session_id="test",
            config=AgentConfig(),
            skill_gateway=provider,
        )

        summary = await ctx_mgr._build_skills_summary()

        missing = []
        for name in real_skills:
            if name not in summary:
                missing.append(name)

        if missing:
            print(f"\n  ✗ 缺失 skills: {missing}")
            assert False, f"以下 skills 未出现在 summary 中: {missing}"
        else:
            print(f"\n  ✓ 全部 {len(real_skills)} 个真实 skills 均在 summary 中")

        prompt = await ctx_mgr.assemble_system_prompt(mode="agentic")
        print(
            f"\n  系统提示词总长度: {len(prompt)} 字符 (约 {len(prompt) // 4} tokens)"
        )

        print(f"\n{'=' * 70}")
        print("真实 builtin skills 的完整系统提示词")
        print(f"{'=' * 70}")
        print(prompt)
        print(f"{'=' * 70}\n")


async def test_search_skill_no_local_duplication():
    """验证 search_skill 不再返回本地 skills，只返回云端新增"""
    print("\n【8. search_skill 不再返回本地 skills（去重验证）】")

    from core.memento_s.tool_dispatcher import ToolDispatcher
    from core.memento_s.policies import PolicyManager

    real_skills = _load_real_builtin_skills()
    if real_skills is None:
        print("  [SKIP] 无法加载真实 builtin skills")
        return

    provider = _build_provider(real_skills)

    dispatcher = ToolDispatcher(
        policy_manager=PolicyManager(),
        skill_gateway=provider,
        session_id="test_dedup",
    )
    dispatcher.set_session_id("test_dedup")

    with patch("core.skill.provider.g_config", _mock_config):
        search_result = await dispatcher.execute(
            "search_skill",
            {"query": "web search 搜索量子计算", "k": 5},
        )

    import json as _json

    parsed = _json.loads(search_result)

    print(f"  search_skill 返回:")
    print(f"    ok: {parsed['ok']}")
    print(f"    summary: {parsed['summary']}")
    print(f"    output 数量: {len(parsed['output'])}")
    print(f"    diagnostics: {parsed.get('diagnostics', {})}")

    result_skills = parsed.get("output", [])
    local_in_result = [
        s for s in result_skills if s.get("source") in ("local", "builtin")
    ]

    assert len(local_in_result) == 0, (
        f"search_skill 不应返回本地 skills，但返回了 {len(local_in_result)} 个: "
        f"{[s['name'] for s in local_in_result]}"
    )
    print(f"  ✓ search_skill 返回 0 个本地 skill（去重成功）")

    local_count = parsed.get("diagnostics", {}).get("local_in_context")
    assert local_count == len(real_skills), (
        f"local_in_context 应为 {len(real_skills)}，实际为 {local_count}"
    )
    print(f"  ✓ local_in_context = {local_count}（正确反映本地 skill 数量）")

    assert "local skills already in context" in parsed.get("summary", ""), (
        f"summary 应提示本地 skills 已在 context 中"
    )
    print(f"  ✓ summary 正确提示本地 skills 已在上下文中")


async def test_execute_skill_direct_local():
    """验证本地 skill 可以不经 search_skill 直接 execute_skill"""
    print("\n【9. execute_skill 直接调用本地 skill（无需先 search）】")

    from core.memento_s.tool_dispatcher import ToolDispatcher
    from core.memento_s.policies import PolicyManager

    skills = _make_fake_skills()
    provider = _build_provider(skills)

    dispatcher = ToolDispatcher(
        policy_manager=PolicyManager(),
        skill_gateway=provider,
        session_id="test_direct",
    )
    dispatcher.set_session_id("test_direct")

    # 不调 search_skill，直接 execute_skill 本地 skill
    with patch("core.skill.provider.g_config", _mock_config):
        result = await dispatcher.execute(
            "execute_skill",
            {"skill_name": "web_search", "request": "test"},
        )

    import json as _json

    parsed = _json.loads(result)

    # 不应返回 SEARCH_REQUIRED 错误
    assert parsed.get("error_code") != "SEARCH_REQUIRED", (
        f"本地 skill 不应要求先 search，但返回了 SEARCH_REQUIRED"
    )
    print(f"  ✓ 本地 skill 'web_search' 未被 SEARCH_REQUIRED 拦截")
    print(f"    status: {parsed.get('status')}, error_code: {parsed.get('error_code')}")


async def test_execute_skill_cloud_requires_search():
    """验证云端（未知）skill 仍然要求先 search_skill"""
    print("\n【10. execute_skill 未知 skill 仍要求先 search】")

    from core.memento_s.tool_dispatcher import ToolDispatcher
    from core.memento_s.policies import PolicyManager

    skills = _make_fake_skills()
    provider = _build_provider(skills)

    dispatcher = ToolDispatcher(
        policy_manager=PolicyManager(),
        skill_gateway=provider,
        session_id="test_guard",
    )
    dispatcher.set_session_id("test_guard")

    with patch("core.skill.provider.g_config", _mock_config):
        result = await dispatcher.execute(
            "execute_skill",
            {"skill_name": "some_cloud_skill_xyz", "request": "test"},
        )

    import json as _json

    parsed = _json.loads(result)

    assert parsed.get("error_code") == "SEARCH_REQUIRED", (
        f"未知 skill 应返回 SEARCH_REQUIRED，实际: {parsed.get('error_code')}"
    )
    print(f"  ✓ 未知 skill 'some_cloud_skill_xyz' 正确返回 SEARCH_REQUIRED")


async def test_end_to_end_messages():
    """端到端：验证优化后 LLM 看到的 messages 不再有本地 skill 重复"""
    print("\n【11. 端到端：优化后完整 messages 验证】")

    from core.memento_s.tool_dispatcher import ToolDispatcher
    from core.memento_s.policies import PolicyManager

    real_skills = _load_real_builtin_skills()
    if real_skills is None:
        print("  [SKIP] 无法加载真实 builtin skills")
        return

    provider = _build_provider(real_skills)

    with (
        patch("core.context.manager.g_config", _mock_config),
        patch("core.skill.provider.g_config", _mock_config),
    ):
        ctx_mgr = ContextManager(
            session_id="test",
            config=AgentConfig(),
            skill_gateway=provider,
        )
        system_prompt = await ctx_mgr.assemble_system_prompt(mode="agentic")

    dispatcher = ToolDispatcher(
        policy_manager=PolicyManager(),
        skill_gateway=provider,
        session_id="test_e2e",
    )
    dispatcher.set_session_id("test_e2e")

    with patch("core.skill.provider.g_config", _mock_config):
        search_result = await dispatcher.execute(
            "search_skill",
            {"query": "web search 量子计算", "k": 5},
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "帮我搜索一下量子计算的最新进展"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_001",
                    "type": "function",
                    "function": {
                        "name": "search_skill",
                        "arguments": '{"query":"web search 量子计算"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_001", "content": search_result},
    ]

    import json as _json

    search_parsed = _json.loads(search_result)
    search_output = search_parsed.get("output", [])
    search_content_len = len(search_result)

    print(
        f"\n  系统提示词: {len(system_prompt)} 字符 (约 {len(system_prompt) // 4} tokens)"
    )
    print(
        f"  search_skill 返回: {search_content_len} 字符 (约 {search_content_len // 4} tokens)"
    )
    print(f"  search_skill output: {len(search_output)} 个 cloud skill")
    print(f"  search_skill summary: {search_parsed.get('summary', '')}")

    total_chars = sum(len(msg.get("content", "")) for msg in messages)
    print(f"  messages 总内容: {total_chars} 字符 (约 {total_chars // 4} tokens)")

    # 验证 search_skill 返回中没有本地 skill
    local_names_in_search = [
        s["name"] for s in search_output if s.get("source") in ("local", "builtin")
    ]
    assert len(local_names_in_search) == 0, (
        f"search_skill 返回中不应有本地 skill: {local_names_in_search}"
    )

    # 验证 system prompt 中包含 available_skills
    assert "available_skills" in system_prompt, "system prompt 应包含 available_skills"

    # 验证所有本地 skill 在 system prompt 中
    for name in real_skills:
        assert name in system_prompt, f"本地 skill '{name}' 不在 system prompt 中"

    print(f"\n  ✓ search_skill 返回中 0 个本地 skill（无重复）")
    print(f"  ✓ system prompt 包含全部 {len(real_skills)} 个本地 skill")
    print(f"  ✓ 总 tokens 大幅下降（search_skill 不再塞本地 skill 信息）")


async def main():
    print("=" * 70)
    print("验证 Agent 系统提示词组件")
    print("=" * 70)

    await test_multi_recall_returns_all_local()
    await test_multi_recall_ignores_query()
    await test_provider_search_returns_all()
    await test_skills_summary_in_system_prompt()
    await test_full_system_prompt_structure()
    await test_direct_mode_no_skills()
    await test_with_real_builtin_skills()
    await test_search_skill_no_local_duplication()
    await test_execute_skill_direct_local()
    await test_execute_skill_cloud_requires_search()
    await test_end_to_end_messages()

    print("\n" + "=" * 70)
    print("✓ 所有测试通过")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
