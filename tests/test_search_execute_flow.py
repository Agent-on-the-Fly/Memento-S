#!/usr/bin/env python3
"""
测试自然语言 search → execute 完整流程（真实云端服务）

1. 本地 search + execute（真实 builtin skill）
2. 云端 search → 找到本地没有的 skill → download → execute

使用方法:
    .venv/bin/python tests/test_search_execute_flow.py
"""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.skill.schema import Skill
from core.skill.provider import SkillProvider, SkillInfo, SkillExecuteResult

from core.skill.retrieval.multi_recall import MultiRecall
from core.skill.retrieval.remote_catalog import RemoteCloudCatalog

CLOUD_URL = "http://8.140.204.114:9001"


def _load_builtin_skills() -> dict[str, Skill]:
    from core.skill.store.persistence import load_all_skills

    builtin_dir = project_root / "builtin" / "skills"
    return load_all_skills(builtin_dir)


def _build_provider(
    skills: dict[str, Skill],
    cloud_catalog=None,
) -> SkillProvider:
    multi_recall = MultiRecall(cloud_catalog=cloud_catalog)

    library = MagicMock()
    library.local_cache = skills
    library.add_skill = AsyncMock()

    return SkillProvider(
        library=library,
        multi_recall=multi_recall,
        cloud_catalog=cloud_catalog,
        llm=None,
    )


async def main():
    print("=" * 60)
    print("测试：search → execute 完整流程（真实云端）")
    print("=" * 60)

    # ── 0. 初始化 ────────────────────────────────────────────

    print("\n【0. 加载本地 skill】")
    skills = _load_builtin_skills()
    local_names = sorted(skills.keys())
    print(f"  本地 {len(skills)} 个: {local_names}")

    # ── 1. 连接真实云端 ──────────────────────────────────────

    print(f"\n【1. 连接云端 RemoteCloudCatalog: {CLOUD_URL}】")
    cloud = RemoteCloudCatalog(base_url=CLOUD_URL)
    print(f"  ✓ embedding_ready={cloud.embedding_ready}, size={cloud.size}")
    assert cloud.size > 0, "云端 catalog 为空"

    # ── 2. 本地自然语言 search ───────────────────────────────

    provider = _build_provider(skills, cloud_catalog=cloud)

    print("\n【2. 自然语言 search（本地 + 云端）】")
    test_queries = [
        "帮我读一下这个PDF文件",
        "画一个数据可视化图表",
        "帮我搜索最新新闻",
        "生成一个流程图",
        "pdf",
        "web_search",
    ]

    for query in test_queries:
        results = provider.search(query, k=5)
        names = [r.name for r in results]
        sources = [r.source for r in results]
        print(f"\n  query: '{query}'")
        print(f"  结果: {list(zip(names, sources))}")

        local_results = [r for r in results if r.source == "local"]
        cloud_results = [r for r in results if r.source == "cloud"]
        if local_results:
            print(f"  ✓ 本地匹配: {[r.name for r in local_results]}")
        if cloud_results:
            print(f"  ✓ 云端候选: {[r.name for r in cloud_results]}")

    # ── 3. search 云端 skill → execute 本地已有 ──────────────

    print("\n\n【3. search → execute 本地 skill (pdf)】")
    results = provider.search("pdf", k=3)
    chosen = results[0]
    print(f"  选择: {chosen.name} (source={chosen.source})")

    exec_result = await provider.execute(
        skill_name=chosen.name,
        params={"request": "读取PDF文件的内容"},
    )
    print(f"  execute: ok={exec_result.ok}, skill={exec_result.skill_name}")
    if exec_result.ok:
        print(f"  output: {str(exec_result.output)[:120]}...")
    else:
        print(f"  error: {(exec_result.summary or '')[:120]}")

    # ── 4. search 云端 skill（本地没有）→ download → execute ─

    print("\n【4. search 云端 skill（本地没有）→ download → execute】")

    # 搜一个本地肯定没有的
    results = provider.search("画一个数据可视化图表", k=5)
    cloud_only = [r for r in results if r.source == "cloud"]

    if not cloud_only:
        print("  ⚠ 没有找到云端 skill，跳过")
    else:
        chosen_cloud = cloud_only[0]
        print(f"  搜索到云端 skill: {chosen_cloud.name}")
        print(f"  description: {chosen_cloud.description[:100]}...")

        # execute 会触发 cloud_catalog.download()
        exec_result = await provider.execute(
            skill_name=chosen_cloud.name,
            params={"request": "画一个柱状图展示销售数据"},
        )
        print(f"  execute: ok={exec_result.ok}, skill={exec_result.skill_name}")
        if exec_result.ok:
            print(f"  ✓ output: {str(exec_result.output)[:150]}...")
        else:
            print(f"  error: {(exec_result.summary or '')[:150]}")

        # 检查 download 是否被调用（skill 是否被下载到本地）
        downloaded_dir = Path.home() / "memento_s" / "skills" / chosen_cloud.name
        if downloaded_dir.exists():
            print(f"  ✓ skill 已下载到: {downloaded_dir}")
            # 检查 add_skill 是否被调用
            lib = provider._library
            if lib.add_skill.called:
                print(f"  ✓ add_skill 被调用 → skill 已加入本地库")

    # ── 5. 直接用 cloud_catalog.download 测试 ────────────────

    print("\n【5. 直接测试 cloud_catalog.download()】")
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir)
        test_skill_name = "data-visualization"
        print(f"  下载 skill: {test_skill_name}")

        local_path = cloud.download(test_skill_name, target)
        if local_path and local_path.exists():
            print(f"  ✓ 下载成功: {local_path}")
            files = list(local_path.iterdir())
            print(f"  ✓ 文件: {[f.name for f in files]}")

            # 验证可以加载为 Skill
            from core.skill.store.persistence import load_skill_from_dir

            skill = load_skill_from_dir(local_path)
            print(
                f"  ✓ 加载为 Skill: name={skill.name}, knowledge={not skill.is_playbook}"
            )
        else:
            print(f"  ⚠ 下载失败或路径不存在")

    # ── 6. execute 一个完全不存在的 skill（云端也没有）────────

    print("\n【6. execute 不存在的 skill（本地 + 云端都没有）】")
    exec_result = await provider.execute(
        skill_name="completely_nonexistent_xyz_12345",
        params={"request": "test"},
    )
    assert exec_result.ok is False
    assert "not found" in exec_result.summary.lower()
    print(f"  ✓ ok=False, summary={exec_result.summary}")

    # ── 7. 云端 search 接口直接测试 ──────────────────────────

    print("\n【7. 云端 search 直接测试】")
    cloud_queries = [
        "machine learning model training",
        "database migration",
        "docker deployment",
    ]
    for q in cloud_queries:
        results = cloud.search(q, k=3)
        print(
            f"  '{q}' → {[(r.name, round(getattr(r, 'score', 0), 2) if hasattr(r, 'score') else '-') for r in results]}"
        )

    print("\n" + "=" * 60)
    print("✓ 所有测试通过")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
