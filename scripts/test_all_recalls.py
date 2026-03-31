"""Test all recall strategies: LocalDbRecall, RemoteRecall, and MultiRecall with discover().

This script tests each recall strategy individually and then combines them in MultiRecall.
"""

import asyncio
from core.skill.config import SkillConfig
from core.skill.gateway import SkillGateway
from core.skill.schema import DiscoverStrategy
from core.skill.initializer import SkillInitializer
from core.skill.retrieval import (
    MultiRecall,
    LocalFileRecall,
    LocalDbRecall,
    RemoteRecall,
)
from core.skill.retrieval.schema import RecallCandidate
from middleware.config import g_config
from pathlib import Path


async def test_local_file_recall(skills_dir: Path):
    """Test LocalFileRecall strategy."""
    print("\n" + "=" * 70)
    print("TEST 1: LocalFileRecall (文件系统扫描)")
    print("=" * 70)

    try:
        recall = LocalFileRecall(skills_dir)
        print(f"✅ Created LocalFileRecall")
        print(f"   Skills dir: {skills_dir}")
        print(f"   Available: {recall.is_available()}")

        # Search without query (list all)
        results = await recall.search("", k=10)
        print(f"\n✅ Search '' (list all): {len(results)} results")
        for r in results[:3]:
            print(f"   - {r.name} (score: {r.score}, source: {r.source})")
        if len(results) > 3:
            print(f"   ... and {len(results) - 3} more")

        # Search with query
        results = await recall.search("file", k=5)
        print(f"\n✅ Search 'file': {len(results)} results")
        for r in results:
            print(f"   - {r.name} (score: {r.score})")

        return recall
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return None


async def test_local_db_recall(config: SkillConfig):
    """Test LocalDbRecall strategy."""
    print("\n" + "=" * 70)
    print("TEST 2: LocalDbRecall (数据库向量召回)")
    print("=" * 70)

    try:
        # Check if sqlite-vec is available
        try:
            import sqlite_vec

            print(f"✅ sqlite-vec available: {sqlite_vec.__version__}")
        except ImportError:
            print("❌ sqlite-vec not available, skipping LocalDbRecall test")
            return None

        # Create LocalDbRecall
        recall = LocalDbRecall(str(config.db_path))
        print(f"✅ Created LocalDbRecall")
        print(f"   DB path: {config.db_path}")
        print(f"   Available: {recall.is_available()}")

        # Check stats
        stats = recall.get_stats()
        print(f"\n📊 Stats:")
        for key, value in stats.items():
            print(f"   {key}: {value}")

        # Search without query
        results = await recall.search("", k=10)
        print(f"\n✅ Search '' (list all): {len(results)} results")
        for r in results[:3]:
            print(f"   - {r.name} (score: {r.score}, source: {r.source})")
        if len(results) > 3:
            print(f"   ... and {len(results) - 3} more")

        # Search with query
        results = await recall.search("document", k=5)
        print(f"\n✅ Search 'document': {len(results)} results")
        for r in results:
            print(f"   - {r.name} (score: {r.score})")

        return recall
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return None


async def test_remote_recall():
    """Test RemoteRecall strategy."""
    print("\n" + "=" * 70)
    print("TEST 3: RemoteRecall (云端技能目录)")
    print("=" * 70)

    try:
        # Create RemoteRecall
        base_url = "http://8.140.204.114:9001"
        recall = RemoteRecall(base_url)
        print(f"✅ Created RemoteRecall")
        print(f"   Base URL: {base_url}")
        print(f"   Available: {recall.is_available()}")

        # Search
        results = await recall.search("pdf", k=5)
        print(f"\n✅ Search 'pdf': {len(results)} results")
        for r in results[:5]:
            print(f"   - {r.name}")
            print(
                f"     Description: {r.description[:50]}..."
                if r.description
                else "     Description: N/A"
            )
            print(f"     Score: {r.score}")

        return recall
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return None


async def test_multi_recall_discover(config: SkillConfig, skills_dir: Path):
    """Test MultiRecall with discover()."""
    print("\n" + "=" * 70)
    print("TEST 4: MultiRecall + discover()")
    print("=" * 70)

    # Create all recall strategies
    recalls = []

    # 1. LocalFileRecall
    local_file = LocalFileRecall(skills_dir)
    recalls.append(local_file)
    print(f"✅ Added LocalFileRecall")

    # 2. LocalDbRecall (if available)
    try:
        import sqlite_vec

        local_db = LocalDbRecall(str(config.db_path))
        recalls.append(local_db)
        print(f"✅ Added LocalDbRecall")
    except ImportError:
        print("⚠️  LocalDbRecall skipped (sqlite-vec not available)")

    # 3. RemoteRecall
    try:
        remote = RemoteRecall("http://8.140.204.114:9001")
        recalls.append(remote)
        print(f"✅ Added RemoteRecall")
    except Exception as e:
        print(f"⚠️  RemoteRecall skipped: {e}")

    # Create Gateway (internally creates MultiRecall)
    gateway = await SkillGateway.from_config(config)
    print(f"✅ SkillGateway created")

    # Test discover with MULTI_RECALL
    print("\n--- Test: discover(strategy=MULTI_RECALL) ---")
    skills = await gateway.discover(strategy=DiscoverStrategy.MULTI_RECALL)
    print(f"✅ Found {len(skills)} skill(s)")

    # Group by source
    by_source = {}
    for skill in skills:
        source = skill.governance.source
        by_source.setdefault(source, []).append(skill.name)

    print("\n📊 Results by source:")
    for source, names in by_source.items():
        print(f"   {source}: {len(names)} skills - {names}")

    # Test with query
    print("\n--- Test: discover(strategy=MULTI_RECALL, query='document') ---")
    skills = await gateway.discover(
        strategy=DiscoverStrategy.MULTI_RECALL, query="document", k=10
    )
    print(f"✅ Found {len(skills)} skill(s) matching 'document'")
    for skill in skills:
        print(f"   - {skill.name} (source: {skill.governance.source})")


async def main():
    g_config.load()
    config = SkillConfig.from_global_config()

    print("=" * 70)
    print("TESTING ALL RECALL STRATEGIES")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Skills dir: {config.skills_dir}")
    print(f"  DB path: {config.db_path}")
    print(f"  Builtin skills dir: {config.builtin_skills_dir}")

    # Sync builtin skills
    print("\n" + "=" * 70)
    print("SETUP: Syncing builtin skills...")
    print("=" * 70)
    initializer = SkillInitializer(config)
    synced = initializer.sync_builtin_skills()
    print(f"✅ Synced {len(synced)} skill(s): {synced}")

    # Run all tests
    await test_local_file_recall(config.skills_dir)
    await test_local_db_recall(config)
    await test_remote_recall()
    await test_multi_recall_discover(config, config.skills_dir)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print("✅ LocalFileRecall: 文件系统扫描，返回本地 skills")
    print("✅ LocalDbRecall: 数据库向量召回，支持语义搜索")
    print("✅ RemoteRecall: 云端技能目录，扩展 skill 库")
    print("✅ MultiRecall: 多路召回合并器，可组合多种策略")
    print("\n使用建议:")
    print("  - 本地开发: LocalFileRecall 足够")
    print("  - 需要语义搜索: LocalDbRecall (需 embedding)")
    print("  - 云端+本地: MultiRecall + LocalFileRecall + RemoteRecall")


if __name__ == "__main__":
    asyncio.run(main())
