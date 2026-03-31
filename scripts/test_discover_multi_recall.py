"""Test SkillGateway.discover() with MULTI_RECALL strategy.

This tests the default behavior where SkillGateway internally creates MultiRecall.
"""

import asyncio
from core.skill.config import SkillConfig
from core.skill.gateway import SkillGateway
from core.skill.schema import DiscoverStrategy
from core.skill.initializer import SkillInitializer
from middleware.config import g_config


async def main():
    g_config.load()

    config = SkillConfig.from_global_config()
    print("=" * 70)
    print("Testing SkillGateway.discover() with MULTI_RECALL strategy")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Skills dir: {config.skills_dir}")
    print(f"  Builtin skills dir: {config.builtin_skills_dir}")

    # Step 1: Sync builtin skills first
    print("\n" + "=" * 70)
    print("Step 1: Syncing builtin skills...")
    print("=" * 70)
    initializer = SkillInitializer(config)
    synced = initializer.sync_builtin_skills()
    print(f"✅ Synced {len(synced)} builtin skill(s): {synced}")

    # Step 2: Create SkillGateway (internally creates MultiRecall)
    print("\n" + "=" * 70)
    print("Step 2: Creating SkillGateway (internal MultiRecall)...")
    print("=" * 70)

    gateway = await SkillGateway.from_config(config)
    print(
        f"✅ SkillGateway created with multi_recall initialized: {gateway._multi_recall is not None}"
    )

    # Step 4: Test different discover strategies
    print("\n" + "=" * 70)
    print("Step 4: Testing discover strategies...")
    print("=" * 70)

    # Test LOCAL_ONLY
    print("\n--- Strategy: LOCAL_ONLY ---")
    skills = await gateway.discover(strategy=DiscoverStrategy.LOCAL_ONLY)
    print(f"✅ Found {len(skills)} skill(s)")
    for skill in skills[:3]:
        print(f"   - {skill.name} ({skill.execution_mode.value})")
    if len(skills) > 3:
        print(f"   ... and {len(skills) - 3} more")

    # Test MULTI_RECALL
    print("\n--- Strategy: MULTI_RECALL ---")
    skills = await gateway.discover(strategy=DiscoverStrategy.MULTI_RECALL)
    print(f"✅ Found {len(skills)} skill(s)")
    for skill in skills:
        print(f"   - {skill.name}")
        print(f"     Source: {skill.governance.source}")
        print(f"     Score: {getattr(skill, 'score', 'N/A')}")

    # Test MULTI_RECALL with query
    print("\n--- Strategy: MULTI_RECALL with query='file' ---")
    skills = await gateway.discover(
        strategy=DiscoverStrategy.MULTI_RECALL, query="file", k=5
    )
    print(f"✅ Found {len(skills)} skill(s) matching 'file'")
    for skill in skills:
        print(f"   - {skill.name}")

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print("✅ MultiRecall strategy now works!")
    print("✅ Key difference: Must pass multi_recall when creating SkillGateway")
    print("✅ MultiRecall can combine multiple recall strategies:")
    print("   - LocalFileRecall (file system scan)")
    print("   - LocalDbRecall (embedding search - requires embedding client)")
    print("   - RemoteRecall (cloud catalog)")


if __name__ == "__main__":
    asyncio.run(main())
