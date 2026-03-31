"""Test SkillGateway.discover() method with different strategies."""

import asyncio
from core.skill.config import SkillConfig
from core.skill.gateway import SkillGateway
from core.skill.schema import DiscoverStrategy
from core.skill.initializer import SkillInitializer
from middleware.config import g_config


async def test_strategy(gateway, strategy_name, strategy):
    """Test discover with a specific strategy."""
    print(f"\n{'=' * 60}")
    print(f"Testing strategy: {strategy_name}")
    print(f"{'=' * 60}")

    try:
        skills = await gateway.discover(strategy=strategy)
        print(f"✅ Success! Found {len(skills)} skill(s):")

        for skill in skills:
            print(f"\n  📦 {skill.name}")
            print(f"     Description: {skill.description[:60]}...")
            print(f"     Execution Mode: {skill.execution_mode.value}")
            print(f"     Source: {skill.governance.source}")
            if skill.dependencies:
                print(f"     Dependencies: {skill.dependencies}")
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()


async def main():
    g_config.load()

    config = SkillConfig.from_global_config()
    print(f"Configuration:")
    print(f"  Skills dir: {config.skills_dir}")
    print(f"  Builtin skills dir: {config.builtin_skills_dir}")
    print(f"  Workspace dir: {config.workspace_dir}")
    print(f"  Cloud catalog URL: {config.cloud_catalog_url or 'Not configured'}")

    print("\n" + "=" * 60)
    print("Creating SkillGateway...")
    print("=" * 60)

    gateway = await SkillGateway.from_config(config)

    # Sync builtin skills first
    print(f"\n{'=' * 60}")
    print("Syncing builtin skills...")
    print(f"{'=' * 60}")
    initializer = SkillInitializer(config)
    synced = initializer.sync_builtin_skills()
    print(f"Synced {len(synced)} builtin skill(s): {synced}")

    # Test 1: No strategy (default)
    print(f"\n{'=' * 60}")
    print("Testing: discover() [no strategy parameter]")
    print(f"{'=' * 60}")
    skills = await gateway.discover()
    print(f"Result: {len(skills)} skill(s)")
    for skill in skills[:5]:  # Show first 5
        print(f"  - {skill.name}")
    if len(skills) > 5:
        print(f"  ... and {len(skills) - 5} more")

    # Test 2: LOCAL_ONLY strategy
    await test_strategy(gateway, "LOCAL_ONLY (default)", DiscoverStrategy.LOCAL_ONLY)

    # Test 3: MULTI_RECALL strategy
    await test_strategy(gateway, "MULTI_RECALL", DiscoverStrategy.MULTI_RECALL)

    # Test 4: String parameter
    print(f"\n{'=' * 60}")
    print("Testing strategy as string parameter:")
    print(f"{'=' * 60}")

    try:
        skills = await gateway.discover(strategy="local_only")
        print(f"✅ String 'local_only' works! Found {len(skills)} skill(s)")
    except Exception as e:
        print(f"❌ Error with string parameter: {e}")

    print(f"\n{'=' * 60}")
    print("DISCOVER TEST SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total strategies tested: 4")
    print(f"Strategies: default, LOCAL_ONLY, MULTI_RECALL, string 'local_only'")


if __name__ == "__main__":
    asyncio.run(main())
