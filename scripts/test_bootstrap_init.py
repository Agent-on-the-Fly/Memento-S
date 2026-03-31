"""Test complete bootstrap initialization flow.

This script tests the full initialization process including:
1. VectorStorage table creation
2. Builtin skills sync
3. Complete initialize flow
"""

import asyncio
import sqlite3
from pathlib import Path
from core.skill.config import SkillConfig
from core.skill.gateway import SkillGateway
from middleware.config import g_config


async def check_database_tables(db_path: Path):
    """Check if skill_embeddings table exists."""
    print(f"\n{'=' * 70}")
    print("Checking database tables...")
    print(f"{'=' * 70}")

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check for skill_embeddings table
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='skill_embeddings'"
        )
        result = cursor.fetchone()

        if result:
            print(f"✅ skill_embeddings table EXISTS")

            # Check row count (need to load sqlite-vec extension)
            try:
                import sqlite_vec

                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
                cursor.execute("SELECT COUNT(*) FROM skill_embeddings")
                count = cursor.fetchone()[0]
                print(f"   Row count: {count}")
            except Exception as e:
                print(f"   Row count: (cannot read without sqlite-vec extension)")
        else:
            print(f"❌ skill_embeddings table NOT FOUND")

            # List all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print(f"   Available tables: {[t[0] for t in tables]}")

        conn.close()
        return result is not None
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        return False


async def test_bootstrap():
    """Test complete bootstrap flow."""
    g_config.load()

    config = SkillConfig.from_global_config()
    print("=" * 70)
    print("TESTING COMPLETE BOOTSTRAP FLOW")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Skills dir: {config.skills_dir}")
    print(f"  DB path: {config.db_path}")
    print(f"  Builtin skills dir: {config.builtin_skills_dir}")

    # Check database before initialization
    print(f"\n{'=' * 70}")
    print("BEFORE: Checking database state...")
    print(f"{'=' * 70}")
    await check_database_tables(config.db_path)

    # Create Gateway (this triggers full initialization)
    print(f"\n{'=' * 70}")
    print("Creating SkillGateway (triggers full bootstrap)...")
    print(f"{'=' * 70}")

    gateway = await SkillGateway.from_config(config)
    print(f"✅ SkillGateway created successfully")

    # Check database after initialization
    print(f"\n{'=' * 70}")
    print("AFTER: Checking database state...")
    print(f"{'=' * 70}")
    has_table = await check_database_tables(config.db_path)

    # Test discover
    print(f"\n{'=' * 70}")
    print("Testing discover()...")
    print(f"{'=' * 70}")

    skills = await gateway.discover()
    print(f"✅ Found {len(skills)} skill(s)")
    for skill in skills[:5]:
        print(f"   - {skill.name}")
    if len(skills) > 5:
        print(f"   ... and {len(skills) - 5} more")

    # Summary
    print(f"\n{'=' * 70}")
    print("BOOTSTRAP TEST SUMMARY")
    print(f"{'=' * 70}")

    if has_table:
        print("✅ SUCCESS: skill_embeddings table created")
        print("✅ Bootstrap flow is working correctly")
    else:
        print("⚠️  Table not created (may need embedding_client)")
        print("   Note: Table is created with default dimension=1536")
        print("   Embedding generation requires embedding_client")

    print(f"\n✅ Total skills discovered: {len(skills)}")


if __name__ == "__main__":
    asyncio.run(test_bootstrap())
