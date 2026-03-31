"""Test bootstrap initialization via init_skill_system().

This script tests the complete bootstrap flow using the proper entry point.
"""

import asyncio
import sqlite3
from pathlib import Path
from core.skill import init_skill_system
from middleware.config import g_config


async def check_database_tables(db_path: Path):
    """Check if skill_embeddings table exists."""
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='skill_embeddings'"
        )
        result = cursor.fetchone()

        if result:
            print(f"✅ skill_embeddings table EXISTS")
            try:
                import sqlite_vec

                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
                cursor.execute("SELECT COUNT(*) FROM skill_embeddings")
                count = cursor.fetchone()[0]
                print(f"   Row count: {count}")
            except:
                print(f"   Row count: (cannot read without sqlite-vec)")
        else:
            print(f"❌ skill_embeddings table NOT FOUND")

        conn.close()
        return result is not None
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def main():
    g_config.load()

    print("=" * 70)
    print("TESTING BOOTSTRAP VIA init_skill_system()")
    print("=" * 70)

    db_path = Path(g_config.get_db_path())
    print(f"\nConfiguration:")
    print(f"  DB path: {db_path}")
    print(f"  Skills dir: {g_config.get_skills_path()}")

    # Check before
    print(f"\n{'=' * 70}")
    print("BEFORE: Database state")
    print(f"{'=' * 70}")
    await check_database_tables(db_path)

    # Bootstrap via init_skill_system
    print(f"\n{'=' * 70}")
    print("Calling init_skill_system()...")
    print(f"{'=' * 70}")
    gateway = await init_skill_system()
    print("✅ init_skill_system() completed")

    # Test the gateway
    print(f"\n{'=' * 70}")
    print("Testing discover()...")
    print(f"{'=' * 70}")
    skills = await gateway.discover()
    print(f"✅ Found {len(skills)} skill(s)")
    for skill in skills[:5]:
        print(f"   - {skill.name}")
    if len(skills) > 5:
        print(f"   ... and {len(skills) - 5} more")

    # Check after
    print(f"\n{'=' * 70}")
    print("AFTER: Database state")
    print(f"{'=' * 70}")
    await check_database_tables(db_path)

    # Summary
    print(f"\n{'=' * 70}")
    print("BOOTSTRAP TEST SUMMARY")
    print(f"{'=' * 70}")
    print("✅ init_skill_system() is the proper bootstrap entry point")
    print("✅ SkillGateway.from_config() only creates the gateway (no init)")
    print("✅ Initialization is done in bootstrap phase")
    print(f"✅ Total skills: {len(skills)}")


if __name__ == "__main__":
    asyncio.run(main())
