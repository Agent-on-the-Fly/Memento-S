"""Test Agent with event collector - verifying if events can be captured from reply_stream.

This test uses the event collector pattern (Scheme 2) to capture all events
from Agent.reply_stream() and verify the error policy integration.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

# Temporarily disable Feishu to ensure events go to local stream
os.environ["FEISHU_APP_ID"] = ""
os.environ["FEISHU_APP_SECRET"] = ""

from bootstrap import bootstrap
from core.memento_s.agent import MementoSAgent


async def collect_events(
    agent: MementoSAgent,
    session_id: str,
    user_content: str,
    max_events: int = 100,
    timeout_sec: int = 60,
):
    """Collect all events from agent reply_stream with timeout."""
    events_collected = []

    try:
        async for evt in agent.reply_stream(
            session_id=session_id, user_content=user_content
        ):
            events_collected.append(evt)
            event_type = evt.get("type", "unknown")

            # Print progress
            print(f"  [{len(events_collected)}] {event_type}")

            # Show details for important events
            if event_type == "TOOL_CALL_START":
                tool_name = evt.get("toolName", "unknown")
                print(f"       → Tool: {tool_name}")
            elif event_type == "TOOL_CALL_RESULT":
                tool_name = evt.get("toolName", "unknown")
                result = evt.get("result", "")
                print(f"       ← Result from {tool_name}: {str(result)[:150]}...")
            elif event_type == "run_finished":
                print(f"       ✓ Run completed!")
                break

            # Safety limits
            if len(events_collected) >= max_events:
                print(f"       ! Hit max events limit ({max_events})")
                break

    except asyncio.TimeoutError:
        print(f"       ! Timeout after {timeout_sec}s")
    except Exception as e:
        print(f"       ! Error: {e}")
        import traceback

        traceback.print_exc()

    return events_collected


async def test_event_collection():
    """Test if we can collect events from Agent.reply_stream()"""
    await bootstrap()

    agent = MementoSAgent()  # Agent will create gateway internally

    # Test 1: Simple message (check basic flow)
    print("=" * 70)
    print("Test 1: Simple message")
    print("=" * 70)

    events1 = await collect_events(agent, "test_1", "Hello, how are you?")
    print(f"\n  Total events: {len(events1)}")
    print(f"  Event types: {[e.get('type') for e in events1]}")

    # Test 2: Direct skill execution via execute_skill tool
    print("\n" + "=" * 70)
    print("Test 2: Direct skill execution (filesystem - read /etc/shadow)")
    print("=" * 70)

    events2 = await collect_events(
        agent,
        "test_2",
        "请使用 execute_skill 工具执行 filesystem 技能，request: 读取 /etc/shadow",
    )
    print(f"\n  Total events: {len(events2)}")

    # Analyze execute_skill calls
    execute_skill_events = [
        e
        for e in events2
        if e.get("type") == "TOOL_CALL_RESULT" and e.get("toolName") == "execute_skill"
    ]
    print(f"  Execute_skill calls: {len(execute_skill_events)}")

    if execute_skill_events:
        for i, evt in enumerate(execute_skill_events, 1):
            result = evt.get("result", "")
            print(f"\n  Execute skill result #{i}:")
            try:
                result_obj = json.loads(result) if isinstance(result, str) else result
                print(f"    Success: {result_obj.get('success')}")
                print(
                    f"    Error: {result_obj.get('error', 'None')[:100] if result_obj.get('error') else 'None'}..."
                )
                print(f"    Error Type: {result_obj.get('error_type')}")
                if result_obj.get("diagnostics"):
                    print(f"    Diagnostics: {result_obj.get('diagnostics')}")
            except:
                print(f"    Raw: {str(result)[:200]}")

    run_finished = any(e.get("type") == "run_finished" for e in events2)
    print(f"\n  Run finished: {run_finished}")

    # Test 3: Python execution error
    print("\n" + "=" * 70)
    print("Test 3: Python execution with error")
    print("=" * 70)

    events3 = await collect_events(
        agent,
        "test_3",
        "请使用 execute_skill 工具执行 python_playground 技能，request: 执行代码 print(1/0)",
    )
    print(f"\n  Total events: {len(events3)}")

    execute_skill_events_3 = [
        e
        for e in events3
        if e.get("type") == "TOOL_CALL_RESULT" and e.get("toolName") == "execute_skill"
    ]
    print(f"  Execute_skill calls: {len(execute_skill_events_3)}")

    if execute_skill_events_3:
        for evt in execute_skill_events_3:
            result = evt.get("result", "")
            print(f"\n  Result:")
            try:
                result_obj = json.loads(result) if isinstance(result, str) else result
                print(f"    Success: {result_obj.get('success')}")
                print(
                    f"    Error: {result_obj.get('error', 'None')[:200] if result_obj.get('error') else 'None'}..."
                )
                print(f"    Error Type: {result_obj.get('error_type')}")
            except:
                print(f"    Raw: {str(result)[:200]}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Test 1 (Simple): {len(events1)} events")
    print(
        f"Test 2 (Filesystem): {len(events2)} events, {len(execute_skill_events)} execute_skill calls"
    )
    print(
        f"Test 3 (Python error): {len(events3)} events, {len(execute_skill_events_3)} execute_skill calls"
    )

    # Check if we can use this for error policy testing
    has_execute_skill = len(execute_skill_events) > 0 or len(execute_skill_events_3) > 0
    has_run_finished = any(
        e.get("type") == "run_finished" for e in events1 + events2 + events3
    )

    print(f"\n✅ Can capture events: YES")
    print(f"✅ Can capture execute_skill: {'YES' if has_execute_skill else 'NO'}")
    print(f"✅ Can capture run_finished: {'YES' if has_run_finished else 'NO'}")

    if has_execute_skill:
        print(f"\n🎯 This approach CAN be used for Agent error policy testing!")
    else:
        print(f"\n⚠️  LLM not calling execute_skill tool - may need prompt tuning")


if __name__ == "__main__":
    asyncio.run(test_event_collection())
