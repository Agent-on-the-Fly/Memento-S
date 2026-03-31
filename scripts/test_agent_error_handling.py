"""Test Agent error handling through event collection.

Tests how Agent handles various error scenarios:
- Permission denied
- Execution errors
- Dependency errors
- Resource missing
- etc.

This test captures TOOL_CALL_RESULT events to verify error classification.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

# Disable Feishu to ensure events go to local stream
os.environ["FEISHU_APP_ID"] = ""
os.environ["FEISHU_APP_SECRET"] = ""

from bootstrap import bootstrap
from core.memento_s.agent import MementoSAgent


async def run_agent_test(
    agent: MementoSAgent, session_id: str, user_content: str, max_events: int = 100
):
    """Run agent and collect all events."""
    events = []

    try:
        async for evt in agent.reply_stream(
            session_id=session_id, user_content=user_content
        ):
            events.append(evt)

            event_type = evt.get("type", "unknown")

            # Print progress for important events
            if event_type == "TOOL_CALL_START":
                tool_name = evt.get("toolName", "unknown")
                print(f"    → Calling: {tool_name}")
            elif event_type == "TOOL_CALL_RESULT":
                tool_name = evt.get("toolName", "unknown")
                result = evt.get("result", "")
                success = "✓" if evt.get("success", False) else "✗"
                print(f"    ← Result [{success}]: {tool_name}")
            elif event_type == "RUN_FINISHED":
                print(f"    ✓ Run completed")
                break

            if len(events) >= max_events:
                print(f"    ! Hit max events limit")
                break

    except Exception as e:
        print(f"    ! Error: {e}")

    return events


def analyze_error(events: list[dict], tool_name: str = None) -> dict:
    """Analyze events for error information."""
    result = {
        "found_tool_call": False,
        "tool_name": None,
        "success": None,
        "error": None,
        "error_type": None,
        "has_run_finished": False,
    }

    for evt in events:
        if evt.get("type") == "TOOL_CALL_RESULT":
            if tool_name is None or evt.get("toolName") == tool_name:
                result["found_tool_call"] = True
                result["tool_name"] = evt.get("toolName")
                result["success"] = evt.get("success", False)

                # Parse result
                try:
                    result_data = evt.get("result", "")
                    if isinstance(result_data, str):
                        result_obj = json.loads(result_data)
                    else:
                        result_obj = result_data

                    result["error"] = result_obj.get("error")
                    result["error_type"] = result_obj.get("error_type")

                    # Check diagnostics
                    diagnostics = result_obj.get("diagnostics", {})
                    if diagnostics and not result["error_type"]:
                        result["error_type"] = diagnostics.get("error_type")

                except:
                    result["error"] = str(result_data)[:200]

        elif evt.get("type") == "RUN_FINISHED":
            result["has_run_finished"] = True

    return result


async def main():
    await bootstrap()

    agent = MementoSAgent()  # Agent will create gateway internally

    test_cases = [
        {
            "name": "permission_denied",
            "description": "Try to read /etc/shadow (should be permission denied)",
            "query": "使用 filesystem 工具读取 /etc/shadow 文件",
            "expected_error_in": ["permission", "denied", "access"],
        },
        {
            "name": "execution_error_python",
            "description": "Execute code with division by zero",
            "query": "使用 python_playground 工具执行代码: print(1/0)",
            "expected_error_in": ["zero", "division", "error"],
        },
        {
            "name": "resource_missing",
            "description": "Try to read non-existent file",
            "query": "使用 filesystem 工具读取 /tmp/this_file_does_not_exist_12345.txt",
            "expected_error_in": ["not found", "no such file", "exist"],
        },
        {
            "name": "normal_execution",
            "description": "Normal weather query (should succeed)",
            "query": "使用 weather 工具查询北京天气",
            "expected_error_in": [],  # Should succeed
        },
    ]

    results = []

    for i, case in enumerate(test_cases, 1):
        print(f"\n{'=' * 70}")
        print(f"Test {i}: {case['name']}")
        print(f"Description: {case['description']}")
        print(f"{'=' * 70}")

        session_id = f"test_{case['name']}"
        events = await run_agent_test(agent, session_id, case["query"])

        # Analyze results
        analysis = analyze_error(events)

        print(f"\n  Results:")
        print(f"    Total events: {len(events)}")
        print(f"    Tool called: {analysis['tool_name']}")
        print(f"    Success: {analysis['success']}")
        print(f"    Has run_finished: {analysis['has_run_finished']}")

        if analysis["error"]:
            print(f"    Error: {str(analysis['error'])[:150]}...")
        if analysis["error_type"]:
            print(f"    Error Type: {analysis['error_type']}")

        # Check if error matches expectation
        error_text = str(analysis.get("error", "")).lower()
        expected_found = (
            any(exp in error_text for exp in case["expected_error_in"])
            if case["expected_error_in"]
            else not analysis["error"]
        )

        status = "✓ PASS" if expected_found else "✗ FAIL"
        print(f"    Status: {status}")

        results.append(
            {
                "name": case["name"],
                "events_count": len(events),
                "tool_name": analysis["tool_name"],
                "success": analysis["success"],
                "error": analysis["error"],
                "error_type": analysis["error_type"],
                "has_run_finished": analysis["has_run_finished"],
                "expected_found": expected_found,
            }
        )

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    passed = sum(1 for r in results if r["expected_found"])
    total = len(results)

    for r in results:
        status = "✓" if r["expected_found"] else "✗"
        print(f"{status} {r['name']}: {r['tool_name']} (error_type={r['error_type']})")

    print(f"\nPassed: {passed}/{total}")

    # Detailed JSON output
    print(f"\nDetailed Results:")
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    asyncio.run(main())
