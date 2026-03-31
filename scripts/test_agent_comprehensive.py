"""Comprehensive Agent Error Handling Tests

Tests Agent's error handling across multiple scenarios:
1. Permission errors
2. Execution errors
3. Resource missing
4. Timeout scenarios
5. Dependency errors
6. Invalid inputs
7. Environment errors
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from typing import Any

os.environ["FEISHU_APP_ID"] = ""
os.environ["FEISHU_APP_SECRET"] = ""

from bootstrap import bootstrap
from core.memento_s.agent import MementoSAgent


async def run_agent_collect_events(agent, session_id, query, max_events=100):
    """Run agent and collect all events."""
    events = []
    try:
        async for evt in agent.reply_stream(session_id=session_id, user_content=query):
            events.append(evt)
            if evt.get("type") == "RUN_FINISHED" or len(events) >= max_events:
                break
    except Exception as e:
        events.append({"type": "ERROR", "error": str(e)})
    return events


def extract_tool_calls_and_errors(events):
    """Extract all tool calls and their results/errors."""
    tool_calls = []
    run_finished = None

    for evt in events:
        evt_type = evt.get("type")

        if evt_type == "TOOL_CALL_START":
            tool_calls.append(
                {
                    "tool": evt.get("toolName"),
                    "status": "started",
                    "result": None,
                    "error": None,
                }
            )
        elif evt_type == "TOOL_CALL_RESULT":
            tool_name = evt.get("toolName")
            result = evt.get("result", "")
            success = evt.get("success", False)

            # Try to parse JSON result
            try:
                if isinstance(result, str):
                    result_obj = json.loads(result)
                else:
                    result_obj = result

                # Extract error info
                error = result_obj.get("error") or result_obj.get("summary", "")
                error_type = result_obj.get("error_type")
            except:
                result_obj = result
                error = str(result) if not success else None
                error_type = None

            tool_calls.append(
                {
                    "tool": tool_name,
                    "status": "completed",
                    "success": success,
                    "result": result_obj
                    if isinstance(result_obj, dict)
                    else str(result)[:200],
                    "error": error[:200] if error else None,
                    "error_type": error_type,
                }
            )
        elif evt_type == "RUN_FINISHED":
            run_finished = {
                "reason": evt.get("reason"),
                "outputText": evt.get("outputText", "")[:100],
            }

    return tool_calls, run_finished


async def main():
    print("=" * 80)
    print("COMPREHENSIVE AGENT ERROR HANDLING TESTS")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    await bootstrap()
    agent = MementoSAgent()  # Agent will create gateway internally

    test_cases = [
        # Group 1: Permission & Security
        {
            "id": "SEC001",
            "group": "Security",
            "name": "permission_denied_system_file",
            "query": "使用 filesystem 工具读取 /etc/shadow",
            "expected_error_keywords": ["denied", "access", "permission"],
            "expected_behavior": "Agent should detect permission error and explain",
        },
        {
            "id": "SEC002",
            "group": "Security",
            "name": "permission_denied_outside_workspace",
            "query": "使用 filesystem 工具读取 /tmp/outside_workspace.txt",
            "expected_error_keywords": ["denied", "access", "outside", "workspace"],
            "expected_behavior": "Agent should enforce workspace boundaries",
        },
        # Group 2: Execution Errors
        {
            "id": "EXE001",
            "group": "Execution",
            "name": "python_division_by_zero",
            "query": "使用 python_playground 执行代码 print(1/0)",
            "expected_error_keywords": ["zero", "division", "error", "exception"],
            "expected_behavior": "Agent should catch and report execution error",
        },
        {
            "id": "EXE002",
            "group": "Execution",
            "name": "python_syntax_error",
            "query": "使用 python_playground 执行代码 print('Hello'",
            "expected_error_keywords": ["syntax", "error", "invalid"],
            "expected_behavior": "Agent should handle syntax errors gracefully",
        },
        # Group 3: Resource Missing
        {
            "id": "RES001",
            "group": "Resource",
            "name": "file_not_found",
            "query": "使用 filesystem 工具读取 ./nonexistent_file_12345.txt",
            "expected_error_keywords": ["not found", "exist", "no such file"],
            "expected_behavior": "Agent should report resource missing",
        },
        {
            "id": "RES002",
            "group": "Resource",
            "name": "skill_not_found",
            "query": "使用 nonexistent_skill_abc123 工具做某事",
            "expected_error_keywords": ["not found", "unknown", "skill"],
            "expected_behavior": "Agent should handle unknown skill requests",
        },
        # Group 4: Normal Operations
        {
            "id": "NOR001",
            "group": "Normal",
            "name": "weather_query",
            "query": "查询北京天气",
            "expected_error_keywords": [],
            "expected_behavior": "Agent should successfully return weather data",
        },
        {
            "id": "NOR002",
            "group": "Normal",
            "name": "simple_chat",
            "query": "你好，请介绍一下自己",
            "expected_error_keywords": [],
            "expected_behavior": "Agent should respond without errors",
        },
        # Group 5: Complex Scenarios
        {
            "id": "COM001",
            "group": "Complex",
            "name": "multi_step_with_error",
            "query": "先读取 /etc/passwd，然后查询天气",
            "expected_error_keywords": ["denied", "access"],
            "expected_behavior": "Agent should handle first error then continue with weather",
        },
    ]

    results = []

    for i, case in enumerate(test_cases, 1):
        print(f"\n{'─' * 80}")
        print(f"Test {i}/{len(test_cases)}: [{case['id']}] {case['name']}")
        print(f"Group: {case['group']}")
        print(f"Expected: {case['expected_behavior']}")
        print(f"{'─' * 80}")

        session_id = f"test_{case['id']}_{datetime.now().strftime('%H%M%S')}"

        # Run test
        events = await run_agent_collect_events(agent, session_id, case["query"])
        tool_calls, run_finished = extract_tool_calls_and_errors(events)

        # Analyze results
        errors_found = []
        for tc in tool_calls:
            if tc.get("error"):
                errors_found.append(
                    {
                        "tool": tc["tool"],
                        "error": tc["error"],
                        "error_type": tc.get("error_type"),
                    }
                )

        # Check if expected error was found
        error_text = " ".join(
            [e["error"].lower() for e in errors_found if e["error"]]
        ).lower()
        expected_found = False

        if case["expected_error_keywords"]:
            expected_found = any(
                kw.lower() in error_text for kw in case["expected_error_keywords"]
            )
        else:
            expected_found = len(errors_found) == 0  # Should have no errors

        # Determine status
        if expected_found:
            status = "✓ PASS"
        else:
            if case["expected_error_keywords"]:
                status = "✗ FAIL (expected error not found)"
            else:
                status = "✗ FAIL (unexpected error)"

        print(f"\n  Results:")
        print(f"    Events: {len(events)}")
        print(
            f"    Tool calls: {len([tc for tc in tool_calls if tc['status'] == 'completed'])}"
        )
        print(f"    Errors found: {len(errors_found)}")
        print(f"    Run finished: {run_finished is not None}")

        if errors_found:
            print(f"\n    Errors:")
            for err in errors_found[:3]:  # Show first 3 errors
                print(f"      - {err['tool']}: {err['error'][:80]}...")

        print(f"\n    Status: {status}")

        results.append(
            {
                "id": case["id"],
                "group": case["group"],
                "name": case["name"],
                "status": "PASS" if expected_found else "FAIL",
                "events_count": len(events),
                "tool_calls": len(tool_calls),
                "errors_found": len(errors_found),
                "run_completed": run_finished is not None,
                "expected_found": expected_found,
            }
        )

    # Summary
    print(f"\n{'=' * 80}")
    print("TEST SUMMARY")
    print(f"{'=' * 80}")

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    total = len(results)

    # Group by category
    groups = {}
    for r in results:
        group = r["group"]
        if group not in groups:
            groups[group] = {"passed": 0, "total": 0}
        groups[group]["total"] += 1
        if r["status"] == "PASS":
            groups[group]["passed"] += 1

    print(f"\nOverall: {passed}/{total} passed ({passed / total * 100:.1f}%)")
    print()

    print("By Group:")
    for group, stats in sorted(groups.items()):
        print(f"  {group:12s}: {stats['passed']}/{stats['total']} passed")

    print()
    print("Detailed Results:")
    for r in results:
        icon = "✓" if r["status"] == "PASS" else "✗"
        print(
            f"  {icon} [{r['id']}] {r['name']:40s} ({r['events_count']:3d} events, {r['errors_found']} errors)"
        )

    print()
    print("Analysis:")
    print(f"  - Total events captured: {sum(r['events_count'] for r in results)}")
    print(
        f"  - Average events per test: {sum(r['events_count'] for r in results) / total:.1f}"
    )
    print(
        f"  - Tests with run completion: {sum(1 for r in results if r['run_completed'])}/{total}"
    )

    print(f"\n{'=' * 80}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")

    # JSON output for programmatic analysis
    print("\n\nJSON Results:")
    print(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total": total,
                    "passed": passed,
                    "failed": failed,
                    "pass_rate": passed / total * 100,
                },
                "by_group": {
                    k: {"passed": v["passed"], "total": v["total"]}
                    for k, v in groups.items()
                },
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
