# SPDX-License-Identifier: Apache-2.0
"""真实场景 token 增长模拟（完整 token 计数）。

运行方式:
  source .venv/bin/activate && python scripts/test_token_budget_real.py
"""

from core.skill.execution.compaction import (
    TokenBudgetPolicy,
    make_budget_policy,
)


def simulate(
    name: str,
    turns: int,
    tool_calls_per_turn: int,
    chars_per_tool_result: int,
    system_prompt_chars: int,
    scratchpad_chars: int,
    policy: TokenBudgetPolicy,
) -> dict:
    """模拟完整 token 增长，返回各触发点。"""
    raw_msgs_chars = 0
    total_tool_msgs = 0
    warn_turn = None
    urgent_turn = None

    for turn in range(1, turns + 1):
        for _ in range(tool_calls_per_turn):
            raw_msgs_chars += chars_per_tool_result * 2
            total_tool_msgs += 2

        est = (system_prompt_chars + scratchpad_chars + raw_msgs_chars) // 4
        pct = est / policy.budget

        if warn_turn is None and pct >= policy.warn_ratio:
            warn_turn = turn
        if urgent_turn is None and pct >= policy.urgent_ratio:
            urgent_turn = turn

    return {
        "name": name,
        "warn_turn": warn_turn,
        "urgent_turn": urgent_turn,
        "total_turns": turns,
    }


def main():
    policy = make_budget_policy(context_window=128_000, max_output_tokens=4096)
    system_prompt_chars = 8000  # system prompt 估算（SKILL.md + 工具描述 + R1~R6）
    scratchpad_chars = 1000      # scratchpad 估算

    print(f"\n{'='*65}")
    print(f"Token 预算模拟（完整计数）")
    print(f"{'='*65}")
    print(f"  budget              : {policy.budget:,}")
    print(f"  warn 阈值 (80%)     : {policy.warn_threshold:,}")
    print(f"  urgent 阈值 (90%)   : {policy.urgent_threshold:,}")
    print(f"  system_prompt 估算  : {system_prompt_chars:,} chars → ~{system_prompt_chars//4:,} tokens")
    print(f"  scratchpad 估算     : {scratchpad_chars:,} chars → ~{scratchpad_chars//4:,} tokens")
    print(f"  固定开销合计        : ~{(system_prompt_chars+scratchpad_chars)//4:,} tokens")
    print(f"{'='*65}\n")

    scenarios = [
        # (name, turns, calls/turn, chars/call)
        ("短命令（bash ls/echo）", 200, 2, 80),
        ("中等输出（python_repl avg=300）", 200, 2, 300),
        ("大文件读取（read_file ~2KB）", 200, 2, 2000),
        ("大文件读取（read_file ~5KB）", 200, 2, 5000),
        ("真实 skill 混合（avg=500）", 200, 3, 500),
        ("密集 skill（avg=800）", 200, 4, 800),
    ]

    print(f"{'场景':<35} | {'microcompact 触发':<20} | {'Stage2 触发':<20}")
    print("-" * 80)

    for name, turns, cpw, cpc in scenarios:
        r = simulate(
            name=name,
            turns=turns,
            tool_calls_per_turn=cpw,
            chars_per_tool_result=cpc,
            system_prompt_chars=system_prompt_chars,
            scratchpad_chars=scratchpad_chars,
            policy=policy,
        )
        warn = f"turn {r['warn_turn']}" if r["warn_turn"] else "未触发"
        urgent = f"turn {r['urgent_turn']}" if r["urgent_turn"] else "未触发"
        print(f"{name:<35} | {warn:<20} | {urgent:<20}")

    # 激进阈值对比
    print(f"\n{'='*65}")
    print(f"激进阈值对比 (warn=50%, urgent=70%)")
    print(f"{'='*65}")

    aggressive = make_budget_policy(context_window=128_000, max_output_tokens=4096)
    aggressive.warn_ratio = 0.50
    aggressive.urgent_ratio = 0.70

    for name, turns, cpw, cpc in scenarios:
        r = simulate(
            name=name,
            turns=turns,
            tool_calls_per_turn=cpw,
            chars_per_tool_result=cpc,
            system_prompt_chars=system_prompt_chars,
            scratchpad_chars=scratchpad_chars,
            policy=aggressive,
        )
        warn = f"turn {r['warn_turn']}" if r["warn_turn"] else "未触发"
        urgent = f"turn {r['urgent_turn']}" if r["urgent_turn"] else "未触发"
        print(f"{name:<35} | {warn:<20} | {urgent:<20}")

    print(f"\n{'='*65}")
    print("超长 system_prompt 场景（SKILL.md 特别大的 skill）")
    print(f"{'='*65}")
    for big_sp in [15000, 25000]:
        print(f"\nsystem_prompt = {big_sp:,} chars")
        for name, turns, cpw, cpc in scenarios:
            r = simulate(
                name=name,
                turns=turns,
                tool_calls_per_turn=cpw,
                chars_per_tool_result=cpc,
                system_prompt_chars=big_sp,
                scratchpad_chars=scratchpad_chars,
                policy=policy,
            )
            warn = f"turn {r['warn_turn']}" if r["warn_turn"] else "未触发"
            urgent = f"turn {r['urgent_turn']}" if r["urgent_turn"] else "未触发"
            print(f"  {name:<33} | warn={warn:<18} | urgent={urgent:<18}")


if __name__ == "__main__":
    main()
