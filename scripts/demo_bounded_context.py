#!/usr/bin/env python3
"""Bounded Context Memory 演示脚本。

模拟一个完整的 agent session：
  Turn 1: 用户请求分析文件 → agent 调 3 个工具（含 1 个长输出 artifact fold）
  Turn 2: 用户追问 → agent 继续调工具，触发 compact
  Turn 3: 新用户输入 → 旧 block seal，新 block 开始

每一步打印:
  - 磁盘结构
  - runtime_state
  - bounded prompt 内容（system + events → messages + user）
  - prompt 总字符数

运行: python demo_bounded_context.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

# ── 确保项目在 path 里 ──
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ── 处理依赖：使用 stub 避免完整启动 ──
from unittest.mock import MagicMock

# Stub heavy deps
sys.modules.setdefault("utils.logger", MagicMock(get_logger=MagicMock(return_value=MagicMock())))
sys.modules.setdefault("utils", MagicMock())
sys.modules.setdefault("utils.token_utils", MagicMock())
for mod in [
    "core", "core.prompts", "core.prompts.templates", "core.prompts.prompt_builder",
    "core.skill", "core.skill.gateway", "core.utils",
    "middleware", "middleware.config", "litellm",
]:
    sys.modules.setdefault(mod, MagicMock())

import importlib.util

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

schemas_mod = _load("core.context.schemas", ROOT / "core" / "context" / "schemas.py")
runtime_mod = _load("core.context.runtime_state", ROOT / "core" / "context" / "runtime_state.py")
block_mod = _load("core.context.block", ROOT / "core" / "context" / "block.py")
scratchpad_mod = _load("core.context.scratchpad", ROOT / "core" / "context" / "scratchpad.py")

ContextConfig = schemas_mod.ContextConfig
RuntimeState = runtime_mod.RuntimeState
RuntimeStateStore = runtime_mod.RuntimeStateStore
BlockManager = block_mod.BlockManager
make_event = block_mod.make_event
ensure_session_dir = block_mod.ensure_session_dir
Scratchpad = scratchpad_mod.Scratchpad


# ── events → messages (same as ContextManager._events_to_messages) ──

def events_to_messages(events):
    messages = []
    for ev in events:
        t = ev.get("type", "")
        if t in ("user_input", "user"):
            messages.append({"role": "user", "content": ev.get("text", "")})
        elif t == "assistant":
            messages.append({"role": "assistant", "content": ev.get("text", "")})
        elif t == "tool_call":
            tc = {
                "id": ev.get("tool_call_id", ev.get("event_id", "")),
                "type": "function",
                "function": {"name": ev.get("tool_name", "?"), "arguments": ev.get("args_summary", "{}")},
            }
            if messages and messages[-1].get("role") == "assistant" and "tool_calls" in messages[-1]:
                messages[-1]["tool_calls"].append(tc)
            else:
                messages.append({"role": "assistant", "content": None, "tool_calls": [tc]})
        elif t == "tool_result":
            messages.append({
                "role": "tool",
                "tool_call_id": ev.get("tool_call_id", ev.get("event_id", "")),
                "content": ev.get("text", ""),
            })
        elif t == "tool_result_ref":
            ref, preview, size = ev.get("ref", ""), ev.get("preview", ""), ev.get("size", 0)
            messages.append({
                "role": "tool",
                "tool_call_id": ev.get("tool_call_id", ev.get("event_id", "")),
                "content": f"[artifact_ref: {ref}]\n{preview}\n[{size} chars archived]",
            })
    return messages


def build_runtime_state_section(store):
    rs = store.get()
    d = rs.to_dict()
    d.pop("session_id", None)
    d.pop("updated_at", None)
    return "## runtime_state\n```json\n" + json.dumps(d, ensure_ascii=False, indent=2) + "\n```"


def print_bounded_prompt(rs_store, bm, user_msg, cfg):
    block = bm.active_block
    recent = block.load_recent_events(k=cfg.bounded_recent_events_k) if block else []
    block_msgs = events_to_messages(recent)

    # Deduplicate trailing user
    if block_msgs and block_msgs[-1].get("role") == "user" and block_msgs[-1].get("content") == user_msg:
        block_msgs = block_msgs[:-1]

    system = "You are a helpful assistant.\n\n" + build_runtime_state_section(rs_store)
    prompt = [
        {"role": "system", "content": system},
        *block_msgs,
        {"role": "user", "content": user_msg},
    ]

    total_chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in prompt)
    print(f"\n  📋 Bounded Prompt ({len(prompt)} messages, {total_chars} chars):")
    for i, m in enumerate(prompt):
        role = m["role"]
        if role == "system":
            # 只打印前 200 字符
            content = m["content"][:200] + "..." if len(m["content"]) > 200 else m["content"]
            print(f"    [{i}] system: {content}")
        elif role == "user":
            print(f"    [{i}] user: {m['content'][:100]}")
        elif role == "assistant":
            if m.get("tool_calls"):
                names = [tc["function"]["name"] for tc in m["tool_calls"]]
                print(f"    [{i}] assistant: [tool_calls: {', '.join(names)}]")
            else:
                print(f"    [{i}] assistant: {str(m.get('content', ''))[:100]}")
        elif role == "tool":
            content = m.get("content", "")
            if "[artifact_ref:" in content:
                print(f"    [{i}] tool: [artifact ref] {content[:80]}...")
            else:
                print(f"    [{i}] tool: {content[:100]}")
    print(f"  📊 Total: {total_chars} chars")


def print_disk_tree(session_dir, ctx_dir):
    print("\n  💾 Disk structure:")
    for p in sorted(session_dir.rglob("*")):
        rel = p.relative_to(session_dir)
        if p.is_file():
            size = p.stat().st_size
            print(f"    {rel} ({size} bytes)")
    # Artifacts
    art_dirs = list(ctx_dir.rglob("artifacts_*"))
    for ad in art_dirs:
        for f in sorted(ad.iterdir()):
            print(f"    [artifact] {f.name} ({f.stat().st_size} bytes)")


# ════════════════════════════════════════════════════════════════
# Main demo
# ════════════════════════════════════════════════════════════════

def main():
    # Setup
    tmpdir = tempfile.mkdtemp(prefix="bounded_ctx_demo_")
    ctx_dir = Path(tmpdir)
    session_id = "demo_session"
    session_dir = ensure_session_dir(ctx_dir, session_id)
    date_dir = ctx_dir / "2026-03-26"
    date_dir.mkdir(parents=True, exist_ok=True)

    cfg = ContextConfig(
        bounded_prompt_enabled=True,
        bounded_recent_events_k=6,
        block_compact_threshold=10,
        artifact_fold_char_limit=200,
        artifact_fold_line_limit=10,
    )

    rs_store = RuntimeStateStore(session_id, session_dir)
    bm = BlockManager(session_id, session_dir)
    scratchpad = Scratchpad(session_id, date_dir,
                            artifact_fold_char_limit=cfg.artifact_fold_char_limit,
                            artifact_fold_line_limit=cfg.artifact_fold_line_limit)

    print("=" * 70)
    print("  Bounded Context Memory — 演示")
    print("=" * 70)

    # ── Turn 1: 用户请求分析文件 ──
    print("\n" + "─" * 70)
    print("  TURN 1: 用户请求 '帮我分析 data.csv'")
    print("─" * 70)

    user1 = "帮我分析 /tmp/data.csv 这个文件的内容"
    block1 = bm.create_block(user1)
    rs = rs_store.get()
    rs.on_user_input(user1, block_id=block1.block_id)
    rs_store.save(rs)

    # Agent 推理 + read_file
    block1.append_event(make_event("assistant", text="好的，让我先读取这个 CSV 文件看看内容。"))
    block1.append_event(make_event("tool_call", tool_name="read_file",
                                    args_summary='{"path": "/tmp/data.csv"}',
                                    extra={"tool_call_id": "tc_01"}))

    # 短结果
    short_csv = "name,age,city\nAlice,30,Beijing\nBob,25,Shanghai\nCharlie,35,Shenzhen"
    msg1 = scratchpad.persist_tool_result("tc_01", "read_file", short_csv)
    block1.append_event(make_event("tool_result", text=short_csv,
                                    extra={"tool_call_id": "tc_01"}, status="effective"))
    rs.on_effective_action("read /tmp/data.csv")

    # Agent 推理 + analyze (长输出)
    block1.append_event(make_event("assistant", text="数据只有 3 行，让我做一个详细分析。"))
    block1.append_event(make_event("tool_call", tool_name="analyze_csv",
                                    args_summary='{"path": "/tmp/data.csv", "mode": "full"}',
                                    extra={"tool_call_id": "tc_02"}))

    long_analysis = "=== CSV Analysis Report ===\n" + "\n".join(
        [f"Row {i}: name={chr(65+i)}, age={20+i}, city=City_{i}, status=active, score={i*10}"
         for i in range(50)]
    )
    msg2 = scratchpad.persist_tool_result("tc_02", "analyze_csv", long_analysis)
    print(f"\n  🔍 Tool result fold: {'FOLDED → artifact' if '[artifact_ref:' in msg2['content'] else 'INLINE'}")

    if "[artifact_ref:" in msg2["content"]:
        ref_line = msg2["content"].split("\n", 1)[0]
        ref_path = ref_line.replace("[artifact_ref: ", "").rstrip("]")
        preview = msg2["content"].split("\n", 2)[1] if "\n" in msg2["content"] else ""
        block1.append_event(make_event("tool_result_ref", ref=ref_path,
                                        preview=preview[:200], size=len(long_analysis),
                                        extra={"tool_call_id": "tc_02"}))
        rs.on_new_artifact(ref_path)
    else:
        block1.append_event(make_event("tool_result", text=long_analysis,
                                        extra={"tool_call_id": "tc_02"}, status="effective"))

    # Agent 结论
    block1.append_event(make_event("assistant", text="分析完成。文件包含 3 人的信息，平均年龄 30 岁。详细分析已归档。"))
    rs.on_effective_action("analyzed CSV")
    rs.on_run_finished()
    rs_store.save(rs)

    print_bounded_prompt(rs_store, bm, "（查看 Turn 1 结果）", cfg)
    print_disk_tree(session_dir, ctx_dir)

    # ── Turn 2: 用户追问，大量工具调用 → compact ──
    print("\n" + "─" * 70)
    print("  TURN 2: 用户追问 '按年龄排序并生成图表'，触发大量工具调用")
    print("─" * 70)

    user2 = "帮我按年龄排序，然后生成一个柱状图"

    # Seal old block + create new
    bm.seal_active_block()
    rs = rs_store.get()
    rs.on_block_sealed()

    block2 = bm.create_block(user2)
    rs.on_user_input(user2, block_id=block2.block_id)
    rs.active_block_id = block2.block_id
    rs_store.save(rs)

    # Simulate 6 rounds of tool calls
    tools = ["sort_data", "validate_output", "create_chart", "style_chart", "save_chart", "verify_chart"]
    for i, tool in enumerate(tools):
        block2.append_event(make_event("assistant", text=f"第 {i+1} 步: 执行 {tool}..."))
        block2.append_event(make_event("tool_call", tool_name=tool,
                                        args_summary=f'{{"step": {i+1}}}',
                                        extra={"tool_call_id": f"tc_2_{i}"}))
        result = f"Step {i+1} ({tool}): {'成功' if i != 1 else '需要修正格式'}" + " | detail: " + "x" * 200
        block2.append_event(make_event("tool_result", text=result,
                                        extra={"tool_call_id": f"tc_2_{i}"},
                                        status="effective" if i != 1 else "ineffective"))
        if i != 1:
            rs.on_effective_action(f"{tool} done")
        else:
            rs.on_ineffective_action(f"{tool} failed")

    block2.append_event(make_event("assistant", text="图表生成完成！已保存为 chart.png。"))
    rs.on_run_finished()
    rs_store.save(rs)

    events_before = block2.load_events()
    print(f"\n  📊 Block events BEFORE compact: {len(events_before)}")

    # Check compact threshold
    if len(events_before) > cfg.block_compact_threshold:
        stats = block2.compact_old_events(
            keep_recent=cfg.bounded_recent_events_k,
            slim_max_chars=120,
        )
        events_after = block2.load_events()
        print(f"  📊 Block events AFTER compact:  {len(events_after)} (same count, tool_result slimmed)")
        print(f"  📊 Tool results slimmed: {stats['tool_results_slimmed']}")

        # Show a slimmed event example
        slimmed = [e for e in events_after if e.get("_trimmed_from")]
        if slimmed:
            ex = slimmed[0]
            print(f"  📊 Example slimmed tool_result: '{ex['text'][:60]}...' (was {ex['_trimmed_from']} chars)")

    print_bounded_prompt(rs_store, bm, "（查看 Turn 2 结果）", cfg)

    # ── Turn 3: 新用户输入 → prompt 依然 bounded ──
    print("\n" + "─" * 70)
    print("  TURN 3: 新用户输入 '把图表发给我'")
    print("─" * 70)

    user3 = "把图表发给我的邮箱"

    bm.seal_active_block()
    rs = rs_store.get()
    rs.on_block_sealed()

    block3 = bm.create_block(user3)
    rs.on_user_input(user3, block_id=block3.block_id)
    rs.active_block_id = block3.block_id
    rs_store.save(rs)

    print_bounded_prompt(rs_store, bm, user3, cfg)
    print_disk_tree(session_dir, ctx_dir)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  总结")
    print("=" * 70)

    metas = bm.list_block_metas()
    print(f"\n  Blocks: {len(metas)}")
    for m in metas:
        print(f"    {m.block_id}: status={m.status}, events={m.event_count}, "
              f"artifacts={len(m.result_refs)}, user='{m.user_input_text[:40]}...'")

    rs_final = rs_store.get()
    print(f"\n  RuntimeState:")
    print(f"    turn_count: {rs_final.turn_count}")
    print(f"    current_goal: '{rs_final.current_goal_text[:50]}'")
    print(f"    recent_refs: {rs_final.recent_refs}")
    print(f"    need_replan: {rs_final.need_replan}")
    print(f"    status: {rs_final.current_status}")

    print(f"\n  ✅ Demo 完成。临时目录: {tmpdir}")
    print(f"     可以检查磁盘文件: ls -la {session_dir}/blocks/")
    print()


if __name__ == "__main__":
    main()
