# Bounded Context 架构合并改动说明

> 分支: `feature/merge-dev-v0.2.0-bounded-context`
> 来源: `opc_memento_s-dev_v0.2.0` → `opc_memento_s` (gitlab)

---

## 一、新增文件（2 个）

| 文件 | 说明 |
|------|------|
| `core/context/block.py` | Block 事件存储系统。每轮用户输入创建一个 Block（含 `events.jsonl`），支持封存、压缩、跨 block 上下文加载 |
| `core/context/runtime_state.py` | 会话级持久运行时状态（RuntimeState）。跟踪计划进度、artifact 引用、执行状态，磁盘持久化支持崩溃恢复 |

---

## 二、修改文件（16 个）

### 2.1 核心上下文管理 (`core/context/`)

| 文件 | 改动内容 |
|------|----------|
| `core/context/__init__.py` | 新增导出: `Block`, `BlockManager`, `BlockMeta`, `make_event`, `RuntimeState`, `RuntimeStateStore` |
| `core/context/schemas.py` | 新增 9 个配置字段: `artifact_fold_char_limit`, `artifact_fold_line_limit`, `artifact_preview_max_lines`, `artifact_preview_max_chars`, `bounded_prompt_enabled`, `bounded_recent_events_k`, `bounded_max_ref_previews`, `bounded_past_blocks_k`, `block_compact_threshold` |
| `core/context/manager.py` | **重点改动**: (1) 构造函数新增 session_dir、RuntimeStateStore、BlockManager 初始化; (2) 新增 `bounded_prompt_enabled` 属性; (3) `init_budget()` 增加 non-positive 安全兜底; (4) `append()` 增加 bounded 模式分支（跳过 LLM 压缩）; (5) `persist_tool_result()` 增加 artifact 引用追踪; (6) 新增 Runtime State 管理段（`runtime_state`, `save_runtime_state`, `sync_and_save_runtime_state`）; (7) 新增 Block 管理段（`start_new_block`, `append_block_event`, `compact_active_block_if_needed`）; (8) 新增 Bounded Prompt 组装段（`assemble_messages_bounded`, `_build_runtime_state_section`, `_build_ref_previews_section`, `_events_to_messages`）|
| `core/context/scratchpad.py` | 新增 artifact 折叠功能: `_make_preview()` 辅助函数, `_should_fold()` 判断是否需要折叠, `save_artifact()` 存储长内容为 artifact 文件, `persist_tool_result()` 改为短内容内联/长内容折叠双模式, `build_reference()` 增加 artifacts 目录提示 |

### 2.2 Agent 核心流程 (`core/memento_s/`)

| 文件 | 改动内容 |
|------|----------|
| `core/memento_s/agent.py` | (1) 新增 `make_event` 导入; (2) `reply_stream` 入口增加 block 生命周期管理（`start_new_block` + `runtime_state.on_user_input`）; (3) DIRECT/INTERRUPT/CONFIRM/AGENTIC 四条路径均增加 `bounded_prompt_enabled` 分支调用 `assemble_messages_bounded`; (4) AGENTIC 路径增加 plan 事件记录到 block; (5) 所有路径结束后增加 `runtime_state` 持久化 |
| `core/memento_s/finalize.py` | 新增 "Final Answer:" 前缀剥离逻辑 |
| `core/memento_s/phases/execution/runner.py` | (1) 新增 `NO_TOOL_NO_FINAL_ANSWER_MSG` 和 `make_event` 导入; (2) 增加 "Final Answer:" 检测（有前缀则标记 DONE，无前缀且无 tool_call 则注入 nudge 消息）; (3) 用户展示文本中剥离 "Final Answer:" 前缀; (4) 新增 block 事件记录（tool_call、tool_result、tool_result_ref）; (5) 执行后调用 `ctx.compact_active_block_if_needed()` |

### 2.3 Prompt 模板 (`core/prompts/`, `core/skill/`)

| 文件 | 改动内容 |
|------|----------|
| `core/prompts/templates.py` | (1) `PROTOCOL_AND_FORMAT` 新增 rule 6: "Final Answer:" 协议规则; (2) 新增 `NO_TOOL_NO_FINAL_ANSWER_MSG` 常量 |
| `core/skill/execution/prompts.py` | (1) 新增 `NO_TOOL_NO_FINAL_ANSWER_MSG` 常量; (2) skill 执行 prompt 的 Rules 段增加 "Final Answer:" 规则 |

### 2.4 中间件安全加固 (`middleware/`)

| 文件 | 改动内容 |
|------|----------|
| `middleware/config/config_models.py` | `input_budget` 属性增加安全兜底: 当 `max_tokens >= context_window` 时返回 `max(context_window // 2, 4096)` 而非负值 |
| `middleware/llm/llm_client.py` | `_load_config()` 增加 context_window 上限校验: 取 `min(模型自动检测值, 用户配置值)` 并回写 |

### 2.5 工具层 (`builtin/tools/`)

| 文件 | 改动内容 |
|------|----------|
| `builtin/tools/file_ops.py` | `create_file` 新增 `overwrite: bool = False` 参数，支持覆写已有文件 |
| `builtin/tools/registry.py` | `create_file` 工具 schema 新增 `overwrite` 布尔属性，描述更新为 "Create or overwrite a file" |

### 2.6 构建脚本 (`build_scripts/`)

| 文件 | 改动内容 |
|------|----------|
| `build_scripts/build_memento_s.py` | 新增 tiktoken 系列 hidden imports 和 collect-data/collect-all 配置; 新增 SSL/网络库 hidden imports（ssl, urllib3, requests, httpx, aiohttp 等） |

### 2.7 测试 (`tests/`)

| 文件 | 改动内容 |
|------|----------|
| `tests/context_gateway/test_scratchpad_persist_json.py` | 更新测试用例适配 artifact 折叠: 长结果测试检查 `[artifact_ref:` 标记而非内联内容; 新增 artifact 文件存在性检查 |
| `tests/test_context_manager.py` | 更新测试用例适配 artifact 折叠: `test_persist_tool_result_long_folded` 检查 artifact 引用; 新增 `test_persist_tool_result_artifact_preview_limits` 和 `test_persist_tool_result_multiple_artifacts_sequential` |

---

## 三、架构说明

本次合并引入了 **Bounded Context** 三层架构:

```
Layer 1: Block 事件存储
  每轮用户输入 → 创建新 Block → events.jsonl 追加事件
  Block 可被封存（seal）→ 旧 Block 事件被 slim 后作为跨轮上下文

Layer 2: Artifact 折叠
  长 tool result (>4000 字符 / >120 行) → 存为独立 artifact 文件
  Prompt 中仅保留 ref + preview（~500 字符）

Layer 3: Bounded Prompt 组装
  不从 DB 加载全量历史 → 从 active block 事件 + sealed block 摘要 + RuntimeState 构建
  System prompt 额外注入 runtime_state JSON 和 artifact preview
```

**"Final Answer:" 协议**: 要求 LLM 在完成任务时用 "Final Answer:" 前缀标记最终回复。无前缀且无 tool_call 时系统注入 nudge 消息要求继续。

---

## 四、保留的 GITLAB 原有改进

以下 GITLAB 已有的改进在合并中被完整保留:

- `core/shared/dependency_aliases.py` — 集中式依赖名归一化
- `core/skill/execution/policy/path_validator.py` — 所有 path-like 参数校验 + `@ROOT` 字面值防护
- `core/skill/execution/tool_bridge/` — 结构化错误报告、路径去重、严格越界检查
- `core/skill/execution/policy/pre_execute.py` — 通过 dependency_aliases 归一化
- `core/skill/execution/policy/recovery.py` — `INTERNAL_ERROR → ABORT`（保留 GITLAB 更安全的策略）
- `core/skill/execution/executor.py` — `session_info` 方式（保留 GITLAB 架构）
- GUI 所有改动（sidebar、file_browser、chat_message、toolbar、layout 等）
- 飞书 bridge 完整实现
- 自动更新加固脚本
- `builtin/tools/bash.py` — 结构化拒绝消息
- `builtin/tools/python_repl.py` — dependency_aliases 归一化
- `middleware/sandbox/` — `.output/` 子目录结构 + `work_dir` 参数

---

## 五、Bug Fix: compact 时 tool_result 原文丢失

**问题**: `Block.compact_old_events()` 在压缩旧轮次的 `tool_result` 事件时，直接将 text 硬截断为 120 字符并全量重写 `events.jsonl`，原始完整内容**不可恢复**。`_trimmed_from` 字段只记录了原文长度，没有提供原文存储位置。

**修复**:
| 改动 | 说明 |
|------|------|
| `core/context/block.py` — 新增 `_fold_tool_result_event()` | compact 时将超长 `tool_result` 原文写入 `block_NNNN/artifacts/{event_id}.txt`，事件类型从 `tool_result` 转为 `tool_result_ref`（含 `ref` 路径 + `preview`）。写入失败时降级为硬截断，不丢失事件 |
| `core/context/block.py` — 修改 `compact_old_events()` | 旧区事件处理从 `_slim_tool_result_event()`（硬截断）改为 `_fold_tool_result_event()`（落盘+转引用） |
| `core/context/block.py` — 保留 `_slim_tool_result_event()` | 仍用于 `load_recent_sealed_events()` 跨 block 回溯场景（不需要保留原文） |
| `core/context/block.py` — 更新模块文档 | 移除未实现的 `kept_event_ids.json` 和 `folded_chunks/` 描述，目录结构增加 `artifacts/` 示例 |

**修复后目录结构**:
```
blocks/
  block_0001/
    block_meta.json
    events.jsonl          # tool_result → tool_result_ref（含 ref 指向 artifacts/）
    artifacts/
      e0003.txt           # compact 时落盘的原始 tool_result 全文
      e0007.txt
```

---

## 六、改进: artifact 引用全量保留，移除滑动窗口

**问题**: `RuntimeState.recent_refs` 仅保留最近 3 个 artifact 引用（滑动窗口），早期 artifact 从 LLM 所有可见上下文中完全消失，导致用户回溯早期操作结果时 LLM 无法响应。

**修复**:
| 改动 | 说明 |
|------|------|
| `core/context/runtime_state.py` — 删除 `MAX_RECENT_REFS = 3` | 移除滑动窗口上限 |
| `core/context/runtime_state.py` — 修改 `on_new_artifact()` | 不再截断，保留全部 artifact 引用（增加去重） |
| `core/context/manager.py` — 修改 `_build_ref_previews_section()` | 移除 `[-bounded_max_ref_previews:]` 切片，全量注入 artifact preview 到 system prompt |

**理由**: `block.py` 已对 `tool_result` 做了 fold（原文落盘 + 只留 ref+preview），上下文本身已精炼，无需在 `recent_refs` 层再做截断。

---

## 七、Bug Fix: runner.py block event 截断导致原文丢失

**问题**: `runner.py` 记录 block event 时将 tool result 硬截断为 300 字符（`result[:300]`），而 scratchpad fold 阈值为 4000 字符。当 result 长度处于 300~4000 区间时，scratchpad 不会存 artifact（未达阈值），block event 只存了截断版，原文两边都不保留。后续 `block.compact` fold 的也只是截断后的 300 字符，不是原始内容。

**修复**:
| 改动 | 说明 |
|------|------|
| `core/memento_s/phases/execution/runner.py` — 修改 block event 记录 | `text=result[:300]` → `text=result`，存完整原文。折叠统一由 `block.compact_old_events()` 的 `_fold_tool_result_event()` 负责 |

**修复后数据流**:
```
tool result (任意长度)
  → block event: text=完整原文（临时存在于 events.jsonl）
  → compact 触发: _fold_tool_result_event() 将超长原文落盘到 artifacts/，转为 tool_result_ref
```

---

## 八、重构: 合并 scratchpad/block 双路径为 Block 统一处理

**问题**: tool result 同时走 scratchpad（实时 fold，阈值 4000 字符）和 block（存全文，compact 时 fold，阈值 120 字符）两条路径，产生两个 artifact 目录、两套引用体系，fold 判定不一致。

**重构**:
| 改动 | 说明 |
|------|------|
| `core/context/block.py` — 新增 `Block.persist_tool_result()` | 一步完成 fold 判定 + artifact 落盘 + event 记录 + 返回 tool_msg。短内容记录 `tool_result` event 并内联返回；长内容（>4000 字符 / >120 行）存入 `block_NNNN/artifacts/`，记录 `tool_result_ref` event，返回 ref+preview |
| `core/context/manager.py` — 重写 `persist_tool_result()` | bounded 模式委托给 `block.persist_tool_result()`；无 active block 时降级走 scratchpad |
| `core/memento_s/phases/execution/runner.py` — 简化 tool result 处理 | 删除 28 行手动解析 `[artifact_ref:]` + `append_block_event` 逻辑，改为单行调用 `ctx.persist_tool_result()` |

**重构后数据流**:
```
改造前（双路径）:
  tool result ──┬── scratchpad.persist_tool_result() → artifacts_{session}/
                └── block.append_block_event()       → block/artifacts/ (compact时)

改造后（单一路径）:
  tool result ──── block.persist_tool_result()
                    ├── 短内容: event=tool_result, msg=内联
                    └── 长内容: 存 block/artifacts/, event=tool_result_ref, msg=ref+preview
```

---

## 九、注意事项

1. `bounded_prompt_enabled` 默认为 `True`，可通过 `ContextConfig` 关闭回退到 DB 全量历史模式
2. `core/context/block.py` 和 `runtime_state.py` 会在 `{context_dir}/sessions/{session_id}/` 下创建目录和文件
3. Artifact 文件统一存储在 `block_NNNN/artifacts/` 下（bounded 模式）；非 bounded 模式降级走 `artifacts_{session_id}/`
4. "Final Answer:" 协议对 LLM 输出格式有要求，需确保使用的模型能理解此指令
