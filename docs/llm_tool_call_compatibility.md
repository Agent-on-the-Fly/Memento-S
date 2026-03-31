# LLM Tool Call 与消息格式兼容性指南

> 本文档记录 Memento-S 在多模型适配过程中遇到的 tool call 格式差异、消息序列约束，以及对应的防御策略。
> 涉及模型：DeepSeek、Gemini、MiniMax、StepFun、KIMI、Qwen、Mistral 等。

---

## 1. 问题总览

不同 LLM 在 tool calling 上存在三类兼容性问题：

| 类别 | 表现 | 涉及模型 |
|------|------|----------|
| **Raw Token 输出** | 模型不走标准 function calling，直接在 content 中输出 XML/JSON/控制 token | DeepSeek, StepFun, MiniMax, KIMI, Qwen |
| **消息序列约束** | API 对 assistant/tool 消息的顺序、内容有严格校验 | DeepSeek, Gemini |
| **字段格式差异** | tool_call_id 为空、arguments 格式不一致等 | MiniMax, 部分 OpenRouter 模型 |

---

## 2. Raw Tool Call Token 格式

模型在 content 中输出的非标准 tool call 格式，需要检测并解析。

### 2.1 已知格式

| 格式 | 示例 | 来源模型 |
|------|------|----------|
| DeepSeek XML | `<execute_skill><skill_name>filesystem</skill_name><parameters><action>list</action></parameters></execute_skill>` | DeepSeek |
| Qwen/ChatGLM | `<tool_call>{"name":"xxx","arguments":{...}}</tool_call>` | Qwen, ChatGLM |
| MiniMax 命名空间 XML | `<minimax:tool_call>...</minimax:tool_call>` | MiniMax M2.5/M2.7 |
| MiniMax Bracket | `[TOOL_CALL]{"name":"xxx",...}[/TOOL_CALL]` | MiniMax M2.7 |
| Anthropic-style XML | `<function_calls><invoke name="xxx"><parameter name="key">val</parameter></invoke></function_calls>` | MiniMax, 部分 proxy |
| StepFun JSON | `{"tool":"xxx","parameters":{...}}` | StepFun step-3.5-flash |
| KIMI 函数引用 | `functions.xxx:1 <\|tool_call_argument_begin\|>{...}<\|tool_call_argument_end\|>` | KIMI K2.5 |
| 通用控制 Token | `<\|tool_calls_section_begin\|>...<\|tool_calls_section_end\|>` | DeepSeek, KIMI |
| GPT function tag | `<function=name>{...}</function>` | GPT 系列 |
| Qwen Legacy | `✿FUNCTION✿...✿RESULT✿` | Qwen 旧版 |

### 2.2 防御策略（三层防线）

```
Stream 层 (async_stream_chat)
  ↓ looks_like_tool_call_text() 检测 → 抑制 delta_content
  ↓ fallback → async_chat

解析层 (_build_response_from_raw)
  ↓ _fallback_extract_tool_calls() → Strategy 3: _parse_raw_content_tool_calls()
  ↓ 从 content 提取结构化 ToolCall

清理层 (sanitize_content)
  ↓ 多级正则：完整块 → 单条格式 → 标签 → 控制 token
  ↓ 确保最终用户输出不含任何 raw token
```

**关键文件**：
- `middleware/llm/utils.py` — 检测（`looks_like_tool_call_text`）与清理（`sanitize_content`）
- `middleware/llm/llm_client.py` — 解析（`_parse_raw_content_tool_calls`）与流式防御

### 2.3 添加新格式的检查清单

1. 在 `utils.py` 中添加 **检测正则**（用于 `looks_like_tool_call_text`）
2. 在 `utils.py` 中添加 **清理正则**（用于 `sanitize_content`，先块级后标签级）
3. 在 `llm_client.py:_parse_raw_content_tool_calls` 中添加 **解析逻辑**（提取为 `ToolCall` 对象）
4. 验证流式检测（`async_stream_chat` 中 `_raw_tokens_detected` 能触发）

---

## 3. 消息序列约束

### 3.1 DeepSeek

| 约束 | 错误信息 | 触发场景 |
|------|----------|----------|
| assistant 必须有 content 或 tool_calls | `Invalid assistant message: content or tool_calls must be set` | 空 assistant 消息（content="" 且无 tool_calls） |
| tool_calls 后必须跟对应 tool 响应 | `tool_call_ids did not have response messages: xxx` | assistant 有 tool_calls 但后续缺少匹配的 tool 消息 |

### 3.2 Gemini

| 约束 | 错误信息 | 触发场景 |
|------|----------|----------|
| 每个 message part 必须有初始化的 data 字段 | `required oneof field 'data' must have one initialized field` | content 为空字符串 "" 时生成了空 text part |
| function call 必须紧跟 user 或 function response | `function call turn comes immediately after a user turn or after a function response turn` | assistant(tool_calls) 前面不是 user/tool 消息 |

### 3.3 防御策略

**`_fix_empty_messages`**（`llm_client.py`）：
- 空 assistant（无 content、无 tool_calls）+ 后续 tool 消息 → 合并为 `user: [Tool Result]`
- assistant 有 tool_calls 但 content 为空字符串 → 设 `content=None`

**`not tools` 转换块**（`_build_completion_kwargs`）：
- 当 `tools=None`（finalize 等场景），会剥掉 assistant 的 `tool_calls` 字段
- 剥掉后如果 assistant 变成空的 → **必须 drop**，否则 DeepSeek 会拒绝
- tool 消息 → 转为 `user: [Tool Result]: ...`

**`_normalize_messages`**：
- `content: None` → `content: ""`（统一格式）
- 非首位 `system` 消息 → 转为 `user: [System]: ...`

### 3.4 消息处理管线

```
原始 state.messages
  ↓ _normalize_messages()     — 统一 content 格式，system 转 user
  ↓ _fix_empty_messages()     — 处理空 assistant + 孤儿 tool
  ↓ not-tools 转换（如适用）  — 剥掉 tool_calls，转 tool→user，drop 空 assistant
  ↓ 最终发送给 LLM API
```

⚠️ **顺序很重要**：`_fix_empty_messages` 在 `not-tools` 转换之前运行，但 `not-tools` 转换会引入新的空 assistant。两个阶段都需要处理。

---

## 4. 字段格式差异

### 4.1 tool_call_id 为空

**模型**：MiniMax M2.7、部分 OpenRouter 模型

**表现**：LLM 返回的 tool_call 中 `id` 字段为空字符串，导致下游 `tool_call_id` 匹配失败。

**修复**：在三个位置生成合成 ID（`tc_{uuid4_hex}`）：
- `_parse_tool_calls` — 标准解析路径
- `async_stream_chat` — 流式 tool call 组装
- `_fallback_extract_tool_calls` — fallback 解析

### 4.2 arguments 格式

**表现**：`arguments` 可能是 JSON 字符串、dict、None、空字符串。

**修复**：`_parse_tool_args_with_repair` 提供多层修复：
- 缺少首尾括号 → 补全
- 单引号 → 双引号
- 尾逗号 → 移除

---

## 5. Intent 分类与 Tool Call 冲突

### 5.1 问题

Intent 分类器将请求标记为 `direct`（纯文本回答），但模型在回答时输出了 raw tool call（想用工具）。

**根因**："请告诉我你的workspace目录" 被理解为 "SAY something"（知识问答），但正确答案需要 "DO something"（文件系统操作）。

### 5.2 防御

1. **Intent Prompt 强化**：Decision Rule 3 明确 — 如果正确答案依赖实时动态状态（无法从静态知识得到），就是 `agentic`。模棱两可时偏向 `agentic`。

2. **Finalize 安全网**：当 `got_tool_calls=True` 且无有效内容时：
   - 跳过无用的 plain-text retry（避免浪费 API 调用）
   - 给用户清晰提示而非内部错误信息

---

## 6. 事件系统与 tool_call_id 一致性

### 6.1 问题

Block event 系统记录 `tool_call` 和 `tool_result`/`tool_result_ref` 事件时，如果未显式传入 `tool_call_id`，`_events_to_messages` 重建消息时会回退到 `event_id`（自增 ID，如 `e0005`）。由于 tool_call 和 tool_result 是分别记录的独立事件，它们的 `event_id` 不同，导致：

- assistant 消息的 `tool_calls[].id` = `e0005`
- tool 消息的 `tool_call_id` = `e0006`
- DeepSeek 校验失败：`tool_call_ids did not have response messages: e0005`

### 6.2 修复

在所有记录 tool_call/tool_result 事件的位置通过 `extra={"tool_call_id": ...}` 传入真实 ID：

| 位置 | 事件类型 | 传入的 ID |
|------|----------|-----------|
| `runner.py` 记录 tool_call | `tool_call` | `sc.id`（LLM 返回的原始 ID） |
| `block.py:persist_tool_result` 短结果 | `tool_result` | `tool_call_id` 参数 |
| `block.py:persist_tool_result` 写入失败降级 | `tool_result` | `tool_call_id` 参数 |
| `block.py:persist_tool_result` artifact ref | `tool_result_ref` | `tool_call_id` 参数 |

### 6.3 `_events_to_messages` 的 ID 回退链

```python
# tool_call 事件 → assistant 消息的 tool_calls[].id
"id": ev.get("tool_call_id", ev.get("event_id", ""))

# tool_result/tool_result_ref 事件 → tool 消息的 tool_call_id
"tool_call_id": ev.get("tool_call_id", ev.get("event_id", ""))
```

当 `tool_call_id` 存在时，两边使用相同的真实 ID；缺失时各自用不同的 `event_id`，导致不匹配。

---

## 7. 常见错误速查

| 错误信息 | 根因 | 修复位置 |
|----------|------|----------|
| `Invalid assistant message: content or tool_calls must be set` | 空 assistant 消息 | `_fix_empty_messages` + `not-tools` drop |
| `tool_call_ids did not have response messages` | 事件记录缺少 tool_call_id，重建消息时 ID 不匹配 | `runner.py` + `block.py:persist_tool_result` 传入 `extra={"tool_call_id": ...}` |
| `required oneof field 'data'` | 空 content 生成了空 text part (Gemini) | `_fix_empty_messages` 设 content=None |
| `function call turn comes immediately after a user turn` | 消息顺序不符合 Gemini 要求 | `_build_messages` 确保 user 消息在 assistant 前 |
| `tool call id is empty` | MiniMax 返回空 tool_call_id | 生成合成 `tc_{uuid}` |
| `[FINALIZE] TEXT MISMATCH` | finalize API 调用失败，fallback 内容与已流式输出不一致 | 修复上游 API 错误 |

---

## 8. 测试矩阵

验证兼容性修改时，应覆盖以下模型和场景：

| 场景 | 验证点 |
|------|--------|
| Direct 模式 + 无 tools | 模型不输出 raw token，或输出后被正确清理 |
| Agentic 模式 + 标准 tool call | tool call 正确解析和执行 |
| Agentic 模式 + raw XML tool call | 从 content 解析出 ToolCall 并执行 |
| Finalize 阶段 | 消息序列合法，无空 assistant |
| Reflection 阶段 | 同上 |
| 多轮对话 | 历史消息中的 tool call 记录不破坏序列 |
| 流式输出 | raw token 在流式中被及时检测和抑制 |
