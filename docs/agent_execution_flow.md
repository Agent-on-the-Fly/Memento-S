# Memento-S Agent 完整执行流程分析

## 概述

本文档详细分析了一个完整的 Agent 任务执行流程，从用户输入到任务完成的整个过程。以"使用 pdf skill 创建 PDF 文件"为例，展示系统各组件的协作机制。

## 执行时间线

### Phase 1: 系统启动 (Bootstrap)

**时间点**: 20:32:03

```
bootstrap:bootstrap:430 - [bootstrap] phase 1: config system initialized
bootstrap:bootstrap:461 - [bootstrap] phase 3: database connection initialized
bootstrap:bootstrap:473 - [bootstrap] phase 7: syncing skills...
```

**关键调用**:
- `bootstrap:bootstrap()` - 主启动函数
- `SkillStore.__init__()` - 初始化技能存储
  - `load_all_skills()` - 从磁盘加载技能
  - 加载 9 个技能: docx, filesystem, image_analysis, pdf, pptx, skill_creator, uv_pip_install, web_search, xlsx

**代码位置**: `bootstrap.py:281-308`

```python
# Skill 同步流程
_sync_skills()
  ├── sync_builtin_skills()  # 同步内置技能
  ├── refresh_from_disk()    # 刷新磁盘技能
  ├── sync_all_to_db()       # 同步到数据库
  └── cleanup_orphaned_skills()  # 清理孤儿技能
```

### Phase 2: GUI 初始化

**时间点**: 20:32:04

```
gui.app:initialize:154 - [INIT] Step 1: Starting GUI initialization...
gui.app:initialize:194 - [INIT] Skill gateway created successfully
gui.app:initialize:203 - [INIT] Skill names: ['docx', 'filesystem', 'image_analysis', 'pdf', ...]
```

**关键调用**:
- `gui.app:initialize()` - GUI 初始化
- `SkillProvider.create_default()` - 创建 Skill Provider
- `UvLocalSandbox._setup_venv()` - 设置 Python 虚拟环境
  - `install_python_deps(['pip'])` - 安装 pip 工具

### Phase 3: 意图识别 (Intent Recognition)

**时间点**: 20:32:46

**用户输入**: "使用pdf, 写"hello"到ht.pfd"

```
middleware.llm.client:async_chat:429 - LLM async_chat: finish_reason=stop, tool_calls=0
```

**关键调用**:
- `core.memento_s.phases.intent:recognize_intent()`
- LLM 分析结果:
  ```json
  {
    "intent_type": "task",
    "complexity": "simple",
    "requires_planning": false,
    "mode": "react"
  }
  ```

**系统决策**: 进入 ReAct 模式执行

### Phase 4: ReAct 循环启动

**时间点**: 20:32:48

**关键调用**:
- `core.memento_s.phases.react_loop:run_react_loop()`
- `StatefulContextManager.assemble_messages()` - 组装上下文消息
  - 加载历史对话（5条消息）
  - 计算 token 预算: 69625
  - 组装系统提示词 + 历史 + 当前请求

**Prompt 组成**:
1. System Prompt (Agent Profile + Guidelines)
2. User History (5 turns)
3. Current Request
4. Available Tools: search_skill, execute_skill

### Phase 5: Skill 搜索 (First Tool Call)

**时间点**: 20:32:49

**LLM 决策**: 调用 `search_skill` 查找 pdf skill

```
🔧 TOOL CALL START: search_skill
  query: pdf
  k: 8
```

**调用链**:
```python
core.memento_s.tool_dispatcher:ToolDispatcher.dispatch()
  └── _search_skill(args={'query': 'pdf', 'k': 8})
      ├── gateway.discover()  # 获取本地技能
      ├── gateway.search(query, cloud_only=True)  # 云端搜索
      └── 返回: 9 local + 1 cloud skill
```

**搜索结果**:
```json
{
  "name": "pdf",
  "description": "Comprehensive PDF manipulation toolkit...",
  "source": "cloud",
  "execution_mode": "knowledge"
}
```

### Phase 6: Skill 执行 - 首次尝试

**时间点**: 20:32:50

**LLM 决策**: 调用 `execute_skill` 执行 pdf skill

```
🔧 TOOL CALL START: execute_skill
  skill_name: pdf
  args: {'request': '创建一个名为ht.pfd的PDF文件，内容为"hello"'}
```

**完整调用链**:

```python
core.memento_s.tool_dispatcher:ToolDispatcher._execute_skill()
  # 1. 参数提取
  skill_args = args.get("args", {})
  
  # 2. Gateway 调用
  await gateway.execute(skill_name="pdf", params=skill_args)
    
    # 3. Provider 层
    core.skill.provider:SkillProvider.execute()
      # 3.1 查找技能
      skill = await self._ensure_local_skill("pdf")
      
      # 3.2 依赖检查 (⚠️ 关键问题点)
      missing_deps = check_missing_dependencies(skill.dependencies)
      # 结果: skill.dependencies 为空，未检查出 reportlab 依赖
      
      # 3.3 执行器调用
      await SkillExecutor.execute(skill, query)
        
        # 4. Executor 层
        core.skill.execution.executor:SkillExecutor.execute()
          # 4.1 构建 Prompt
          prompt = self._build_prompt(skill, query)
          
          # 4.2 调用 LLM 生成代码
          response = await llm.async_chat(messages=[...], tools=BUILTIN_TOOL_SCHEMAS)
          
          # 4.3 LLM 生成 bash 命令
          # command: python << 'EOF'
          # from reportlab.lib.pagesizes import letter
          # from reportlab.pdfgen import canvas
          # ...
          
          # 4.4 执行 tool_calls
          await self._execute_with_tool_calls(skill, response.tool_calls)
            
            # 5. Tool 执行层
            for tc in tool_calls:
              tool_name = tc.name  # "bash"
              args = tc.arguments
              
              # 5.1 执行 bash
              await execute_builtin_tool("bash", args)
                
                # 6. Sandbox 层
                core.skill.execution.sandbox.uv:UvLocalSandbox.run()
                  # 6.1 确保 venv
                  self._ensure_uv_installed()
                  self._setup_venv()
                  
                  # 6.2 执行命令
                  bash_tool.execute(command)
```

**执行失败**:
```
Tool call done: #1 tool=bash
  result="EXIT CODE: 1
  STDERR:
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  ModuleNotFoundError: No module named 'reportlab'"
```

**根本原因分析**:
- `pdf` skill 的 `dependencies` 字段未声明 `reportlab`
- 执行时动态生成代码需要 `reportlab`，但未预装
- 运行时检查失败

### Phase 7: 反思与重试策略

**时间点**: 20:32:53

**ReAct 循环处理错误**:

```python
core.memento_s.phases.react_loop:_check_error_policy()
  # 分析错误类型: ModuleNotFoundError
  # 决策: 继续执行，让 LLM 处理
```

**Reflection Prompt**:
```
[Reflection] The PDF creation failed due to missing 'reportlab' module. 
The agent needs to either install the required module or use an alternative approach.
```

### Phase 8: 依赖安装

**时间点**: 20:32:56

**LLM 决策**: 调用 `uv_pip_install` skill 安装依赖

```
🔧 TOOL CALL START: execute_skill
  skill_name: uv_pip_install
  args: {'request': '安装 reportlab'}
```

**执行流程**:
```python
# uv_pip_install skill 执行
SkillExecutor.execute()
  └── LLM 生成安装命令
      └── bash: uv pip install reportlab
          └── UvLocalSandbox.install_python_deps(['reportlab'])
              └── 成功安装 ✓
```

**安装成功**:
```
Dependencies installed: ['reportlab']
Tool call done: #1 tool=bash result="SUCCESS: (No output)"
```

### Phase 9: 重试执行 - 成功

**时间点**: 20:33:05

**再次调用** `execute_skill` 执行 pdf skill

**执行成功**:
```
Tool call done: #1 tool=bash 
  result="STDOUT:
PDF created successfully: ht.pdf"
```

### Phase 10: 任务完成确认

**时间点**: 20:33:11

**ReAct 循环结束检查**:

```python
# Reflection 确认任务完成
{"decision": "finalize", 
 "reason": "The PDF skill executed successfully...",
 "completed_step_id": null}
```

**最终结果**:
- 文件位置: `/Users/manson/memento_s/workspace/ht.pdf`
- 文件内容: hello
- 执行时间: ~28 秒

## 关键组件交互图

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Input                               │
│              "使用pdf, 写"hello"到ht.pfd"                        │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Intent Recognition                              │
│                    Mode: react                                   │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ReAct Loop                                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │   Step 1    │───▶│   Step 2    │───▶│   Step 3    │          │
│  │search_skill │    │execute_skill│    │uv_pip_install│         │
│  └─────────────┘    └──────┬──────┘    └──────┬──────┘          │
│                            │                  │                 │
│                            ▼                  ▼                 │
│                      ┌─────────────┐    ┌─────────────┐          │
│                      │  Failed ❌  │───▶│  Success ✓  │          │
│                      │reportlab❌  │    │  Retry ✓   │          │
│                      └─────────────┘    └─────────────┘          │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Final Response                               │
│           PDF created successfully at ht.pdf                     │
└─────────────────────────────────────────────────────────────────┘
```

## 代码调用详细链路

### 1. Agent 层

```python
# core/memento_s/agent.py:MementoSAgent.reply_stream()
  └── core/memento_s/phases/intent.py:recognize_intent()
        └── Mode: react → run_react_loop()
              └── core/memento_s/phases/react_loop.py:run_react_loop()
```

### 2. Tool Dispatcher 层

```python
# core/memento_s/tool_dispatcher.py:ToolDispatcher
  ├── _search_skill() → gateway.search()
  └── _execute_skill() → gateway.execute()
```

### 3. Skill Gateway 层

```python
# core/skill/provider.py:SkillProvider
  ├── discover() → store.local_cache
  ├── search() → _multi_recall.recall()
  └── execute() → SkillExecutor.execute()
        ├── check_missing_dependencies()  # 预检查
        └── SkillExecutor.execute(skill, query)
```

### 4. Skill Executor 层

```python
# core/skill/execution/executor.py:SkillExecutor
  ├── execute(skill, query)
  │     ├── _build_prompt(skill, query)  # 构建执行提示词
  │     ├── llm.async_chat(tools=BUILTIN_TOOL_SCHEMAS)  # LLM 生成工具调用
  │     └── _execute_with_tool_calls(skill, tool_calls)
  │
  └── _execute_with_tool_calls(skill, tool_calls)
        └── for tc in tool_calls:
              execute_builtin_tool(tc.name, tc.arguments)
```

### 5. Sandbox 层

```python
# core/skill/execution/sandbox/uv.py:UvLocalSandbox
  ├── run(command)
  │     ├── _ensure_uv_installed()
  │     ├── _setup_venv()
  │     └── bash_tool.execute(command)
  │
  └── install_python_deps(deps)
        └── uv pip install <deps>
```

## 关键问题与改进点

### 1. 依赖声明缺失

**问题**: `pdf` skill 未在 `dependencies` 中声明 `reportlab`，导致预检查失败。

**日志证据**:
```
# SkillProvider.execute() 中
missing_deps = check_missing_dependencies(skill.dependencies)
# skill.dependencies = [] (空列表)
# 未检查出缺失的 reportlab
```

**改进建议**:
- 为 knowledge mode skill 自动检测代码中的 import 语句
- 或者在 SKILL.md 中显式声明所有可能的依赖

### 2. 依赖安装机制

**当前流程**:
1. 预检查（仅检查声明的依赖）
2. 执行（失败）
3. 运行时错误处理（LLM 决定安装）
4. 重试

**问题**: 第一次执行必然失败，用户体验差。

**改进建议**:
- 在 Skill 初始化时静态分析代码依赖
- 或在首次执行前自动安装常见依赖

### 3. 错误恢复机制

**优点**: ReAct 循环能自动识别 ModuleNotFoundError 并触发安装流程。

**代码位置**:
```python
# executor.py:361
missing = extract_missing_module_from_error(result.error or "")
if missing:
    # 返回友好错误提示
```

## 性能数据

| 阶段 | 耗时 | Token 使用 |
|------|------|-----------|
| 意图识别 | 2s | 489 |
| Skill 搜索 | 1s | 3064 |
| 首次执行 (失败) | 3s | 4479 |
| 依赖安装 | 2s | 2354 |
| 重试执行 (成功) | 3s | 4451 |
| **总计** | **~28s** | **~20000** |

## 总结

本次执行展示了 Memento-S 的完整协作流程：

1. **分层架构清晰**: Agent → ToolDispatcher → Gateway → Provider → Executor → Sandbox
2. **ReAct 循环强大**: 能自动处理错误、安装依赖、重试任务
3. **依赖管理待优化**: 需要更智能的依赖预检测机制
4. **用户体验良好**: 虽然第一次失败，但系统自动恢复并最终成功

关键成功因素:
- 详细的错误信息帮助 LLM 理解问题
- `uv_pip_install` skill 提供标准安装流程
- Reflection 机制确保任务最终完成