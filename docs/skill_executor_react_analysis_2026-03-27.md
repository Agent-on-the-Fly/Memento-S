# Memento-S Skill Executor ReAct 过程深度分析

> 日志来源: `/Users/manson/memento_s/logs/app_2026-03-27.log`
> 分析日期: 2026-03-27

## 一、整体架构概览

- **用户任务**: "生成一个关于知识产权的普法PPT，受众是技术人员，包含真实案例"
- **总耗时**: 706.53秒 (约11分47秒)
- **模型**: openai/KIMI-K2.5 (context_window=100000, max_tokens=8192)
- **执行模式**: agentic (由 Intent 分析确定)
- **Plan**: 2 步 (web_search -> pptx)
- **Session ID**: a3c79633-efe9-40eb-94b1-6dbe63e592b2

## 二、ReAct 过程完整流程

### Phase 0: Intent 识别 (15:27:34 - 15:27:36)

```
Input:  用户消息
Output: mode=agentic, task=生成一个关于知识产权的普法PPT
```

LLM 判断为 agentic 模式，正确识别了这是一个需要多步骤执行的复杂任务。

### Phase 1: Plan 生成 (15:27:36 - 15:27:43, ~7s)

LLM 生成了 **2 步计划**:

| Step | Action | Skill | 依赖 |
|------|--------|-------|------|
| 1 | 搜索技术人员相关知识产权真实案例 | `web_search` | 无 |
| 2 | 创建知识产权普法PPT文件 | `pptx` | Step 1 |

---

## 三、Skill 1: `web_search` 执行详情

**时间范围**: 15:27:45 - 15:29:49 (约2分钟)

### 迭代状态更新过程

| 迭代 | 时间 | Action | Tool | 输入 | 输出/结果 |
|------|------|--------|------|------|-----------|
| 1 | 15:27:45 | Agent 调用 execute_skill | - | skill_name="web_search" | 进入 skill executor |
| 2 | 15:27:46 | update_scratchpad | - | 搜索策略规划 | 记录分4类搜索 |
| 3 | 15:27:48 | **search_web x2** (并行) | search_web | "软件著作权侵权案例..." / "专利纠纷案例..." | 上海知产法院900万案 / 近5年判赔过亿案件 |
| 4 | 15:27:56 | **search_web x2** (并行) | search_web | "商业秘密泄露案例..." / "开源协议违规案例..." | 员工跳槽案 / GPL适用研究 |
| 5 | 15:28:06 | **search_web x2** (并行) | search_web | "侵犯商业秘密罪 程序员..." / "程序员跳槽带走源代码..." | 游戏源代码案 / 员工跳槽侵权 |
| 6 | 15:28:16 | **fetch_webpage x2** (并行) | fetch_webpage | chinanews.com / zhichanli.com | 获取案例详情 |
| 7 | 15:28:19 | **search_web x2** (并行) | search_web | "侵犯商业秘密罪 程序员 2022-2024" / "GPL开源协议 侵权案例" | 重复获取 + GPL判决书 |
| 8 | 15:28:28 | **fetch_webpage + search_web** (并行) | fetch_webpage, search_web | ciplawyer.cn / "网经科技 启奥 GPL" | GPL二审判决详情 |
| 9 | 15:28:37 | **search_web x2** (并行) | search_web | "网经科技 GPL 赔偿50万" / "小米 专利侵权" | 搜狐转载 / 小米商标案 |
| 10 | 15:28:45 | **fetch_webpage + search_web** (并行) | fetch_webpage, search_web | sohu.com / "华为 专利侵权" | 最高法判决 / 华为三星8000万案 |
| 11 | 15:28:51 | **search_web x2** (并行) | search_web | "腾讯 程序员 泄露源代码" / "百度 专利侵权" | 游戏源代码案 / 百度商标案500万 |
| 12 | 15:29:03 | **fetch_webpage + search_web** (并行) | fetch_webpage, search_web | qq.com / "阿里巴巴 软件著作权" | 程序员判刑详情 / 深圳知产案 |
| 13 | 15:29:09 | LLM 整理汇总 | - | 全部搜索结果 | 2169字符案例汇总 |
| 14 | 15:29:44 | **python_repl** | python_repl | 写入 md 文件 | 创建"技术人员知识产权案例汇总.md" (5048 bytes) |
| 15 | 15:29:49 | 最终总结 | - | - | 输出执行结果 + 案例汇总表 |

### Tool 调用统计

- `search_web`: 16次 (8次并行对)
- `fetch_webpage`: 5次
- `python_repl`: 1次
- `update_scratchpad`: 1次
- **全部工具调用均成功**

---

## 四、Skill 2: `pptx` 执行详情

**时间范围**: 15:29:53 - 15:39:07 (约9分14秒)

### 迭代状态更新过程 (含大量错误重试)

| 迭代 | 时间 | Action | 结果 | 错误原因 |
|------|------|--------|------|----------|
| 1 | 15:29:57 | update_scratchpad | OK | 记录PPT页面结构 |
| 2 | 15:29:58 | read_file (案例汇总.md) | OK | 读取数据 |
| 3 | 15:29:59 | read_file (重复读取) | OK | **冗余操作** |
| 4 | 15:30:00 | bash: cat 文件 | OK | 内容截断 |
| 5 | 15:30:02 | bash: `cat "file" \| wc -l` | **FAIL** | 管道符被当作文件名 |
| 6 | 15:30:14 | python_repl: 创建封面 (第1页) | **OK** | 含完整初始化代码 |
| 7 | 15:30:22 | python_repl: 第2页目录 | **FAIL** | `NameError: prs is not defined` (无状态) |
| 8 | 15:30:34 | python_repl: 第1-2页完整代码 | **OK** | LLM 意识到需要完整代码 |
| 9 | 15:30:42 | python_repl: 第2页单独 | **FAIL** | `NameError: prs is not defined` (又忘了) |
| 10 | 15:30:56 | python_repl: 第1-2页完整 | **OK** | 再次重新包含初始化 |
| 11 | 15:31:07 | python_repl: 第3页单独 | **FAIL** | `NameError: prs is not defined` |
| 12 | 15:31:30 | python_repl: 第1-3页完整 | **OK** | 包含初始化 |
| 13 | 15:31:40 | python_repl: 第4页单独 | **FAIL** | `NameError: prs is not defined` + SyntaxError (中文引号) |
| 14 | 15:32:10 | python_repl: 第1-4页完整 | **OK** | 修复引号问题 |
| 15 | 15:32:26 | python_repl: 第5-7页 | **FAIL** | `NameError: prs is not defined` |
| 16 | 15:33:06 | python_repl: 第1-5页完整 | **OK** | |
| 17 | 15:33:27 | python_repl: 第6-7页 | **FAIL** | `NameError: prs is not defined` |
| 18 | 15:34:19 | python_repl: 第1-7页完整 | **OK** | |
| 19 | 15:34:48 | python_repl: 第8-10页 | **FAIL** | `NameError: prs is not defined` |
| 20 | 15:35:45 | python_repl: 第1-7页完整 | **OK** | 但只到7页 |
| 21 | 15:36:16 | python_repl: 第8-10页 | **FAIL** | `NameError: prs is not defined` |
| 22 | 15:37:39 | file_create: create_ppt.py | **OK** | LLM 终于改用文件策略 |
| 23 | 15:37:49 | bash: python create_ppt.py | **OK** | 但 bash 在沙盒中行为异常 |
| 24 | 15:38:00 | bash: ls *.pptx | **FAIL** | 通配符被转义 |
| 25 | 15:38:10 | python_repl: `exec(open().read())` | **OK** | 终于成功生成10页PPT |
| 26 | 15:38:20 | bash: ls *.pptx | **FAIL** | 通配符仍被转义 |
| 27 | 15:38:30 | list_dir | **OK** | 确认文件存在 |
| 28 | 15:38:41 | bash: markitdown | **FAIL** | 模块未安装 |
| 29 | 15:38:53 | python_repl: 验证PPT | **OK** | 确认10页内容正确 |
| 30 | 15:39:07 | 最终总结 | **OK** | 输出结果 |

### python_repl 调用统计

- 总调用: ~20次
- 成功: ~10次
- 失败: ~10次
- **失败率: 约50%**

### 错误模式分析

```
NameError: name 'prs' is not defined    出现 8+ 次
SyntaxError: 中文引号冲突               出现 1 次
bash 管道/通配符转义错误                  出现 3 次
模块未安装 (markitdown)                  出现 1 次
```

---

## 五、过程有效性分析

### 有效的方面

1. **Plan 设计合理**: 2步计划 (搜索 -> 生成PPT) 逻辑清晰，依赖关系正确
2. **web_search 执行高效**: 充分利用并行搜索（每次2个查询并行），搜索策略从广到深，覆盖4个维度
3. **Tool Bridge 错误分类准确**: 每次工具执行都有清晰的 `basis` 分类，`state_reason` 准确
4. **Scratchpad 使用得当**: 在执行前先记录策略，帮助 LLM 保持上下文
5. **最终产物质量可接受**: PPT 最终生成了10页，结构完整，包含真实案例

### 严重低效的方面

#### 问题1: Stateless 沙盒问题 (核心瓶颈)

- LLM **反复犯同样的错误**: `NameError: prs is not defined` 出现了 **8次以上**
- 每次都试图增量添加幻灯片，每次都因为环境无状态而失败
- LLM 虽多次说"环境是stateless的"，但仍不断重蹈覆辙
- 这是 **executor/sandbox 未提供持久化 session** 和 **LLM prompt 中未明确告知 stateless 约束** 的双重问题

#### 问题2: 代码膨胀问题

- 每成功一步，就必须在下次调用中重复所有之前的代码
- 第1-7页的完整代码已经很长，导致 LLM 无法在单次调用中完成全部10页
- 最终靠写入 `.py` 文件 + `exec(open().read())` 才解决

#### 问题3: Bash 工具参数转义 Bug

- `cat "file" | wc -l` 中管道 `|` 被当作文件名参数
- `ls *.pptx` 中 `*` 通配符被引号转义为字面字符
- 说明 `bash` 工具的参数转义逻辑对特殊字符处理有缺陷

#### 问题4: 冗余操作

- 同一个 `.md` 文件被 `read_file` 读取2次，又被 `bash cat` 读取1次
- 搜索后期的一些查询返回了已搜过的相同结果（如程序员泄露源代码案重复出现3次）

#### 问题5: Token 浪费严重

- pptx skill 最终的 `prompt_tokens` 达到 **48,119** tokens
- 大量 token 被用于重复的错误信息和重复的完整代码

---

## 六、对于多样性复杂任务的优化建议

### 1. 沙盒状态持久化 (最高优先级)

**问题**: python_repl 每次调用都是独立进程，变量无法跨调用保留

**方案**:
- A. 提供 stateful session (如 IPython kernel / Jupyter kernel)
- B. 在 executor 层自动拼接上下文代码（维护一个累积的代码 buffer）
- C. 至少在 system prompt 中明确告知 "每次代码执行是独立的，必须包含所有 import 和初始化代码"

### 2. 大型代码任务的分治策略

**问题**: 10页PPT代码太长，单次调用放不下

**方案**:
- A. 支持"代码追加模式": 将代码写入文件，逐段 append，最后执行
- B. skill executor 自动检测代码生成类任务，提前切换为 file-based 模式
- C. 在 pptx skill 的 prompt 中预置 "先 write_file 再 exec" 的最佳实践

### 3. 错误重试的智能降级

**问题**: 同一个 NameError 重复出现8次，LLM 无法跳出循环

**方案**:
- A. 连续2次相同错误后，executor 强制注入 "环境是 stateless 的" 提示
- B. 实现 error_pattern_detector：检测重复错误模式后升级策略
- C. 设置同类错误最大重试次数（如3次后强制切换策略）

### 4. Bash 工具参数处理修复

**问题**: 管道符和通配符被错误处理

**方案**:
- A. 对 bash 命令不做参数级转义，直接传入 shell
- B. 或使用 `shell=True` 模式执行
- C. 当前 `resolved_args` 中 `'|'` 被单引号包裹是明显的 bug

### 5. 搜索去重与质量控制

**问题**: 多次搜索返回相同结果，且部分查询偏离主题（如小米商标案非技术领域）

**方案**:
- A. 在 scratchpad 中维护已获取的案例列表，避免重复搜索
- B. 添加 search_result_dedup 中间件
- C. 设置搜索轮次上限（如4轮后停止）

### 6. Skill 间数据传递优化

**问题**: web_search 的输出只以自然语言形式传给 pptx skill，信息损失大

**方案**:
- A. Step 1 的 artifact (md 文件路径) 应自动注入 Step 2 的上下文
- B. 支持结构化的 `input_from` 数据传递，而不仅是上下文拼接
- C. `PRIMARY_ARTIFACT_PATH` 环境变量取值为 `None`，说明产物链路断裂

---

## 七、效率指标对比

| 指标 | 当前值 | 优化目标 |
|------|--------|----------|
| 总耗时 | 706s | < 180s |
| pptx skill 失败次数 | ~10次 | < 2次 |
| web_search 冗余搜索 | ~4次 | 0次 |
| Token 消耗 (pptx) | 48K | < 15K |
| python_repl 成功率 | ~50% | > 90% |

---

## 八、结论

当前系统的最大效率瓶颈不在 Plan 和 Search 环节（这两个环节工作得相当好），而在于 **skill executor 的 stateless sandbox 与 LLM 增量代码生成模式之间的根本性不匹配**。

LLM 天然倾向于增量式编写代码（先写第1页，再写第2页...），但 stateless sandbox 要求每次调用都包含完整代码。这种架构性矛盾导致了：
- 大量重复的 NameError 错误
- 代码随页数增长而膨胀
- 最终不得不通过写文件 + exec 的 workaround 解决

解决这个架构级问题后，此类任务的效率预计可提升 **3-5 倍**。
