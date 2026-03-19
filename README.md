# Memento-Skills

**Let Agents Design Agents** — An AI Agent framework built on a 4-phase architecture, with skill self-evolution, multi-step workflows, and hybrid retrieval.

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

<p align="center">
  <a href="#中文文档">🇨🇳 中文文档 / Chinese</a>
</p>

---

## Features

| Feature | Description |
|---------|-------------|
| **4-Phase ReAct Architecture** | Intent → Planning → ReAct Loop → Reflection — structured reasoning and execution |
| **Skill Self-Evolution** | Automatically diagnoses failures, generates shadow skills, and retries |
| **Multi-Step Workflows** | Chains multiple skills to accomplish complex tasks |
| **Hybrid Retrieval** | BM25 lexical search + sqlite-vec semantic vector search for precise skill routing |
| **Multi-LLM Support** | Unified access to Anthropic, OpenAI, Ollama, OpenRouter, vLLM, and more |
| **Smart Context Management** | Auto-compresses long conversations to avoid context overflow |
| **Sandboxed Execution** | uv-based isolated sandbox for safe skill script execution |
| **Dual Interface** | Flet desktop GUI + Typer CLI — choose your preferred interaction mode |

---

## Quick Start (5 min)

### Install

```bash
git clone https://github.com/Agent-on-the-Fly/Memento-S.git && cd Memento-S && python -m venv .venv && source .venv/bin/activate && pip install -e .
```

Windows:

```powershell
git clone https://github.com/Agent-on-the-Fly/Memento-S.git && cd Memento-S && python -m venv .venv && .venv\Scripts\activate && pip install -e .
```

After installation, `memento` and `memento-gui` commands are available in the active virtual environment.

### Configure

On first launch, `~/memento_s/config.json` is auto-generated. Edit it with your API key:

```jsonc
{
  "llm": {
    "active_profile": "default",
    "profiles": {
      "default": {
        "model": "your-provider/your-model",
        "api_key": "your-api-key",
        "base_url": "https://your-api-url/v1",
        "max_tokens": 8192,
        "temperature": 0.7,
        "timeout": 120
      }
    }
  },
  "env": {
    "TAVILY_API_KEY": "your-search-api-key"
  }
}
```

The `model` field uses `provider/model` format (litellm routing): `anthropic/claude-3.5-sonnet`, `openai/gpt-4o`, `ollama/llama3`, etc. `TAVILY_API_KEY` is for web search — leave empty if not needed.

### Launch

```bash
memento agent          # CLI interactive mode
memento-gui            # Desktop GUI
memento doctor         # Environment check
```

CLI startup screen:

```
╭─────────────────────────────────────╮
│  Memento-S  v0.1.0                  │
╰─────────────────────────────────────╯
  Workspace  /Users/you/memento_s
  Session    sess-xxxxxxxx
  Model      anthropic/claude-3.5-sonnet

Skill system ready (10 skills)

Interactive mode. Type exit or Ctrl+C to quit.

You ›
```

Exit: type `exit` / `quit` or press `Ctrl+C`.

---

## Supported LLM Providers

The `model` field uses `provider/model` format (litellm routing):

| Provider | model example | base_url |
|----------|--------------|----------|
| **Anthropic Claude** | `anthropic/claude-3.5-sonnet` | default |
| **OpenAI** | `openai/gpt-4o` | default |
| **OpenRouter** | `anthropic/claude-3.5-sonnet` | `https://openrouter.ai/api/v1` |
| **Ollama (local)** | `ollama/llama3` | `http://localhost:11434` |
| **Self-hosted (vLLM/SGLang)** | `openai/your-model` | custom endpoint |

---

## Built-in Skills

| Skill | Description |
|-------|-------------|
| `filesystem` | File system operations (read/write, search, directory management) |
| `web-search` | Web search (Serper / SerpAPI) and content scraping |
| `image-analysis` | Image analysis (VQA, OCR, caption generation) |
| `pdf` | PDF reading, form filling, merge, split, OCR |
| `docx` | Word document creation and editing |
| `xlsx` | Excel spreadsheet processing |
| `pptx` | PowerPoint creation and editing |
| `skill-creator` | LLM-driven automatic skill generation |
| `uv-pip-install` | Python package installation (via uv) |
| `im-platform` | IM platform integration (Feishu/Lark, etc.) |

---

## CLI Commands

```bash
memento agent             # Interactive agent session
memento agent -m "..."    # Single-message mode
memento doctor            # Environment check (deps, config, connectivity)
memento verify            # Skill verification
memento feishu            # Feishu IM bridge
```

---

## GUI Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Quit |
| `Ctrl+B` | Toggle sidebar |
| `Ctrl+T` | Toggle log panel |
| `Ctrl+L` | Clear chat |
| `ESC` | Abort current task |

### Chat Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/clear` | Clear chat history |
| `/context` | Show token usage |
| `/compress` | Force context compression |
| `/skills` | List available skills |

---

## Project Structure

```
memento-s/
├── core/                             # Core framework
│   ├── memento_s/                    # Agent orchestrator
│   │   ├── agent.py                  # MementoSAgent entry point
│   │   ├── phases/                   # 4-phase execution engine
│   │   │   ├── intent.py             #   Intent recognition
│   │   │   ├── planning.py           #   Plan generation
│   │   │   ├── react_loop.py         #   ReAct loop (multi-step reasoning)
│   │   │   └── reflection.py         #   Task reflection
│   │   ├── tool_dispatcher.py        # Tool routing & dispatch
│   │   └── stream_output.py          # Event stream output pipeline
│   ├── skill/                        # Skill framework
│   │   ├── gateway.py                # Unified skill gateway
│   │   ├── provider.py               # Skill discovery & loading
│   │   ├── retrieval/                # Skill retrieval (BM25 + vector)
│   │   ├── execution/                # Skill execution (sandboxed)
│   │   ├── store/                    # Skill persistence
│   │   └── embedding/                # Semantic embedding
│   ├── manager/                      # Session & context management
│   └── prompts/                      # Prompt templates
├── middleware/                       # Middleware layer
│   ├── config/                       # Config management (JSON + Pydantic)
│   ├── llm/                          # LLM client (litellm multi-provider)
│   └── storage/                      # Storage service (SQLite + SQLAlchemy)
├── gui/                              # Flet desktop GUI
│   ├── app.py                        # GUI entry point
│   ├── modules/                      # Feature controllers
│   ├── widgets/                      # Reusable UI components
│   └── i18n/                         # Internationalization
├── cli/                              # Typer CLI
│   ├── main.py                       # CLI entry point
│   └── commands/                     # Subcommands (agent, doctor, verify, feishu)
├── builtin/skills/                   # Built-in skill packages
├── bootstrap.py                      # App initialization
├── tests/                            # Test suite
└── pyproject.toml                    # Project configuration
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      User Interface                      │
│              ┌────────────┬────────────┐                 │
│              │  GUI (Flet) │  CLI (Typer)│                │
│              └────────────┴────────────┘                 │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│              MementoSAgent Orchestrator                   │
│  ┌───────────────────────────────────────────────────┐  │
│  │            4-Phase ReAct Architecture              │  │
│  │  Intent → Planning → ReAct Loop → Reflection      │  │
│  └───────────────────────────────────────────────────┘  │
│                       ▼                                  │
│                ToolDispatcher                             │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│                 SkillGateway                              │
│  ┌────────────┬─────────────┬────────────┐              │
│  │  Retrieval  │  Execution   │   Store    │             │
│  │ BM25+Vector │  Sandbox(uv) │ Persistence│             │
│  └────────────┴─────────────┴────────────┘              │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  Middleware Services                      │
│  ┌────────────┬─────────────┬────────────┐              │
│  │   Config    │  LLM Client  │  Storage   │             │
│  │JSON+Pydantic│   litellm    │  SQLite    │             │
│  └────────────┴─────────────┴────────────┘              │
└─────────────────────────────────────────────────────────┘
```

**Execution Flow:**

```
User Input → CLI/GUI → bootstrap → ConfigManager → LLMClient
    → MementoSAgent.run()
        → Intent (recognition)
        → Planning (generation)
        → ReAct Loop (skill retrieval → execution → observation → reasoning)
        → Reflection (summary)
    → Response Output
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **User Interface** | Flet (GUI), Typer + Rich (CLI) |
| **Agent Framework** | Custom 4-phase ReAct architecture |
| **LLM Access** | litellm (multi-provider unified) |
| **Skill Retrieval** | BM25 (jieba tokenizer) + sqlite-vec (semantic vector) |
| **Skill Execution** | uv sandbox + subprocess isolation |
| **Config Management** | JSON files + Pydantic validation + auto-migration |
| **Data Storage** | SQLite + aiosqlite (async) + SQLAlchemy ORM |
| **Async Runtime** | asyncio + aiofiles + anyio |
| **Build & Package** | hatchling / PyInstaller / Nuitka |
| **Testing** | pytest + pytest-asyncio |

---

## FAQ

| Problem | Solution |
|---------|----------|
| Skills not found | Check `skills` config in `~/memento_s/config.json` |
| API timeout | Increase the `timeout` value in the corresponding profile in `config.json` |
| Import errors | Make sure the virtual environment is activated and `pip install -e .` has been run |
| Browser skill fails | Run `uv run python -m playwright install chromium` |

---

## License

MIT

---

<a name="中文文档"></a>

## 中文文档

<details>
<summary><b>点击展开中文文档 / Click to expand Chinese documentation</b></summary>

### 简介

**Memento-S** — 基于 4 阶段架构的 AI Agent 框架，支持技能自演化、多步骤工作流与混合检索。

### 特性

| 特性 | 说明 |
|------|------|
| **4 阶段 ReAct 架构** | Intent → Planning → ReAct Loop → Reflection，结构化推理与执行 |
| **技能自演化** | 任务失败后自动诊断、生成影子技能并重试 |
| **多步骤工作流** | 串联多个技能完成复杂任务 |
| **混合检索** | BM25 词法搜索 + sqlite-vec 语义向量搜索，精准路由技能 |
| **多 LLM 支持** | 统一接入 Anthropic、OpenAI、Ollama、OpenRouter、vLLM 等 |
| **智能上下文管理** | 自动压缩长对话，避免上下文溢出 |
| **沙箱执行** | 基于 uv 的隔离沙箱，安全执行技能脚本 |
| **双端界面** | Flet 桌面 GUI + Typer CLI，灵活选择交互方式 |

### 快速开始

#### 安装

```bash
git clone https://github.com/Agent-on-the-Fly/Memento-S.git && cd Memento-S && python -m venv .venv && source .venv/bin/activate && pip install -e .
```

Windows：

```powershell
git clone https://github.com/Agent-on-the-Fly/Memento-S.git && cd Memento-S && python -m venv .venv && .venv\Scripts\activate && pip install -e .
```

安装完成后，`memento` 和 `memento-gui` 命令即在当前虚拟环境中可用。

#### 配置

首次运行时自动生成 `~/memento_s/config.json`，编辑填入 API Key 即可：

```jsonc
{
  "llm": {
    "active_profile": "default",
    "profiles": {
      "default": {
        "model": "your-provider/your-model",
        "api_key": "your-api-key",
        "base_url": "https://your-api-url/v1",
        "max_tokens": 8192,
        "temperature": 0.7,
        "timeout": 120
      }
    }
  },
  "env": {
    "TAVILY_API_KEY": "your-search-api-key"
  }
}
```

`model` 使用 `provider/model` 格式（litellm 路由）：`anthropic/claude-3.5-sonnet`、`openai/gpt-4o`、`ollama/llama3` 等。`TAVILY_API_KEY` 用于网页搜索，无需可留空。

**多 Profile 配置示例：**

```jsonc
{
  "llm": {
    "active_profile": "kimi-moonshot",
    "profiles": {
      "kimi-moonshot": {
        "model": "moonshot/moonshot-v1-128k",
        "api_key": "YOUR_KIMI_API_KEY",
        "base_url": "https://api.moonshot.cn/v1",
        "max_tokens": 8192,
        "temperature": 0.3
      },
      "openai-gpt4o": {
        "model": "openai/gpt-4o",
        "api_key": "YOUR_OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "max_tokens": 4096,
        "temperature": 0.7
      }
    }
  }
}
```

通过修改 `active_profile` 来切换不同的模型配置。

#### 启动

```bash
memento agent          # CLI 交互模式
memento agent -m "..." # 单轮对话模式
memento-gui            # 桌面 GUI
memento doctor         # 环境自检
memento verify         # 技能验证
memento feishu         # 飞书 IM 桥接
```

交互模式下输入 `exit`、`quit` 或按 `Ctrl+C` 退出。

### 常见问题

| 问题 | 解决方案 |
|------|----------|
| 找不到技能 | 检查 `~/memento_s/config.json` 中 `skills` 配置 |
| API 超时 | 增大 `config.json` 中对应 profile 的 `timeout` 值 |
| 导入错误 | 确认已激活虚拟环境并执行过 `pip install -e .` |
| 浏览器技能失败 | 运行 `uv run python -m playwright install chromium` |

</details>
