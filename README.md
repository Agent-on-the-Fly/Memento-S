# Memento-S: Self-Evolving Skills Runner

An intelligent agent system with self-evolving skills, multi-step workflows, and an interactive CLI.

**Repository:** https://github.com/Agent-on-the-Fly/Memento-S

---

## Features

- **Self-Evolving Skills** — Automatically optimizes skills based on task feedback via `memento-evolve`
- **Multi-Step Workflows** — Chains multiple skills to complete complex tasks
- **Interactive CLI** — Slash commands, tab completion, step streaming, and session history
- **Semantic Skill Routing** — BM25 + embedding-based skill discovery from a cloud catalog
- **Context Management** — Smart token compression for long conversations
- **Extensible Skills** — Drop-in skill packages (`SKILL.md` + scripts) for pdf, docx, pptx, xlsx, web-search, image-analysis, and more

---

## Quick Start

### One-line install

```bash
curl -sSL https://raw.githubusercontent.com/Agent-on-the-Fly/Memento-S/main/install.sh | bash
```

### Install from source

```bash
git clone https://github.com/Agent-on-the-Fly/Memento-S.git
cd Memento-S
chmod +x install.sh
./install.sh
```

The installer will:

1. Install `uv` if missing
2. Run `uv sync --python 3.12` to set up the virtual environment
3. Download skill catalog (`router_data/skills_catalog.jsonl`) from HuggingFace
4. Install browser dependencies (Playwright / crawl4ai) for web skills
5. Prompt for API keys (see below) and write `.env`
6. Create a global `memento` launcher command

### Interactive API configuration

During installation, the script will prompt you for the following values in order:

| # | Prompt | Required | Description |
|---|---|---|---|
| 1 | `OPENROUTER_API_KEY` | Yes | Your LLM API key (OpenRouter, Anthropic, or other provider). Input is hidden. |
| 2 | `OPENROUTER_MODEL` | Yes | Model identifier. Default: `anthropic/claude-3.5-sonnet` |
| 3 | `SERPAPI_API_KEY` | No | API key for web search skill. Press Enter to skip. Input is hidden. |

> If you already have a `.env` file, the installer will show existing values and let you press Enter to keep them.

### Run

```bash
memento                  # interactive CLI
memento -p "your query"  # single prompt
uv run python -m cli     # run directly without launcher
```

---

## Configuration

Create or edit `.env` in the project root. The installer handles this interactively, but you can also configure it manually.

### Anthropic (Claude API)

```env
LLM_API=anthropic
OPENROUTER_API_KEY=sk-ant-xxxxx
OPENROUTER_BASE_URL=https://api.anthropic.com
OPENROUTER_MODEL=claude-3-5-sonnet-20241022
OPENROUTER_MAX_TOKENS=100000
OPENROUTER_TIMEOUT=120
```

### OpenRouter

```env
LLM_API=openrouter
OPENROUTER_API_KEY=sk-or-xxxxx
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
OPENROUTER_MAX_TOKENS=100000
```

### Custom OpenAI-compatible API

```env
LLM_API=openai
OPENROUTER_API_KEY=your-api-key
OPENROUTER_BASE_URL=https://your-api-endpoint.com/v1
OPENROUTER_MODEL=your-model-name
```

### Context management

```env
CONTEXT_MAX_TOKENS=80000
CONTEXT_COMPRESS_THRESHOLD=60000
SUMMARY_MAX_TOKENS=2000
```

### Other settings

```env
SERPAPI_API_KEY=xxx              # optional, for web-search skill
SKILLS_DIR=./skills              # built-in skills directory
SKILLS_EXTRA_DIRS=./skill_extra  # user-added skills
WORKSPACE_DIR=./workspace        # agent working directory
```

---

## CLI Commands

### Slash commands (inside chat)

| Command | Description |
|---|---|
| `/help` | Show available commands |
| `/status` | Show session status |
| `/skills [query] [-n N]` | Search cloud skills or list local |
| `/config [show\|get\|set\|unset]` | View/update `.env` config |
| `/history [N]` | Show session history |
| `/history load <index>` | Load saved session into context |
| `/clear` | Clear conversation context |
| `/exit` | Exit the CLI |

### Keyboard shortcuts

| Key | Action |
|---|---|
| `Ctrl+C` | Interrupt current task |
| `Ctrl+D` | Exit CLI |

---

## Project Structure

```
Memento-S/
├── cli/                  # Interactive CLI (prompt_toolkit)
├── core/
│   ├── config.py         # Centralized .env configuration
│   ├── memento_agent.py  # LangGraph-based agent
│   ├── memento_client.py # LLM client abstraction
│   ├── memento_server.py # MCP skill server
│   ├── model_factory.py  # Model provider factory
│   ├── skill_engine/     # Skill routing & catalog
│   └── utils/            # Shared utilities
├── evolve/               # Self-evolution loop (memento-evolve)
│   ├── main.py           # CLI entry point
│   ├── dataset.py        # Benchmark data loading
│   ├── judge.py          # Answer evaluation
│   ├── feedback.py       # Failure analysis & tips
│   ├── optimizer.py      # Skill improvement
│   └── runner.py         # Task execution
├── skills/               # Built-in skill packages
│   ├── docx/             # Word document processing
│   ├── pdf/              # PDF reading & extraction
│   ├── pptx/             # PowerPoint processing
│   ├── xlsx/             # Excel processing
│   ├── web-search/       # Web search & content fetching
│   ├── image-analysis/   # Image analysis
│   ├── mcp-builder/      # MCP server builder
│   ├── skill-creator/    # Skill package creator
│   └── uv-pip-install/   # Package installation
├── skill_extra/          # User-added skills (gitignored)
├── router_data/          # Skill catalog & embeddings
├── workspace/            # Agent working directory
├── install.sh            # One-click installer
└── pyproject.toml        # Project metadata & dependencies
```

---

## Self-Evolution (`memento-evolve`)

The evolve loop runs benchmark tasks, evaluates answers, and automatically improves skills:

```bash
memento-evolve               # run evolution loop
uv run python -m evolve      # alternative
```

Pipeline: **Run task → Judge answer → Analyze failure → Generate feedback → Optimize skill → Re-test**

---

## Troubleshooting

| Issue | Solution |
|---|---|
| Skills not found | Check `SKILLS_DIR` and `SKILLS_EXTRA_DIRS` in `.env` |
| API timeouts | Increase `OPENROUTER_TIMEOUT` in `.env` |
| Import errors | Ensure `uv sync --python 3.12` completed successfully |
| Permission denied | Run `chmod +x install.sh` |
| Browser skills fail | Run `uv run python -m playwright install chromium` |

---

## License

MIT
