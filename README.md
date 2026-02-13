# 🧠 Memento-S: Self-Evolving Skills Runner

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![CLI](https://img.shields.io/badge/Interface-CLI-blue?style=for-the-badge)
[![Website](https://img.shields.io/badge/Website-skills.memento.run-0ea5e9?style=for-the-badge)](https://skills.memento.run/)

**An intelligent agent system with self-evolving skills, multi-step workflows, and a focused CLI interface.**

[Quick Start](#-quick-install) • [Features](#-features) • [Configuration](#-configuration) • [Usage](#-usage)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔄 **Self-Evolving Skills** | Automatically optimizes skills based on task failures |
| 🎯 **Multi-Step Workflows** | Chains multiple skills to complete complex tasks |
| 💻 **Focused CLI** | Command-first workflow with slash commands and step streaming |
| 🧠 **Context Management** | Smart compression to handle long conversations |
| 🔌 **Extensible Skills** | Easy to create and install custom skills |

---

## 🚀 Quick Install

Repository: https://github.com/Agent-on-the-Fly/Memento-S

### One-line install (curl + sh)

```bash
curl -sSL https://raw.githubusercontent.com/Agent-on-the-Fly/Memento-S/main/install.sh | bash
```

### Recommended (inside repo): `install.sh` one-click install

```bash
git clone https://github.com/Agent-on-the-Fly/Memento-S.git
cd Memento-S
chmod +x install.sh
./install.sh
```

### What `install.sh` does

- Clones or updates the repository
- Installs `uv` (if missing)
- Runs `uv sync --python 3.12`
- Downloads router assets to `./router_data`:
  - Dataset index (`skills_catalog.jsonl`): https://huggingface.co/datasets/AgentFly/router-data/blob/main/skills_catalog.jsonl
  - Optional embeddings (`embeddings/`) are not downloaded by default
- Installs optional browser dependencies
- Creates `memento` launcher

```bash
# Start CLI
memento

# Or run directly
uv run python -m cli
```

---

## ⚙️ Configuration

Create a `.env` file in the project root:

### 🤖 Anthropic (Claude API)

```env
LLM_API=anthropic
OPENROUTER_API_KEY=sk-ant-xxxxx
OPENROUTER_BASE_URL=https://api.anthropic.com
OPENROUTER_MODEL=claude-3-5-sonnet-20241022
OPENROUTER_MAX_TOKENS=100000
OPENROUTER_TIMEOUT=120
```

### 🌐 OpenRouter

```env
LLM_API=openrouter
OPENROUTER_API_KEY=sk-or-xxxxx
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=anthropic/claude-3.5-sonnet
OPENROUTER_MAX_TOKENS=100000
```

### 🔧 Custom OpenAI-compatible API

```env
LLM_API=openai
OPENROUTER_API_KEY=your-api-key
OPENROUTER_BASE_URL=https://your-api-endpoint.com/v1
OPENROUTER_MODEL=your-model-name
```

### 📊 Context Management

```env
CONTEXT_MAX_TOKENS=80000
CONTEXT_COMPRESS_THRESHOLD=60000
SUMMARY_MAX_TOKENS=2000
```

---

## 🎮 Usage

### CLI Commands

```bash
# 🖥️ Launch interactive CLI
python -m cli

# 💬 Single-turn mode
python -m cli "Hello, how are you?"
```

### ⌨️ CLI Controls

| Key | Action |
|-----|--------|
| `Ctrl+C` | 🛑 Interrupt current task |
| `Ctrl+D` | 🚪 Exit CLI |

### 💬 In-Chat Commands

| Command | Description |
|---------|-------------|
| `/help` | 📖 Show available commands |
| `/clear` | 🧹 Clear chat history |
| `/status` | 📊 Show session/runtime status |
| `/retry` | 🔁 Retry previous user request |
| `/last` | 🧠 Show last assistant reply |
| `/skills <query>` | ☁️ Search cloud skills |
| `/skills local` | 🧩 List local skills |

---

## ❓ Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| 🔍 Skills not found | Run `openskills sync` |
| ⏱️ API timeouts | Increase `OPENROUTER_TIMEOUT` in `.env` |
| 📦 Import errors | Ensure virtual environment is activated |
| 🔒 Permission denied | Run `chmod +x install.sh` |


---

## 📜 License

MIT

---

<div align="center">

**Made with ❤️ by the Memento-S Team**

[⬆ Back to Top](#-memento-s-self-evolving-skills-runner)

</div>
