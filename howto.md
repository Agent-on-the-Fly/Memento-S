# Memento-S 快速上手指南

本文档提供在本地从源码运行 Memento-S 的说明，主要包括 **GUI (图形用户界面)** 和 **CLI (命令行界面)** 两种模式。

---

## 1. 安装

```bash
git clone https://github.com/Agent-on-the-Fly/Memento-S.git && cd Memento-S && python -m venv .venv && source .venv/bin/activate && pip install -e .
```

安装完成后 `memento` 和 `memento-gui` 命令即可使用。

---

## 2. 环境与配置

### 核心配置文件

Memento-S 的所有配置都通过 JSON 文件管理，而非 `.env` 文件或环境变量。核心配置文件位于：

`~/memento_s/config.json`

当应用首次启动时，如果该文件不存在，它会根据内置的模板自动创建。您所有的自定义设置都应在此文件中完成。

### 配置大语言模型 (LLM)

这是运行 Agent 前 **必须** 配置的部分。您需要修改 `~/memento_s/config.json` 文件中的 `llm` 部分。

1.  **`active_profile`**: 设置您想要激活的配置方案名称。
2.  **`profiles`**: 一个包含多种配置方案的字典。您可以定义多个 profile，并通过 `active_profile` 来切换。

**配置示例：**

以下是一个 `config.json` 的示例，展示了如何配置 Kimi 和 OpenAI 两个模型：

```json
{
  // ... 其他配置项
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
  // ... 其他配置项
}
```

**重要提示**: 请将 `YOUR_KIMI_API_KEY` 和 `YOUR_OPENAI_API_KEY` 替换为您自己的 API Key。

---

## 3. 运行方式

确保你的 Python 虚拟环境（如 `.venv`）已被激活。

### 方式一：GUI (图形用户界面)

```bash
memento-gui
```

### 方式二：CLI (命令行界面)

通过命令行与 Agent 进行交互。

*   **交互模式**:
    启动后，你将看到 `You ›` 提示符，可以开始对话。输入 `exit`、`quit` 或按 `Ctrl+C` 退出。
    ```bash
    # 通过 pip install -e . 安装后
    memento agent

    # 或直接运行
    python cli/main.py agent
    ```

*   **单轮对话模式**:
    通过 `-m` 或 `--message` 参数发送单条消息，程序将在收到回复后自动退出。
    ```bash
    memento agent -m "你好，请帮我写一个快速排序算法。"
    # 或
    python cli/main.py agent -m "你好，请帮我写一个快速排序算法。"
    ```

---

## 4. 其他CLI命令

`memento` 命令还提供了其他有用的子命令：

*   **环境检查**: `memento doctor`
*   **技能验证**: `memento verify --help`
*   **飞书桥接**: `memento feishu`
