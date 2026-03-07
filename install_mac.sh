#!/usr/bin/env bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-./}")" && pwd)"
REPO_URL="https://github.com/Agent-on-the-Fly/Memento-S.git"
DEFAULT_INSTALL_DIR="$HOME/Memento-S"

if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    PROJECT_ROOT="$SCRIPT_DIR"
else
    echo ""
    echo -e "${CYAN}Detected curl | bash mode. Need to clone the repository first.${NC}"
    read -r -p "Install directory [$DEFAULT_INSTALL_DIR]: " INSTALL_DIR < /dev/tty
    INSTALL_DIR="${INSTALL_DIR:-$DEFAULT_INSTALL_DIR}"

    if [ -d "$INSTALL_DIR" ] && [ -f "$INSTALL_DIR/requirements.txt" ]; then
        echo -e "${BLUE}[INFO]${NC} Directory exists, pulling latest changes..."
        git -C "$INSTALL_DIR" pull || true
    else
        echo -e "${BLUE}[INFO]${NC} Cloning Memento-S into $INSTALL_DIR..."
        git clone "$REPO_URL" "$INSTALL_DIR"
    fi
    PROJECT_ROOT="$INSTALL_DIR"
fi
PYTHON_VERSION="3.12"
EMBEDDING_DOWNLOAD_REQUIRED=false
RERANK_DOWNLOAD_REQUIRED=false

print_banner() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                       ║"
    echo "║   ███╗   ███╗███████╗███╗   ███╗███████╗███╗   ██╗████████╗ ██████╗    ║"
    echo "║   ████╗ ████║██╔════╝████╗ ████║██╔════╝████╗  ██║╚══██╔══╝██╔═══██╗   ║"
    echo "║   ██╔████╔██║█████╗  ██╔████╔██║█████╗  ██╔██╗ ██║   ██║   ██║   ██║   ║"
    echo "║   ██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║   ██║   ██║   ║"
    echo "║   ██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║███████╗██║ ╚████║   ██║   ╚██████╔╝   ║"
    echo "║   ╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝    ║"
    echo "║                           Memento-S                                   ║"
    echo "║                   Install (macOS, Local Source)                      ║"
    echo "║                                                                       ║"
    echo "╚═══════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_command() { command -v "$1" &> /dev/null; }

ask_yes_no() {
    local prompt="$1"
    local default_answer="${2:-N}"
    local reply

    while true; do
        if [[ "$default_answer" == "Y" ]]; then
            read -r -p "$prompt [Y/n]: " reply < /dev/tty
            reply="${reply:-Y}"
        else
            read -r -p "$prompt [y/N]: " reply < /dev/tty
            reply="${reply:-N}"
        fi

        case "$reply" in
            [Yy]|[Yy][Ee][Ss]) return 0 ;;
            [Nn]|[Nn][Oo]) return 1 ;;
            *) echo "Please answer yes or no." ;;
        esac
    done
}

ask_non_empty() {
    local prompt="$1"
    local value
    while true; do
        read -r -p "$prompt: " value < /dev/tty
        if [[ -n "$value" ]]; then
            echo "$value"
            return 0
        fi
        echo "Value cannot be empty."
    done
}

check_or_install_tmux() {
    if check_command tmux; then
        log_success "tmux: $(tmux -V 2>&1)"
        return 0
    fi

    log_info "tmux not found, attempting to install..."
    if [[ "$(uname -s)" == "Darwin" ]] && check_command brew; then
        if brew install tmux 2>/dev/null; then
            if check_command tmux; then
                log_success "tmux installed: $(tmux -V 2>&1)"
                return 0
            fi
        fi
    fi

    log_warn "tmux is not installed. Install manually: macOS: brew install tmux"
    return 1
}

install_uv() {
    if check_command uv; then
        log_success "uv: $(uv --version 2>&1)"
        return 0
    fi

    log_info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh

    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

    if check_command uv; then
        log_success "uv installed: $(uv --version 2>&1)"
    else
        log_error "Failed to install uv. Please install manually: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
}

ensure_python312() {
    if uv python find 3.12 &>/dev/null; then
        log_success "Python 3.12 available via uv"
        return 0
    fi
    log_info "Installing Python 3.12 via uv..."
    if ! uv python install 3.12; then
        log_error "Failed to install Python 3.12."
        exit 1
    fi
    log_success "Python 3.12 installed."
}

create_venv_and_install_deps() {
    cd "$PROJECT_ROOT" || { log_error "Project root not found: $PROJECT_ROOT"; exit 1; }

    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}            Creating .venv and Installing Dependencies          ${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""

    log_info "Creating .venv with Python $PYTHON_VERSION..."
    if ! uv venv .venv --python "$PYTHON_VERSION"; then
        log_error "Failed to create .venv."
        exit 1
    fi
    log_success ".venv created."

    log_info "Installing dependencies from requirements.txt..."
    if ! uv pip install --python .venv/bin/python -r requirements.txt; then
        log_error "Failed to install Python dependencies."
        exit 1
    fi

    log_info "Installing local CLI entry (memento)..."
    if ! uv pip install --python .venv/bin/python -e .; then
        log_error "Failed to install local project package for memento command."
        exit 1
    fi
    log_success "Dependencies and local CLI installed."
}

install_nvm_and_nodejs() {
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

    if check_command npm; then
        log_success "npm: $(npm --version 2>&1)"
        return 0
    fi

    log_info "npm not found, installing Node.js via nvm..."

    if [ ! -s "$NVM_DIR/nvm.sh" ]; then
        log_info "Installing nvm..."
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
        export NVM_DIR="$HOME/.nvm"
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    fi

    if ! command -v nvm &>/dev/null; then
        log_error "Failed to install nvm. Please install Node.js manually."
        return 1
    fi

    log_info "Installing Node.js LTS..."
    nvm install --lts
    nvm use --lts

    if check_command npm; then
        log_success "Node.js installed: $(node --version 2>&1), npm: $(npm --version 2>&1)"
    else
        log_warn "Failed to install Node.js. Skills installation will be skipped."
        return 1
    fi
}

install_openskills() {
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

    if ! check_command npm; then
        log_warn "npm not available. Skipping openskills installation."
        log_warn "To install later: npm install -g openskills && openskills sync -y"
        return
    fi

    if ! check_command openskills; then
        log_info "Installing openskills..."
        if ! npm install -g openskills 2>/dev/null; then
            sudo npm install -g openskills 2>/dev/null || {
                log_warn "Failed to install openskills. Skipping skills."
                return
            }
        fi
    fi

    log_info "Installing skills..."
    cd "$PROJECT_ROOT" || return
    for d in skills/*; do
        [ -d "$d" ] || continue
        name="$(basename "$d")"
        if [ -d ".agent/skills/$name" ]; then
            openskills update "$name" 2>/dev/null || true
        else
            openskills install "./skills/$name" --universal --yes 2>/dev/null || true
        fi
    done
    openskills sync -y 2>/dev/null || true
    log_success "Skills installed."
}

ensure_env_file() {
    local env_file="$PROJECT_ROOT/.env"
    if [ ! -f "$env_file" ]; then
        if [ -f "$PROJECT_ROOT/.env.example" ]; then
            cp "$PROJECT_ROOT/.env.example" "$env_file"
            log_info "Created .env from .env.example"
        else
            touch "$env_file"
            log_info "Created empty .env"
        fi
    fi
}

set_env_var() {
    local key="$1"
    local value="$2"
    local env_file="$PROJECT_ROOT/.env"

    python3 - "$env_file" "$key" "$value" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]
line = f"{key}={value}"

if path.exists():
    text = path.read_text(encoding="utf-8")
else:
    text = ""

pattern = re.compile(rf"^\s*#?\s*{re.escape(key)}\s*=.*$", re.MULTILINE)
if pattern.search(text):
    text = pattern.sub(line, text, count=1)
else:
    if text and not text.endswith("\n"):
        text += "\n"
    text += line + "\n"

path.write_text(text, encoding="utf-8")
PY
}

configure_retrieval_models() {
    ensure_env_file

    log_info "Setting default retrieval config: BM25-only, top_k=5, no embedding, no rerank."
    set_env_var "EMBEDDING_MODEL" "none"
    set_env_var "EMBEDDING_BASE_URL" ""
    set_env_var "EMBEDDING_API_KEY" ""
    set_env_var "EMBEDDING_WEIGHT" "0"
    set_env_var "BM25_WEIGHT" "1"
    set_env_var "RERANKER_ENABLED" "false"
    set_env_var "RERANKER_MODEL" ""
    set_env_var "RERANKER_BASE_URL" ""
    set_env_var "RERANKER_API_KEY" ""
    set_env_var "RETRIEVAL_TOP_K" "5"
    EMBEDDING_DOWNLOAD_REQUIRED=false
    RERANK_DOWNLOAD_REQUIRED=false
    log_success "Retrieval config: BM25-only, RETRIEVAL_TOP_K=5"
}

download_optional_models() {
    local py="$PROJECT_ROOT/.venv/bin/python"
    if [ ! -x "$py" ]; then
        log_warn ".venv not found, skipping model downloads."
        return
    fi

    if [[ "$EMBEDDING_DOWNLOAD_REQUIRED" == "true" ]]; then
        log_info "Downloading BAAI/bge-m3 embedding model (may take a while)..."
        if HF_HUB_DISABLE_PROGRESS_BARS=1 "$py" -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"; then
            log_success "BAAI/bge-m3 downloaded."
        else
            log_warn "Embedding model download failed (network or disk). You can retry later."
        fi
    else
        log_info "Skipping embedding model download."
    fi

    if [[ "$RERANK_DOWNLOAD_REQUIRED" == "true" ]]; then
        log_info "Downloading BAAI/bge-reranker-v2-m3 model (may take a while)..."
        if HF_HUB_DISABLE_PROGRESS_BARS=1 "$py" -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"; then
            log_success "BAAI/bge-reranker-v2-m3 downloaded."
        else
            log_warn "Rerank model download failed (network or disk). You can retry later."
        fi
    else
        log_info "Skipping rerank model download."
    fi
}

configure_llm_and_keys() {
    ensure_env_file

    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}                    LLM & API Configuration                    ${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""

    # --- LLM_API ---
    echo -e "${YELLOW}Select LLM Provider:${NC}"
    echo "  1) anthropic"
    echo "  2) openai"
    echo "  3) google"
    echo "  4) openrouter"
    echo "  5) ollama"
    local llm_choice
    read -r -p "Select number [3]: " llm_choice < /dev/tty
    llm_choice="${llm_choice:-3}"
    local llm_api
    case "$llm_choice" in
        1) llm_api="anthropic" ;;
        2) llm_api="openai" ;;
        3) llm_api="google" ;;
        4) llm_api="openrouter" ;;
        5) llm_api="ollama" ;;
        *) llm_api="google" ;;
    esac
    set_env_var "LLM_API" "$llm_api"
    log_success "LLM Provider: $llm_api"

    # --- LLM_MODEL ---
    local default_model="google/gemini-3-flash-preview"
    local llm_model
    read -r -p "LLM Model [$default_model]: " llm_model < /dev/tty
    llm_model="${llm_model:-$default_model}"
    set_env_var "LLM_MODEL" "$llm_model"
    log_success "LLM Model: $llm_model"

    # --- LLM_API_KEY ---
    local llm_key
    read -r -p "LLM API Key: " llm_key < /dev/tty
    if [[ -n "$llm_key" ]]; then
        set_env_var "LLM_API_KEY" "$llm_key"
        log_success "LLM API Key: set"
    else
        log_warn "LLM API Key: skipped (set it later in .env)"
    fi

    # --- LLM_BASE_URL ---
    local llm_base
    read -r -p "LLM Base URL (press Enter to skip): " llm_base < /dev/tty
    if [[ -n "$llm_base" ]]; then
        set_env_var "LLM_BASE_URL" "$llm_base"
        log_success "LLM Base URL: $llm_base"
    else
        set_env_var "LLM_BASE_URL" ""
        log_info "LLM Base URL: default"
    fi

    # --- SERPER_API_KEY ---
    echo ""
    local serper_key
    read -r -p "Serper API Key (for web search, press Enter to skip): " serper_key < /dev/tty
    if [[ -n "$serper_key" ]]; then
        set_env_var "SERPER_API_KEY" "$serper_key"
        log_success "Serper API Key: set"
    else
        log_warn "Serper API Key: skipped (web search will be unavailable)"
    fi

    # --- Defaults ---
    set_env_var "LLM_MAX_TOKENS" "4096"
    set_env_var "LLM_TEMPERATURE" "0.7"
    set_env_var "LLM_TIMEOUT" "120"
    set_env_var "AGENT_MAX_ITERATIONS" "100"
    set_env_var "LOG_LEVEL" "ERROR"
    set_env_var "EMBEDDING_MODEL" "none"
    set_env_var "SKILLS_CATALOG_PATH" "router_data/skills_catalog.jsonl"
    set_env_var "WORKSPACE_DIR" "workspace"
    set_env_var "CONVERSATIONS_DIR" "conversations"

    echo ""
    log_success "Configuration written to .env"
}

init_vector_db() {
    local py="$PROJECT_ROOT/.venv/bin/python"
    if [ ! -x "$py" ]; then
        log_warn ".venv not found, skipping vector DB init."
        return
    fi

    if [ ! -f "$PROJECT_ROOT/cli/main.py" ]; then
        log_warn "cli/main.py not found in $PROJECT_ROOT, skipping vector DB init."
        return
    fi

    log_info "Initializing vector database (running agent until ready)..."
    (
        cd "$PROJECT_ROOT" || exit 1
        "$py" cli/main.py agent 2>&1 | while IFS= read -r line; do
            echo "$line"
            if echo "$line" | grep -q "You"; then
                break
            fi
        done
    ) || true
    log_success "Vector database initialized."
}

create_global_symlink() {
    local venv_bin="$PROJECT_ROOT/.venv/bin/memento"
    local target_dir="$HOME/.local/bin"
    local target="$target_dir/memento"

    if [ ! -x "$venv_bin" ]; then
        log_warn "memento entry point not found at $venv_bin, skipping global symlink."
        return
    fi

    # Create a wrapper script instead of symlink so it works even if
    # the venv's Python needs the project directory as cwd.
    mkdir -p "$target_dir"

    cat > "$target" <<WRAPPER
#!/usr/bin/env bash
cd "$PROJECT_ROOT" || exit 1
exec "$venv_bin" "\$@"
WRAPPER
    chmod +x "$target"

    # Ensure ~/.local/bin is on PATH in common shell rc files
    local added_path=false
    for rc in "$HOME/.zshrc" "$HOME/.bashrc" "$HOME/.bash_profile"; do
        if [ -f "$rc" ] && ! grep -q '\.local/bin' "$rc" 2>/dev/null; then
            echo '' >> "$rc"
            echo '# Added by Memento-S installer' >> "$rc"
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$rc"
            added_path=true
        fi
    done

    if [[ ":$PATH:" != *":$target_dir:"* ]]; then
        export PATH="$target_dir:$PATH"
    fi

    if check_command memento; then
        log_success "Global command installed: memento"
    else
        log_warn "Created $target but it may not be on PATH yet."
        log_warn "Run: export PATH=\"\$HOME/.local/bin:\$PATH\"  or restart your terminal."
    fi

    if $added_path; then
        log_info "Added ~/.local/bin to PATH in shell rc files. Restart terminal to take effect."
    fi
}

print_success() {
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}                 Installation Complete!                        ${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${CYAN}Install directory:${NC} $PROJECT_ROOT"
    echo ""
    echo -e "  ${YELLOW}To start Memento-S:${NC}"
    echo ""
    echo -e "    ${GREEN}memento${NC}"
    echo ""
    echo -e "  ${CYAN}Other commands:${NC}"
    echo -e "    memento config    - Manage configuration"
    echo -e "    memento doctor    - Check configuration"
    echo -e "    memento --help    - Show all commands"
    echo ""
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        echo -e "  ${YELLOW}Note:${NC} If 'memento' is not found, restart your terminal or run:"
        echo -e "    export PATH=\"\$HOME/.local/bin:\$PATH\""
        echo ""
    fi
}

main() {
    print_banner

    if ! check_command curl; then
        log_error "curl is required (for uv/nvm). Please install curl first."
        exit 1
    fi

    if [ ! -d "$PROJECT_ROOT" ] || [ ! -f "$PROJECT_ROOT/requirements.txt" ]; then
        log_error "Project root not found or invalid (no requirements.txt): $PROJECT_ROOT"
        exit 1
    fi

    check_or_install_tmux || true

    install_uv
    ensure_python312

    create_venv_and_install_deps

    install_nvm_and_nodejs
    install_openskills

    configure_llm_and_keys
    configure_retrieval_models
    download_optional_models

    init_vector_db

    create_global_symlink

    print_success
}

main "$@"
