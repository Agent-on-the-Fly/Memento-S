#!/usr/bin/env bash
#
# Memento-S One-Click Installer
# Usage: ./install.sh
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

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
    echo "║                       One-Click Installer                             ║"
    echo "║                                                                       ║"
    echo "╚═══════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[  OK]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[FAIL]${NC} $1"; }

check_command() { command -v "$1" &>/dev/null; }

# ── resolve project root ─────────────────────────────────────────────
REPO_URL="https://github.com/Agent-on-the-Fly/Memento-S.git"
INSTALL_DIR="${MEMENTO_INSTALL_DIR:-$HOME/Memento-S}"
ROUTER_DATASET_REPO="AgentFly/router-data"
DEFAULT_OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL="anthropic/claude-3.5-sonnet"

# ── .env helpers ──────────────────────────────────────────────────────
_read_env_value() {
    local file="$1" key="$2"
    [ -f "$file" ] || return 0
    local line
    line="$(grep -E "^${key}=" "$file" | tail -n 1 || true)"
    [ -n "$line" ] || return 0
    line="${line#*=}"
    line="${line%\"}"
    line="${line#\"}"
    printf '%s' "$line"
}

_upsert_env_value() {
    local file="$1" key="$2" value="$3"
    mkdir -p "$(dirname "$file")"
    [ -f "$file" ] || touch "$file"
    local tmp="${file}.tmp.$$"
    awk -v k="$key" -v v="$value" '
        BEGIN { done = 0 }
        $0 ~ ("^" k "=") {
            if (!done) { print k "=" v; done = 1 }
            next
        }
        { print }
        END { if (!done) print k "=" v }
    ' "$file" > "$tmp"
    mv "$tmp" "$file"
}

_prompt_required() {
    local label="$1" current="$2" secret="$3" default_value="$4"
    local input=""
    while true; do
        if [ -n "$current" ]; then
            if [ "$secret" = "1" ]; then
                printf "%s (press Enter to keep existing): " "$label" >/dev/tty
                IFS= read -r -s input </dev/tty || input=""
                printf "\n" >/dev/tty
                [ -z "$input" ] && input="$current"
            else
                printf "%s [%s]: " "$label" "$current" >/dev/tty
                IFS= read -r input </dev/tty || input=""
                [ -z "$input" ] && input="$current"
            fi
        elif [ -n "$default_value" ]; then
            printf "%s [%s]: " "$label" "$default_value" >/dev/tty
            IFS= read -r input </dev/tty || input=""
            [ -z "$input" ] && input="$default_value"
        else
            if [ "$secret" = "1" ]; then
                printf "%s: " "$label" >/dev/tty
                IFS= read -r -s input </dev/tty || input=""
                printf "\n" >/dev/tty
            else
                printf "%s: " "$label" >/dev/tty
                IFS= read -r input </dev/tty || input=""
            fi
        fi
        [ -n "$input" ] && { printf '%s' "$input"; return 0; }
        printf "Input required. Please try again.\n" >/dev/tty
    done
}

# ═════════════════════════════════════════════════════════════════════
# Steps
# ═════════════════════════════════════════════════════════════════════

step_resolve_project() {
    # If running via curl|bash, BASH_SOURCE[0] is empty — need to clone first
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-}")" 2>/dev/null && pwd 2>/dev/null || echo "")"
    if [ -n "$script_dir" ] && [ -f "$script_dir/pyproject.toml" ]; then
        PROJECT_DIR="$script_dir"
        log_success "Project directory: $PROJECT_DIR"
    else
        if [ -d "$INSTALL_DIR/.git" ]; then
            log_info "Updating existing installation at $INSTALL_DIR..."
            git -C "$INSTALL_DIR" pull --ff-only 2>/dev/null || true
        else
            log_info "Cloning Memento-S to $INSTALL_DIR..."
            git clone "$REPO_URL" "$INSTALL_DIR"
        fi
        PROJECT_DIR="$INSTALL_DIR"
        log_success "Project cloned to: $PROJECT_DIR"
    fi
}

step_check_prerequisites() {
    echo ""
    log_info "Checking prerequisites..."

    # Python
    if check_command python3; then
        log_success "python3: $(python3 --version 2>&1)"
    else
        log_error "python3 is required. Please install Python 3.12+."
        exit 1
    fi

    # git
    if check_command git; then
        log_success "git: $(git --version 2>&1)"
    else
        log_warn "git not found (only needed for cloning/updating)"
    fi
}

step_install_uv() {
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
        log_error "Failed to install uv. Manual: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
}

step_install_dependencies() {
    echo ""
    echo -e "${CYAN}───────────────────────────────────────────────────────────────${NC}"
    echo -e "${CYAN}  Installing Dependencies                                      ${NC}"
    echo -e "${CYAN}───────────────────────────────────────────────────────────────${NC}"
    echo ""
    cd "$PROJECT_DIR"

    log_info "Pinning Python 3.12..."
    uv python install 3.12

    log_info "Running uv sync..."
    uv sync --python 3.12
    log_success "Dependencies installed"

    # nltk data (for crawl4ai)
    uv run python -c "import nltk; nltk.download('punkt_tab', quiet=True)" 2>/dev/null \
        && log_success "nltk data ready" \
        || log_warn "nltk data download skipped"

    # playwright / crawl4ai (optional)
    uv run crawl4ai-setup -q 2>/dev/null || true
    uv run python -m playwright install chromium 2>/dev/null \
        && log_success "Playwright ready" \
        || log_warn "Playwright skipped (can install later)"
}

step_download_router_assets() {
    cd "$PROJECT_DIR"
    local flag="${MEMENTO_DOWNLOAD_ROUTER:-1}"
    case "$(printf '%s' "$flag" | tr '[:upper:]' '[:lower:]')" in
        0|false|no|off) log_warn "Skipping router assets (MEMENTO_DOWNLOAD_ROUTER=$flag)"; return 0 ;;
    esac

    echo ""
    echo -e "${CYAN}───────────────────────────────────────────────────────────────${NC}"
    echo -e "${CYAN}  Downloading Router Assets                                    ${NC}"
    echo -e "${CYAN}───────────────────────────────────────────────────────────────${NC}"
    echo ""

    if ! uv run python -c "import huggingface_hub" >/dev/null 2>&1; then
        log_info "Installing huggingface_hub..."
        uv pip install huggingface_hub >/dev/null 2>&1 || {
            log_warn "Failed to install huggingface_hub; skipping"; return 0
        }
    fi

    local dl_emb="${MEMENTO_DOWNLOAD_ROUTER_EMBEDDINGS:-0}"
    log_info "Downloading skills_catalog.jsonl from $ROUTER_DATASET_REPO..."

    if MEMENTO_ROUTER_DATASET_REPO="$ROUTER_DATASET_REPO" \
       MEMENTO_DOWNLOAD_ROUTER_EMBEDDINGS="$dl_emb" \
       uv run python - <<'PY'
import os, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

root = Path.cwd()
router_root = root / "router_data"
router_root.mkdir(parents=True, exist_ok=True)
token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or None
repo = str(os.getenv("MEMENTO_ROUTER_DATASET_REPO") or "").strip()
dl_emb = str(os.getenv("MEMENTO_DOWNLOAD_ROUTER_EMBEDDINGS", "0")).strip().lower() not in {"", "0", "false", "no", "off"}

src = hf_hub_download(repo_id=repo, repo_type="dataset", filename="skills_catalog.jsonl", token=token)
shutil.copy2(src, router_root / "skills_catalog.jsonl")

if dl_emb and repo:
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(router_root), allow_patterns=["embeddings/*"], token=token)

print(f"catalog={'OK' if (router_root / 'skills_catalog.jsonl').exists() else 'MISSING'}")
PY
    then
        log_success "Router assets downloaded to: $PROJECT_DIR/router_data"
    else
        log_warn "Router asset download failed (network/auth). Retry later manually."
    fi
}

step_configure_env() {
    cd "$PROJECT_DIR"
    local env_file="$PROJECT_DIR/.env"
    local have_tty=0
    [ -e /dev/tty ] && [ -r /dev/tty ] && have_tty=1

    local existing_api_key existing_model existing_serpapi
    existing_api_key="$(_read_env_value "$env_file" "OPENROUTER_API_KEY")"
    existing_model="$(_read_env_value "$env_file" "OPENROUTER_MODEL")"
    existing_serpapi="$(_read_env_value "$env_file" "SERPAPI_API_KEY")"

    local api_key model serpapi_key
    api_key="${OPENROUTER_API_KEY:-$existing_api_key}"
    model="${OPENROUTER_MODEL:-$existing_model}"
    serpapi_key="${SERPAPI_API_KEY:-$existing_serpapi}"

    if [ "$have_tty" -eq 1 ]; then
        echo ""
        echo -e "${CYAN}───────────────────────────────────────────────────────────────${NC}"
        echo -e "${CYAN}  Configure API Keys                                           ${NC}"
        echo -e "${CYAN}───────────────────────────────────────────────────────────────${NC}"
        echo ""
        api_key="$(_prompt_required "OPENROUTER_API_KEY" "$api_key" "1" "")"
        model="$(_prompt_required "OPENROUTER_MODEL" "$model" "0" "$DEFAULT_OPENROUTER_MODEL")"
        serpapi_key="$(_prompt_required "SERPAPI_API_KEY (optional, Enter to skip)" "$serpapi_key" "1" "SKIP")"
        [ "$serpapi_key" = "SKIP" ] && serpapi_key=""
    else
        if [ -z "$api_key" ]; then
            log_error "OPENROUTER_API_KEY is required. Set it as env var or run interactively."
            exit 1
        fi
        [ -z "$model" ] && model="$DEFAULT_OPENROUTER_MODEL"
    fi

    _upsert_env_value "$env_file" "LLM_API" "openrouter"
    _upsert_env_value "$env_file" "OPENROUTER_API_KEY" "$api_key"
    _upsert_env_value "$env_file" "OPENROUTER_BASE_URL" "$DEFAULT_OPENROUTER_BASE_URL"
    _upsert_env_value "$env_file" "OPENROUTER_MODEL" "$model"
    [ -n "$serpapi_key" ] && _upsert_env_value "$env_file" "SERPAPI_API_KEY" "$serpapi_key"
    _upsert_env_value "$env_file" "SKILLS_DIR" "./skills"
    _upsert_env_value "$env_file" "SKILLS_EXTRA_DIRS" "./skill_extra"
    _upsert_env_value "$env_file" "WORKSPACE_DIR" "./workspace"
    _upsert_env_value "$env_file" "SEMANTIC_ROUTER_CATALOG_JSONL" "router_data/skills_catalog.jsonl"
    _upsert_env_value "$env_file" "SKILL_DYNAMIC_FETCH_CATALOG_JSONL" "router_data/skills_catalog.jsonl"

    chmod 600 "$env_file" 2>/dev/null || true

    # Ensure skill_extra directory exists
    mkdir -p "$PROJECT_DIR/skill_extra"

    log_success ".env configured"
}

step_create_launcher() {
    cd "$PROJECT_DIR"
    log_info "Creating global 'memento' command..."

    # Launcher wrapper script
    local launcher="$PROJECT_DIR/memento"
    cat > "$launcher" << LAUNCHER_EOF
#!/usr/bin/env bash
# Memento-S launcher — auto-generated by install.sh
SCRIPT_DIR="$PROJECT_DIR"
cd "\$SCRIPT_DIR"
exec uv run memento "\$@"
LAUNCHER_EOF
    chmod +x "$launcher"

    # Symlink into PATH
    local installed_link=""
    local link_dir="$HOME/.local/bin"

    if [ -w "/usr/local/bin" ]; then
        ln -sf "$launcher" /usr/local/bin/memento 2>/dev/null && installed_link="/usr/local/bin/memento"
    fi

    if [ -z "$installed_link" ]; then
        mkdir -p "$link_dir"
        ln -sf "$launcher" "$link_dir/memento"
        installed_link="$link_dir/memento"

        # Ensure ~/.local/bin is in PATH
        if ! echo "$PATH" | grep -q "$link_dir"; then
            local shell_rc="$HOME/.zshrc"
            [ -n "$BASH_VERSION" ] && shell_rc="$HOME/.bashrc"
            if [ -f "$shell_rc" ] && ! grep -q "$link_dir" "$shell_rc" 2>/dev/null; then
                echo "" >> "$shell_rc"
                echo "# Added by Memento-S installer" >> "$shell_rc"
                echo "export PATH=\"$link_dir:\$PATH\"" >> "$shell_rc"
            fi
            export PATH="$link_dir:$PATH"
        fi
    fi

    hash -r 2>/dev/null || true
    log_success "Launcher: $launcher"
    log_success "Symlink:  $installed_link"
}

step_verify() {
    echo ""
    echo -e "${CYAN}───────────────────────────────────────────────────────────────${NC}"
    echo -e "${CYAN}  Verification                                                 ${NC}"
    echo -e "${CYAN}───────────────────────────────────────────────────────────────${NC}"
    echo ""

    cd "$PROJECT_DIR"

    # Check import
    if uv run python -c "from cli.main import main; print('cli import OK')" 2>/dev/null; then
        log_success "CLI module imports correctly"
    else
        log_error "CLI module import failed!"
        exit 1
    fi

    # Check entry point
    if uv run memento --help >/dev/null 2>&1; then
        log_success "'uv run memento' works"
    else
        log_warn "'uv run memento --help' failed (may need --help flag support)"
    fi

    # List detected local skills
    local skill_count=0
    for d in "$PROJECT_DIR"/skills/*/SKILL.md; do
        [ -f "$d" ] && skill_count=$((skill_count + 1))
    done
    log_success "Local skills detected: $skill_count"

    # Check .env
    if [ -f "$PROJECT_DIR/.env" ]; then
        log_success ".env file present"
    else
        log_warn ".env not found — run install.sh again to configure"
    fi
}

print_success() {
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}            Installation Complete!                             ${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${CYAN}Project dir :${NC} $PROJECT_DIR"
    echo -e "  ${CYAN}Command     :${NC} ${GREEN}memento${NC}"
    echo ""
    echo -e "  ${YELLOW}Quick start:${NC}"
    echo -e "    ${GREEN}memento${NC}              # interactive CLI"
    echo -e "    ${GREEN}memento -p \"...\"${NC}     # single prompt"
    echo -e "    ${GREEN}memento --help${NC}       # all options"
    echo ""
    echo -e "  ${YELLOW}Inside the CLI:${NC}"
    echo -e "    /status   - session info"
    echo -e "    /skills   - list local skills"
    echo -e "    /help     - all commands"
    echo -e "    Ctrl+C    - interrupt current task"
    echo -e "    /exit     - quit"
    echo ""
    echo -e "  ${YELLOW}Note:${NC} If 'memento' is not found, restart your terminal or run:"
    echo -e "        ${CYAN}source ~/.zshrc${NC}"
    echo ""
}

# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════
main() {
    print_banner
    step_resolve_project
    step_check_prerequisites
    step_install_uv
    step_install_dependencies
    step_download_router_assets
    step_configure_env
    step_create_launcher
    step_verify
    print_success
}

main "$@"
