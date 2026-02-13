#!/usr/bin/env bash
#
# Memento-S One-Click Installer (uv version)
# Usage: curl -sSL https://raw.githubusercontent.com/Agent-on-the-Fly/Memento-S/main/install.sh | bash
#        or: ./install.sh
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Config
REPO_URL="https://github.com/Agent-on-the-Fly/Memento-S.git"
INSTALL_DIR="${MEMENTO_INSTALL_DIR:-$HOME/Memento-S}"
DEFAULT_OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL="anthropic/claude-3.5-sonnet"
ROUTER_DATASET_REPO="AgentFly/router-data"

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
    echo "║                   One-Click Installer (uv)                            ║"
    echo "║                                                                       ║"
    echo "╚═══════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_command() { command -v "$1" &> /dev/null; }

_read_env_value() {
    local file="$1"
    local key="$2"
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
    local file="$1"
    local key="$2"
    local value="$3"
    mkdir -p "$(dirname "$file")"
    [ -f "$file" ] || touch "$file"

    local tmp="${file}.tmp.$$"
    awk -v k="$key" -v v="$value" '
        BEGIN { done = 0 }
        $0 ~ ("^" k "=") {
            if (!done) {
                print k "=" v
                done = 1
            }
            next
        }
        { print }
        END {
            if (!done) print k "=" v
        }
    ' "$file" > "$tmp"
    mv "$tmp" "$file"
}

_prompt_required() {
    local label="$1"
    local current="$2"
    local secret="$3"
    local default_value="$4"
    local input=""

    while true; do
        if [ -n "$current" ]; then
            if [ "$secret" = "1" ]; then
                printf "%s (press Enter to keep existing): " "$label" > /dev/tty
                IFS= read -r -s input < /dev/tty || input=""
                printf "\n" > /dev/tty
                if [ -z "$input" ]; then
                    input="$current"
                fi
            else
                printf "%s [%s]: " "$label" "$current" > /dev/tty
                IFS= read -r input < /dev/tty || input=""
                if [ -z "$input" ]; then
                    input="$current"
                fi
            fi
        elif [ -n "$default_value" ]; then
            printf "%s [%s]: " "$label" "$default_value" > /dev/tty
            IFS= read -r input < /dev/tty || input=""
            if [ -z "$input" ]; then
                input="$default_value"
            fi
        else
            if [ "$secret" = "1" ]; then
                printf "%s: " "$label" > /dev/tty
                IFS= read -r -s input < /dev/tty || input=""
                printf "\n" > /dev/tty
            else
                printf "%s: " "$label" > /dev/tty
                IFS= read -r input < /dev/tty || input=""
            fi
        fi

        if [ -n "$input" ]; then
            printf '%s' "$input"
            return 0
        fi
        printf "Input required. Please try again.\n" > /dev/tty
    done
}

# Install uv if not present
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

# Check if running from local project directory
is_local_install() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    [ -f "$script_dir/cli/main.py" ] && [ -d "$script_dir/skills" ]
}

# Clone or update repository
setup_repository() {
    log_info "Setting up repository..."

    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    if is_local_install; then
        log_info "Detected local installation from: $script_dir"
        INSTALL_DIR="$script_dir"
        cd "$INSTALL_DIR"
        log_success "Using local directory: $INSTALL_DIR"
    elif [ -d "$INSTALL_DIR/.git" ]; then
        log_info "Repository exists, updating..."
        cd "$INSTALL_DIR"
        git pull --rebase || log_warn "Git pull failed, continuing with existing code"
        log_success "Repository updated at $INSTALL_DIR"
    else
        log_info "Cloning repository to $INSTALL_DIR..."
        git clone "$REPO_URL" "$INSTALL_DIR"
        cd "$INSTALL_DIR"
        log_success "Repository cloned to $INSTALL_DIR"
    fi
}

# Install dependencies using uv sync
install_dependencies() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}            Installing Dependencies (uv sync)                  ${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""

    cd "$INSTALL_DIR"

    # Ensure Python 3.12 and sync dependencies
    log_info "Installing Python 3.12..."
    uv python install 3.12

    log_info "Running uv sync with Python 3.12..."
    uv sync --python 3.12

    log_success "Dependencies installed!"

    # Download nltk data for crawl4ai
    log_info "Downloading nltk data..."
    uv run python -c "import nltk; nltk.download('punkt_tab', quiet=True)" 2>/dev/null || log_warn "nltk data download skipped"

    # Setup playwright/crawl4ai (optional, may fail)
    log_info "Setting up browser support..."
    uv run crawl4ai-setup -q 2>/dev/null || log_warn "crawl4ai setup skipped"
    uv run python -m playwright install chromium 2>/dev/null || log_warn "Playwright setup skipped (can install later)"
}

download_router_assets() {
    cd "$INSTALL_DIR"
    local download_flag="${MEMENTO_DOWNLOAD_ROUTER:-1}"
    case "$(printf '%s' "$download_flag" | tr '[:upper:]' '[:lower:]')" in
        0|false|no|off)
            log_warn "Skipping router asset download (MEMENTO_DOWNLOAD_ROUTER=$download_flag)"
            return 0
            ;;
    esac

    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}               Downloading Router Assets                        ${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""

    if ! uv run python -c "import huggingface_hub" >/dev/null 2>&1; then
        log_info "Installing huggingface_hub..."
        uv pip install huggingface_hub >/dev/null 2>&1 || {
            log_warn "Failed to install huggingface_hub; skipping router asset download"
            return 0
        }
    fi

    local download_embeddings_flag="${MEMENTO_DOWNLOAD_ROUTER_EMBEDDINGS:-0}"
    local emb_flag_lc
    emb_flag_lc="$(printf '%s' "$download_embeddings_flag" | tr '[:upper:]' '[:lower:]')"

    log_info "Downloading router dataset index: $ROUTER_DATASET_REPO (skills_catalog.jsonl)"

    if MEMENTO_ROUTER_DATASET_REPO="$ROUTER_DATASET_REPO" \
       MEMENTO_DOWNLOAD_ROUTER_EMBEDDINGS="$download_embeddings_flag" \
       uv run python - <<'PY'
import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download


def as_bool(value: str) -> bool:
    return str(value or "").strip().lower() not in {"", "0", "false", "no", "off"}


root = Path.cwd()
router_root = root / "router_data"
router_root.mkdir(parents=True, exist_ok=True)
token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or None
dataset_repo = str(os.getenv("MEMENTO_ROUTER_DATASET_REPO") or "").strip()
download_embeddings = as_bool(os.getenv("MEMENTO_DOWNLOAD_ROUTER_EMBEDDINGS", "0"))

# Always fetch skills catalog first (fast, required).
catalog_src = hf_hub_download(
    repo_id=dataset_repo,
    repo_type="dataset",
    filename="skills_catalog.jsonl",
    token=token,
)
shutil.copy2(catalog_src, router_root / "skills_catalog.jsonl")

# Optional: fetch embeddings only when explicitly enabled.
if download_embeddings and dataset_repo:
    snapshot_download(
        repo_id=dataset_repo,
        repo_type="dataset",
        local_dir=str(router_root),
        allow_patterns=["embeddings/*"],
        token=token,
    )

print(f"router_data_dir={router_root}")
print(f"skills_catalog_exists={(router_root / 'skills_catalog.jsonl').exists()}")
print(f"embeddings_downloaded={download_embeddings}")
PY
    then
        log_success "Router assets downloaded to: $INSTALL_DIR/router_data"
        if [ "$emb_flag_lc" = "0" ] || [ "$emb_flag_lc" = "false" ] || [ "$emb_flag_lc" = "no" ] || [ "$emb_flag_lc" = "off" ] || [ -z "$emb_flag_lc" ]; then
            log_info "Skipped dataset embeddings by default. Set MEMENTO_DOWNLOAD_ROUTER_EMBEDDINGS=1 to download."
        fi
    else
        log_warn "Router asset download failed (network/auth). You can retry later manually."
    fi
}

configure_env() {
    cd "$INSTALL_DIR"
    local env_file="$INSTALL_DIR/.env"
    local have_tty=0
    if [ -e /dev/tty ] && [ -r /dev/tty ]; then
        have_tty=1
    fi

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
        echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
        echo -e "${CYAN}               Configure Required API Keys                      ${NC}"
        echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
        echo ""
        api_key="$(_prompt_required "OPENROUTER_API_KEY" "$api_key" "1" "")"
        model="$(_prompt_required "OPENROUTER_MODEL" "$model" "0" "$DEFAULT_OPENROUTER_MODEL")"
        serpapi_key="$(_prompt_required "SERPAPI_API_KEY" "$serpapi_key" "1" "")"
    else
        if [ -z "$api_key" ] || [ -z "$model" ] || [ -z "$serpapi_key" ]; then
            log_error "Missing required configuration in non-interactive mode."
            log_error "Set env vars before install: OPENROUTER_API_KEY OPENROUTER_MODEL SERPAPI_API_KEY"
            exit 1
        fi
    fi

    _upsert_env_value "$env_file" "LLM_API" "openrouter"
    _upsert_env_value "$env_file" "OPENROUTER_API_KEY" "$api_key"
    _upsert_env_value "$env_file" "OPENROUTER_BASE_URL" "$DEFAULT_OPENROUTER_BASE_URL"
    _upsert_env_value "$env_file" "OPENROUTER_MODEL" "$model"
    _upsert_env_value "$env_file" "SERPAPI_API_KEY" "$serpapi_key"
    _upsert_env_value "$env_file" "SKILLS_DIR" "./skills"
    _upsert_env_value "$env_file" "SKILLS_EXTRA_DIRS" "./skills-extra"
    _upsert_env_value "$env_file" "WORKSPACE_DIR" "./workspace"
    _upsert_env_value "$env_file" "SEMANTIC_ROUTER_CATALOG_JSONL" "router_data/skills_catalog.jsonl"
    _upsert_env_value "$env_file" "SKILL_DYNAMIC_FETCH_CATALOG_JSONL" "router_data/skills_catalog.jsonl"

    chmod 600 "$env_file" 2>/dev/null || true
    log_success "Configured .env (OPENROUTER_BASE_URL uses default: $DEFAULT_OPENROUTER_BASE_URL)"
}

sync_local_skills_catalog() {
    cd "$INSTALL_DIR"
    log_info "Syncing local skills into AGENTS.md..."
    if uv run python - <<'PY'
from pathlib import Path
import re

from core.skill_engine.skill_catalog import build_available_skills_xml

ROOT = Path.cwd()
AGENTS = ROOT / "AGENTS.md"
roots = [ROOT / "skills", ROOT / "skills-extra"]

skills = []
seen = set()
for root in roots:
    if not root.exists() or not root.is_dir():
        continue
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        skill_md = child / "SKILL.md"
        if not skill_md.exists():
            continue
        text = skill_md.read_text(encoding="utf-8", errors="ignore")
        front = ""
        m = re.match(r"\s*---\s*\n(.*?)\n---\s*\n?", text, re.DOTALL)
        if m:
            front = m.group(1)
        name_m = re.search(r"(?mi)^name:\s*(.+?)\s*$", front)
        desc_m = re.search(r"(?mi)^description:\s*(.+?)\s*$", front)
        name = (name_m.group(1).strip().strip('"').strip("'") if name_m else child.name)
        desc = (desc_m.group(1).strip().strip('"').strip("'") if desc_m else "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        skills.append({"name": name, "description": desc})

if not skills:
    raise SystemExit("no local skills found")

xml = build_available_skills_xml(skills)
if AGENTS.exists():
    content = AGENTS.read_text(encoding="utf-8")
    if re.search(r"<available_skills>.*?</available_skills>", content, re.DOTALL):
        content = re.sub(r"<available_skills>.*?</available_skills>", xml, content, count=1, flags=re.DOTALL)
    else:
        content = content.rstrip() + "\n\n" + xml + "\n"
else:
    content = (
        "# AGENTS\n\n"
        "<skills_system priority=\"1\">\n\n"
        "<usage>\n"
        "Auto-generated by install.sh. Local skills are listed below.\n"
        "</usage>\n\n"
        f"{xml}\n\n"
        "</skills_system>\n"
    )
AGENTS.write_text(content, encoding="utf-8")
print(f"synced {len(skills)} skills")
PY
    then
        log_success "Local skills synced to AGENTS.md"
    else
        log_warn "Skill sync failed; keeping existing AGENTS.md"
    fi
}

# Create launcher script
create_launcher() {
    log_info "Creating launcher script..."

    LAUNCHER="$INSTALL_DIR/memento"
    cat > "$LAUNCHER" << 'EOF'
#!/usr/bin/env bash
SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
    DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"

cd "$SCRIPT_DIR"

# Use uv run - it handles venv automatically
if [ $# -eq 0 ]; then
    uv run python -m cli
else
    uv run python -m cli "$@"
fi
EOF
    chmod +x "$LAUNCHER"

    # Create command link with robust fallback
    local installed_link=""
    if [ -w "/usr/local/bin" ]; then
        if ln -sf "$LAUNCHER" /usr/local/bin/memento 2>/dev/null; then
            installed_link="/usr/local/bin/memento"
            log_success "Symlink: $installed_link"
        fi
    elif [ -d "/usr/local/bin" ]; then
        if sudo ln -sf "$LAUNCHER" /usr/local/bin/memento 2>/dev/null; then
            installed_link="/usr/local/bin/memento"
            log_success "Symlink: $installed_link (sudo)"
        else
            log_warn "Could not write /usr/local/bin/memento, falling back to ~/.local/bin"
        fi
    fi

    if [ -z "$installed_link" ]; then
        mkdir -p "$HOME/.local/bin"
        ln -sf "$LAUNCHER" "$HOME/.local/bin/memento"
        installed_link="$HOME/.local/bin/memento"
        add_to_path "$HOME/.local/bin"
        log_success "Symlink: $installed_link"
    fi

    # Refresh shell command cache so `memento` resolves immediately in current shell.
    hash -r 2>/dev/null || true

    local resolved_cmd=""
    resolved_cmd="$(command -v memento 2>/dev/null || true)"
    if [ -n "$resolved_cmd" ]; then
        log_success "Command available: memento -> $resolved_cmd"
    else
        log_warn "Command not yet in PATH for this shell. Reopen terminal, then run: memento"
    fi

    log_success "Launcher created: $LAUNCHER"
}

# Add directory to PATH
add_to_path() {
    local dir="$1"
    echo "$PATH" | grep -q "$dir" && return

    local shell_rc="$HOME/.zshrc"
    [ -f "$HOME/.bashrc" ] && [ "$SHELL" = *bash* ] && shell_rc="$HOME/.bashrc"

    if [ -f "$shell_rc" ] && ! grep -q "$dir" "$shell_rc" 2>/dev/null; then
        echo "" >> "$shell_rc"
        echo "# Added by Memento-S installer" >> "$shell_rc"
        echo "export PATH=\"$dir:\$PATH\"" >> "$shell_rc"
        log_success "Added to $shell_rc"
    fi

    export PATH="$dir:$PATH"
    log_warn "Restart terminal or run: source $shell_rc"
}

print_success() {
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}                 Installation Complete!                        ${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  ${CYAN}Install directory:${NC} $INSTALL_DIR"
    echo -e "  ${CYAN}Entrypoint:${NC} ${GREEN}memento -> uv run python -m cli${NC}"
    echo ""
    echo -e "  ${YELLOW}To start Memento-S:${NC}"
    echo ""
    echo -e "    ${GREEN}memento${NC}                  # Start CLI"
    echo -e "    ${GREEN}cd $INSTALL_DIR && uv run python -m cli${NC}"
    echo ""
    echo -e "  ${CYAN}Other commands:${NC}"
    echo -e "    /status          - Show session status"
    echo -e "    /skills <query>  - Search cloud skills"
    echo -e "    memento --help   - Show all commands"
    echo ""
    echo -e "  ${YELLOW}Note:${NC} If 'memento' not found, restart terminal or run:"
    echo -e "        ${CYAN}source ~/.zshrc${NC} (or ~/.bashrc)"
    echo ""
}

# Main
main() {
    print_banner

    # Check git
    if ! check_command git; then
        log_error "git is required. Please install git first."
        exit 1
    fi

    install_uv
    setup_repository
    install_dependencies
    download_router_assets
    configure_env
    sync_local_skills_catalog
    create_launcher
    print_success
}

main "$@"
