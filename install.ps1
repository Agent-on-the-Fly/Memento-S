#
# Memento-S One-Click Installer for Windows (uv version)
# Usage: irm https://raw.githubusercontent.com/Agent-on-the-Fly/Memento-S/main/install.ps1 | iex
#        or: .\install.ps1
#

$ErrorActionPreference = "Stop"

# Config
$REPO_URL = "https://github.com/Agent-on-the-Fly/Memento-S.git"
$INSTALL_DIR = if ($env:MEMENTO_INSTALL_DIR) { $env:MEMENTO_INSTALL_DIR } else { Join-Path $env:USERPROFILE "Memento-S" }

function Print-Banner {
    Write-Host ""
    Write-Host "+=======================================================================+" -ForegroundColor Cyan
    Write-Host "|                                                                       |" -ForegroundColor Cyan
    Write-Host "|   ###   ###  #######  ###   ###  #######  ###   ##  ########  ######   |" -ForegroundColor Cyan
    Write-Host "|   #### ####  ##       #### ####  ##       ####  ##     ##    ##    ##  |" -ForegroundColor Cyan
    Write-Host "|   ## ### ##  #####    ## ### ##   #####    ## ## ##     ##    ##    ##  |" -ForegroundColor Cyan
    Write-Host "|   ##  #  ##  ##       ##  #  ##   ##       ##  ####     ##    ##    ##  |" -ForegroundColor Cyan
    Write-Host "|   ##     ##  #######  ##     ##   #######  ##   ###     ##     ######   |" -ForegroundColor Cyan
    Write-Host "|                           Memento-S                                    |" -ForegroundColor Cyan
    Write-Host "|                   One-Click Installer (uv)                             |" -ForegroundColor Cyan
    Write-Host "|                                                                       |" -ForegroundColor Cyan
    Write-Host "+=======================================================================+" -ForegroundColor Cyan
    Write-Host ""
}

function Log-Info    { param($msg) Write-Host "[INFO] "  -ForegroundColor Blue   -NoNewline; Write-Host $msg }
function Log-Success { param($msg) Write-Host "[OK] "    -ForegroundColor Green  -NoNewline; Write-Host $msg }
function Log-Warn    { param($msg) Write-Host "[WARN] "  -ForegroundColor Yellow -NoNewline; Write-Host $msg }
function Log-Error   { param($msg) Write-Host "[ERROR] " -ForegroundColor Red    -NoNewline; Write-Host $msg }

function Test-Command {
    param($Name)
    $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

# Refresh PATH from registry so newly installed tools are visible
function Refresh-Path {
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath    = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path    = "$machinePath;$userPath"
}

# Install uv if not present
function Install-Uv {
    if (Test-Command "uv") {
        $ver = & uv --version 2>&1
        Log-Success "uv: $ver"
        return
    }

    Log-Info "Installing uv..."
    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    } catch {
        Log-Error "Failed to download uv installer: $_"
        exit 1
    }

    Refresh-Path

    # Also add common uv install locations to current session
    $uvPaths = @(
        (Join-Path $env:USERPROFILE ".local\bin"),
        (Join-Path $env:USERPROFILE ".cargo\bin"),
        (Join-Path $env:LOCALAPPDATA "uv\bin")
    )
    foreach ($p in $uvPaths) {
        if ((Test-Path $p) -and ($env:Path -notlike "*$p*")) {
            $env:Path = "$p;$env:Path"
        }
    }

    if (Test-Command "uv") {
        $ver = & uv --version 2>&1
        Log-Success "uv installed: $ver"
    } else {
        Log-Error "Failed to install uv. Please install manually: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    }
}

# Check if running from local project directory
function Test-LocalInstall {
    $scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
    (Test-Path (Join-Path $scriptDir "cli\\main.py")) -and (Test-Path (Join-Path $scriptDir "skills"))
}

# Clone or update repository
function Setup-Repository {
    Log-Info "Setting up repository..."

    $scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }

    if (Test-LocalInstall) {
        Log-Info "Detected local installation from: $scriptDir"
        $script:INSTALL_DIR = $scriptDir
        Set-Location $INSTALL_DIR
        Log-Success "Using local directory: $INSTALL_DIR"
    } elseif (Test-Path (Join-Path $INSTALL_DIR ".git")) {
        Log-Info "Repository exists, updating..."
        Set-Location $INSTALL_DIR
        try {
            & git pull --rebase 2>&1 | Out-Null
            Log-Success "Repository updated at $INSTALL_DIR"
        } catch {
            Log-Warn "Git pull failed, continuing with existing code"
        }
    } else {
        Log-Info "Cloning repository to $INSTALL_DIR..."
        & git clone $REPO_URL $INSTALL_DIR
        Set-Location $INSTALL_DIR
        Log-Success "Repository cloned to $INSTALL_DIR"
    }
}

# Install dependencies using uv sync
function Install-Dependencies {
    Write-Host ""
    Write-Host "===============================================================" -ForegroundColor Cyan
    Write-Host "            Installing Dependencies (uv sync)                  " -ForegroundColor Cyan
    Write-Host "===============================================================" -ForegroundColor Cyan
    Write-Host ""

    Set-Location $INSTALL_DIR

    # Ensure Python 3.12 and sync dependencies
    Log-Info "Installing Python 3.12..."
    & uv python install 3.12

    Log-Info "Running uv sync with Python 3.12..."
    & uv sync --python 3.12

    Log-Success "Dependencies installed!"

    # Download nltk data for crawl4ai
    Log-Info "Downloading nltk data..."
    try {
        & uv run python -c "import nltk; nltk.download('punkt_tab', quiet=True)" 2>&1 | Out-Null
    } catch {
        Log-Warn "nltk data download skipped"
    }

    # Setup playwright/crawl4ai (optional, may fail)
    Log-Info "Setting up browser support..."
    try {
        & uv run crawl4ai-setup -q 2>&1 | Out-Null
    } catch {
        Log-Warn "crawl4ai setup skipped"
    }
    try {
        & uv run python -m playwright install chromium 2>&1 | Out-Null
    } catch {
        Log-Warn "Playwright setup skipped (can install later)"
    }
}

# Create launcher scripts
function Create-Launcher {
    Log-Info "Creating launcher scripts..."

    # Create .cmd launcher for CMD
    $cmdLauncher = Join-Path $INSTALL_DIR "memento.cmd"
    $cmdContent = @"
@echo off
pushd "%~dp0"
if "%~1"=="" (
    uv run python -m cli
) else (
    uv run python -m cli %*
)
popd
"@
    Set-Content -Path $cmdLauncher -Value $cmdContent -Encoding ASCII
    Log-Success "CMD launcher created: $cmdLauncher"

    # Create .ps1 launcher for PowerShell
    $ps1Launcher = Join-Path $INSTALL_DIR "memento.ps1"
    $ps1Content = @'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Push-Location $ScriptDir
try {
    if ($args.Count -eq 0) {
        & uv run python -m cli
    } else {
        & uv run python -m cli @args
    }
} finally {
    Pop-Location
}
'@
    Set-Content -Path $ps1Launcher -Value $ps1Content -Encoding UTF8
    Log-Success "PowerShell launcher created: $ps1Launcher"

    # Add INSTALL_DIR to user PATH if not already there
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($userPath -notlike "*$INSTALL_DIR*") {
        Log-Info "Adding $INSTALL_DIR to user PATH..."
        [Environment]::SetEnvironmentVariable("Path", "$INSTALL_DIR;$userPath", "User")
        $env:Path = "$INSTALL_DIR;$env:Path"
        Log-Success "Added to user PATH"
        Log-Warn "Restart terminal for PATH changes to take effect"
    }
}

function Print-Success {
    Write-Host ""
    Write-Host "===============================================================" -ForegroundColor Green
    Write-Host "                 Installation Complete!                        " -ForegroundColor Green
    Write-Host "===============================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Install directory: " -ForegroundColor Cyan -NoNewline; Write-Host $INSTALL_DIR
    Write-Host ""
    Write-Host "  To start Memento-S:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "    memento"                                              -ForegroundColor Green -NoNewline; Write-Host "                  # Start CLI (after restarting terminal)"
    Write-Host "    cd $INSTALL_DIR; uv run python -m cli" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Other commands:" -ForegroundColor Cyan
    Write-Host "    /status          - Show session status"
    Write-Host "    /skills <query>  - Search cloud skills"
    Write-Host "    memento --help   - Show all commands"
    Write-Host ""
    Write-Host "  Note: " -ForegroundColor Yellow -NoNewline
    Write-Host "If 'memento' not found, restart your terminal so PATH updates take effect."
    Write-Host ""
}

# Main
function Main {
    Print-Banner

    # Check git
    if (-not (Test-Command "git")) {
        Log-Error "git is required. Please install git first: https://git-scm.com/download/win"
        exit 1
    }

    Install-Uv
    Setup-Repository
    Install-Dependencies
    Create-Launcher
    Print-Success
}

Main
