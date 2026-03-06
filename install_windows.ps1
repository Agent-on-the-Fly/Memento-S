#Requires -Version 5.1
<#
.SYNOPSIS
    Memento-S  (Windows)

.DESCRIPTION
     PROJECT_ROOT uv/python3.12/.venvNode.jsopenskills
     BAAI/bge-m3
    : PowerShell 5.1+
    : tmuxWindows 

.EXAMPLE
    .\install_windows.ps1  
#>

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$PYTHON_VERSION = "3.12"
$PROJECT_ROOT = $PSScriptRoot
$EMBEDDING_DOWNLOAD_REQUIRED = $false
$RERANK_DOWNLOAD_REQUIRED = $false

function Log-Info { param([string]$Msg) Write-Host "[INFO] $Msg" -ForegroundColor Blue }
function Log-Success { param([string]$Msg) Write-Host "[OK] $Msg"   -ForegroundColor Green }
function Log-Warn { param([string]$Msg) Write-Host "[WARN] $Msg" -ForegroundColor Yellow }
function Log-Error { param([string]$Msg) Write-Host "[ERROR] $Msg" -ForegroundColor Red }

function Test-CommandExists {
    param([string]$Command)
    $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

function ConvertTo-EnvValue {
    param([AllowEmptyString()][string]$Value)

    if ($null -eq $Value) {
        return ""
    }

    $escapedValue = $Value.Replace('"', '\"')
    $needsQuote = ($escapedValue -match '\s') -or $escapedValue.Contains('#') -or $escapedValue.Contains('=') -or $escapedValue.StartsWith('"') -or $escapedValue.EndsWith('"')

    if ($needsQuote) {
        return '"' + $escapedValue + '"'
    }

    return $escapedValue
}

function Set-EnvVar {
    param(
        [Parameter(Mandatory = $true)][string]$Key,
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Value
    )

    $envFile = Join-Path $PROJECT_ROOT ".env"
    if (-not (Test-Path $envFile)) {
        $envExample = Join-Path $PROJECT_ROOT ".env.example"
        if (Test-Path $envExample) {
            Copy-Item $envExample $envFile -Force
            Log-Info "Created .env from .env.example"
        }
        else {
            New-Item -ItemType File -Path $envFile -Force | Out-Null
            Log-Info "Created empty .env"
        }
    }

    $lines = @()
    if (Test-Path $envFile) {
        $lines = Get-Content $envFile -ErrorAction SilentlyContinue
    }

    $pattern = "^\s*#?\s*" + [regex]::Escape($Key) + "\s*=.*$"
    $newLine = "$Key=$(ConvertTo-EnvValue -Value $Value)"
    $updated = $false

    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match $pattern) {
            $lines[$i] = $newLine
            $updated = $true
            break
        }
    }

    if (-not $updated) {
        $lines += $newLine
    }

    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllLines($envFile, $lines, $utf8NoBom)
}

function Configure-RetrievalModels {
    Write-Host ""
    Write-Host "===============================================================" -ForegroundColor Cyan
    Write-Host "               Embedding / Rerank Configuration               " -ForegroundColor Cyan
    Write-Host "===============================================================" -ForegroundColor Cyan
    Write-Host ""

    $useEmbeddingApi = Read-Host "Use Embedding API? (y/N)"
    if ($useEmbeddingApi -match '^(y|yes)$') {
        $embeddingModel = Read-Host "Embedding model (e.g. BAAI/bge-m3)"
        $embeddingBase = Read-Host "Embedding API base URL"
        $embeddingKey = Read-Host "Embedding API key"

        Set-EnvVar -Key "EMBEDDING_MODEL" -Value $embeddingModel
        Set-EnvVar -Key "EMBEDDING_BASE_URL" -Value $embeddingBase
        Set-EnvVar -Key "EMBEDDING_API_KEY" -Value $embeddingKey
        $script:EMBEDDING_DOWNLOAD_REQUIRED = $false
        Log-Success "Embedding configured with API."
    }
    else {
        $downloadEmbedding = Read-Host "No Embedding API. Download local embedding model BAAI/bge-m3? (Y/n)"
        if ($downloadEmbedding -notmatch '^(n|no)$') {
            Set-EnvVar -Key "EMBEDDING_MODEL" -Value "BAAI/bge-m3"
            Set-EnvVar -Key "EMBEDDING_BASE_URL" -Value ""
            Set-EnvVar -Key "EMBEDDING_API_KEY" -Value ""
            $script:EMBEDDING_DOWNLOAD_REQUIRED = $true
            Log-Info "Will download local embedding model."
        }
        else {
            Set-EnvVar -Key "EMBEDDING_MODEL" -Value ""
            Set-EnvVar -Key "EMBEDDING_BASE_URL" -Value ""
            Set-EnvVar -Key "EMBEDDING_API_KEY" -Value ""
            Set-EnvVar -Key "EMBEDDING_WEIGHT" -Value "0"
            Set-EnvVar -Key "BM25_WEIGHT" -Value "1"
            $script:EMBEDDING_DOWNLOAD_REQUIRED = $false
            Log-Info "Embedding disabled. BM25-only retrieval enabled (BM25_WEIGHT=1, EMBEDDING_WEIGHT=0)."
        }
    }

    $useRerankApi = Read-Host "Use Rerank API? (y/N)"
    if ($useRerankApi -match '^(y|yes)$') {
        $rerankModel = Read-Host "Rerank model (e.g. BAAI/bge-reranker-v2-m3)"
        $rerankBase = Read-Host "Rerank API base URL"
        $rerankKey = Read-Host "Rerank API key"

        Set-EnvVar -Key "RERANKER_ENABLED" -Value "true"
        Set-EnvVar -Key "RERANKER_MODEL" -Value $rerankModel
        Set-EnvVar -Key "RERANKER_BASE_URL" -Value $rerankBase
        Set-EnvVar -Key "RERANKER_API_KEY" -Value $rerankKey
        $script:RERANK_DOWNLOAD_REQUIRED = $false
        Log-Success "Reranker configured with API."
    }
    else {
        $downloadRerank = Read-Host "No Rerank API. Download local rerank model BAAI/bge-reranker-v2-m3? (y/N)"
        if ($downloadRerank -match '^(y|yes)$') {
            Set-EnvVar -Key "RERANKER_ENABLED" -Value "true"
            Set-EnvVar -Key "RERANKER_MODEL" -Value "BAAI/bge-reranker-v2-m3"
            Set-EnvVar -Key "RERANKER_BASE_URL" -Value ""
            Set-EnvVar -Key "RERANKER_API_KEY" -Value ""
            $script:RERANK_DOWNLOAD_REQUIRED = $true
            Log-Info "Will download local rerank model."
        }
        else {
            Set-EnvVar -Key "RERANKER_ENABLED" -Value "false"
            Set-EnvVar -Key "RERANKER_MODEL" -Value ""
            Set-EnvVar -Key "RERANKER_BASE_URL" -Value ""
            Set-EnvVar -Key "RERANKER_API_KEY" -Value ""
            $script:RERANK_DOWNLOAD_REQUIRED = $false
            Log-Info "Rerank disabled."
        }
    }

    Log-Success "Retrieval model configuration written to .env"
}

function Show-Banner {
    Write-Host ""
    Write-Host "========================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "   __  __                           _          ____"                        -ForegroundColor Cyan
    Write-Host "  |  \/  | ___ _ __ ___   ___ _ __ | |_ ___   / ___|"                      -ForegroundColor Cyan
    Write-Host "  | |\/| |/ _ \ '_ `` _ \ / _ \ '_ \| __/ _ \  \___ \"                     -ForegroundColor Cyan
    Write-Host "  | |  | |  __/ | | | | |  __/ | | | ||  __/   ___) |"                     -ForegroundColor Cyan
    Write-Host "  |_|  |_|\___|_| |_| |_|\___|_| |_|\__\___|  |____/"                      -ForegroundColor Cyan
    Write-Host ""
    Write-Host "                           Memento-S"                                       -ForegroundColor Cyan
    Write-Host "                   Install (Windows, Local Source)"                         -ForegroundColor Cyan
    Write-Host ""
    Write-Host "========================================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Test-Tmux {
    if (Test-CommandExists "tmux") {
        $ver = & tmux -V 2>&1
        Log-Success "tmux: $ver"
        return
    }
    Log-Warn "tmux is not installed. Memento-S TUI may require tmux."
    Log-Warn "On Windows you can install tmux via MSYS2: pacman -S tmux"
    Log-Warn "Or use WSL: sudo apt install tmux"
}

function Install-Uv {
    if (Test-CommandExists "uv") {
        $ver = & uv --version 2>&1
        Log-Success "uv: $ver"
        return
    }

    Log-Info "Installing uv..."
    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    }
    catch {
        Log-Error "Failed to install uv: $_"
        Log-Error "Please install manually: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    }

    $env:Path = "$env:USERPROFILE\.local\bin;$env:USERPROFILE\.cargo\bin;$env:Path"

    if (Test-CommandExists "uv") {
        $ver = & uv --version 2>&1
        Log-Success "uv installed: $ver"
    }
    else {
        Log-Error "Failed to install uv. Please install manually: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    }
}

function Install-Python312 {
    & uv python find $PYTHON_VERSION 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Log-Success "Python $PYTHON_VERSION available via uv"
        return
    }

    Log-Info "Installing Python $PYTHON_VERSION via uv..."
    & uv python install $PYTHON_VERSION
    if ($LASTEXITCODE -ne 0) {
        Log-Error "Failed to install Python $PYTHON_VERSION."
        exit 1
    }
    Log-Success "Python $PYTHON_VERSION installed."
}

function Install-VenvAndDeps {
    if (-not (Test-Path $PROJECT_ROOT)) {
        Log-Error "Project root not found: $PROJECT_ROOT"
        exit 1
    }

    Write-Host ""
    Write-Host "===============================================================" -ForegroundColor Cyan
    Write-Host "            Creating .venv and Installing Dependencies          " -ForegroundColor Cyan
    Write-Host "===============================================================" -ForegroundColor Cyan
    Write-Host ""

    Push-Location $PROJECT_ROOT
    try {
        Log-Info "Creating .venv with Python $PYTHON_VERSION..."
        & uv venv .venv --python $PYTHON_VERSION
        if ($LASTEXITCODE -ne 0) {
            Log-Error "Failed to create .venv."
            exit 1
        }
        Log-Success ".venv created."

        Log-Info "Installing dependencies from requirements.txt..."
        & uv pip install --python ".venv\Scripts\python.exe" -r requirements.txt
        if ($LASTEXITCODE -ne 0) {
            Log-Error "Failed to install Python dependencies."
            exit 1
        }

        Log-Info "Installing local CLI entry (memento)..."
        & uv pip install --python ".venv\Scripts\python.exe" -e .
        if ($LASTEXITCODE -ne 0) {
            Log-Error "Failed to install local project package for memento command."
            exit 1
        }
        Log-Success "Dependencies and local CLI installed."
    }
    finally {
        Pop-Location
    }
}

function Install-NodeJs {
    $hasNode = Test-CommandExists "node"
    $hasNpm = Test-CommandExists "npm"

    if ($hasNode -and $hasNpm) {
        $nodeVer = & node --version 2>&1
        $npmVer = & npm --version 2>&1
        Log-Success "node: $nodeVer"
        Log-Success "npm: $npmVer"
        return $true
    }

    if ($hasNode -and -not $hasNpm) {
        Log-Warn "node exists but npm not found in PATH."
    }
    elseif (-not $hasNode -and $hasNpm) {
        Log-Warn "npm exists but node not found in PATH."
    }
    else {
        Log-Info "node/npm not found."
    }

    Log-Warn "Please install Node.js manually from https://nodejs.org/"
    Log-Warn "Or use nvm-windows: https://github.com/coreybutler/nvm-windows"
    return $false
}

function Install-OpenSkills {
    if (-not (Test-CommandExists "npm")) {
        Log-Warn "npm not available. Skipping openskills installation."
        Log-Warn "To install later: npm install -g openskills && openskills sync -y"
        return
    }

    if (-not (Test-CommandExists "openskills")) {
        Log-Info "Installing openskills..."
        & npm install -g openskills 2>$null
        if ($LASTEXITCODE -ne 0) {
            Log-Warn "Failed to install openskills. Skipping skills."
            return
        }
    }

    Log-Info "Installing skills..."
    Push-Location $PROJECT_ROOT
    try {
        $skillDirs = Get-ChildItem -Path "skills" -Directory -ErrorAction SilentlyContinue
        foreach ($d in $skillDirs) {
            $name = $d.Name
            if (Test-Path ".agent\skills\$name") {
                & openskills update $name 2>$null
            }
            else {
                & openskills install ".\skills\$name" --universal --yes 2>$null
            }
        }
        & openskills sync -y 2>$null
        Log-Success "Skills installed."
    }
    finally {
        Pop-Location
    }
}

function Install-OptionalModels {
    $py = Join-Path $PROJECT_ROOT ".venv\Scripts\python.exe"
    if (-not (Test-Path $py)) {
        Log-Warn ".venv not found, skipping model download."
        return
    }

    if ($script:EMBEDDING_DOWNLOAD_REQUIRED) {
        Log-Info "Downloading BAAI/bge-m3 embedding model (may take a while)..."
        $env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
        & $py -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
        if ($LASTEXITCODE -eq 0) {
            Log-Success "BAAI/bge-m3 downloaded."
        }
        else {
            Log-Warn "Embedding model download failed (network or disk). You can retry later."
        }
    }
    else {
        Log-Info "Skipping embedding model download."
    }

    if ($script:RERANK_DOWNLOAD_REQUIRED) {
        Log-Info "Downloading BAAI/bge-reranker-v2-m3 model (may take a while)..."
        $env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
        & $py -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-v2-m3')"
        if ($LASTEXITCODE -eq 0) {
            Log-Success "BAAI/bge-reranker-v2-m3 downloaded."
        }
        else {
            Log-Warn "Rerank model download failed (network or disk). You can retry later."
        }
    }
    else {
        Log-Info "Skipping rerank model download."
    }
}

function Invoke-PostInstallConfig {
    $py = Join-Path $PROJECT_ROOT ".venv\Scripts\python.exe"
    if (-not (Test-Path $py)) {
        Log-Warn ".venv not found, skipping post-install config commands."
        return
    }

    $cliInit = Join-Path $PROJECT_ROOT "cli\__init__.py"
    if (-not (Test-Path $cliInit)) {
        Log-Warn "cli module not found in $PROJECT_ROOT, skipping config commands."
        return
    }

    Log-Info "Running post-install config commands..."
    Push-Location $PROJECT_ROOT
    try {
        & $py -m cli.main config list
        if ($LASTEXITCODE -ne 0) {
            Log-Warn "Failed to run 'python -m cli.main config list'. You can run it manually later."
        }

        & $py -m cli.main config
        if ($LASTEXITCODE -ne 0) {
            Log-Warn "Failed to run 'python -m cli.main config'. You can run it manually later."
        }
    }
    finally {
        Pop-Location
    }
}

function Initialize-VectorDb {
    $py = Join-Path $PROJECT_ROOT ".venv\Scripts\python.exe"
    if (-not (Test-Path $py)) {
        Log-Warn ".venv not found, skipping vector DB init."
        return
    }

    $cliInit = Join-Path $PROJECT_ROOT "cli\__init__.py"
    if (-not (Test-Path $cliInit)) {
        Log-Warn "cli module not found, skipping vector DB init."
        return
    }

    Log-Info "Initializing vector database (running agent until ready)..."
    Push-Location $PROJECT_ROOT
    try {
        $readyPattern = '^\s*You\s*>'
        $timeoutSec = 180

        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $py
        $psi.Arguments = "-m cli.main agent"
        $psi.UseShellExecute = $false
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.CreateNoWindow = $true

        $proc = [System.Diagnostics.Process]::Start($psi)
        $deadline = (Get-Date).AddSeconds($timeoutSec)
        $isReady = $false

        while (-not $proc.HasExited -and (Get-Date) -lt $deadline) {
            while (-not $proc.StandardOutput.EndOfStream) {
                $line = $proc.StandardOutput.ReadLine()
                if ($null -eq $line) { break }
                Write-Host $line
                if ($line -match $readyPattern) {
                    $isReady = $true
                    break
                }
            }

            while (-not $proc.StandardError.EndOfStream) {
                $errLine = $proc.StandardError.ReadLine()
                if ($null -ne $errLine -and $errLine -ne "") {
                    Log-Warn "agent: $errLine"
                }
            }

            if ($isReady) { break }
            Start-Sleep -Milliseconds 200
        }

        if (-not $proc.HasExited) {
            try {
                $proc.Kill()
                $proc.WaitForExit(5000) | Out-Null
            }
            catch {
                Log-Warn "Failed to terminate temporary agent process cleanly."
            }
        }

        if ($isReady) {
            Log-Success "Vector database initialized."
        }
        else {
            Log-Warn "Vector DB init timed out or ready prompt not detected. You can initialize it later by running '.venv\Scripts\python.exe -m cli.main agent'."
        }
    }
    catch {
        Log-Warn "Vector DB init failed: $_. You can initialize it later."
    }
    finally {
        Pop-Location
    }
}

function Show-Success {
    Write-Host ""
    Write-Host "===============================================================" -ForegroundColor Green
    Write-Host "                 Installation Complete!                         " -ForegroundColor Green
    Write-Host "===============================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Install directory: " -NoNewline; Write-Host "$PROJECT_ROOT" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  To start Memento-S:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "    cd $PROJECT_ROOT; .venv\Scripts\memento.exe" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Other commands:" -ForegroundColor Cyan
    Write-Host "    .venv\Scripts\python.exe tui.py doctor   - Check configuration"
    Write-Host "    .venv\Scripts\python.exe tui.py config   - Show current config"
    Write-Host "    .venv\Scripts\python.exe tui.py --help   - Show all commands"
    Write-Host ""
}

function Main {
    Show-Banner

    if (-not (Test-Path $PROJECT_ROOT) -or -not (Test-Path (Join-Path $PROJECT_ROOT "requirements.txt"))) {
        Log-Error "Project root not found or invalid (no requirements.txt): $PROJECT_ROOT"
        exit 1
    }

    Test-Tmux

    Install-Uv
    Install-Python312

    Install-VenvAndDeps

    Install-NodeJs
    Install-OpenSkills

    Configure-RetrievalModels
    Install-OptionalModels

    Invoke-PostInstallConfig

    Initialize-VectorDb

    Show-Success
}

Main
