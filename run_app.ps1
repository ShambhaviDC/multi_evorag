# run_app.ps1 — EvoRAG Step 2: Start the EvoRAG Application
# Starts Ollama (if installed & not running), starts the FastAPI backend, and opens the frontend.
# Usage:
#   .\run_app.ps1

$ScriptDir = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "            EvoRAG Application Launcher (Phase 6)           " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Directory : $ScriptDir" -ForegroundColor DarkGray
Write-Host "  Host/Port : http://localhost:8000/" -ForegroundColor DarkGray
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Verify Python
$PythonCmd = "python"
if (-not (Get-Command $PythonCmd -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] 'python' was not found on PATH. Please ensure Python 3.12 is installed." -ForegroundColor Red
    exit 1
}

# 2. Check FAISS Index Readiness
$IndexFile = Join-Path $ScriptDir "index\evorag.faiss"
$MetaFile  = Join-Path $ScriptDir "index\evorag_meta.json"
if (-not (Test-Path $IndexFile) -or -not (Test-Path $MetaFile)) {
    Write-Host "[WARNING] FAISS index or metadata was not found in index/." -ForegroundColor Yellow
    Write-Host "          Running initial ingestion check via run_scheduler.ps1 -Once..." -ForegroundColor Yellow
    & $PythonCmd knowledge_scheduler.py --once
}

# 3. Check / Start Ollama
$OllamaProcess = Get-Process ollama -ErrorAction SilentlyContinue
if ($OllamaProcess) {
    Write-Host "[OK]   Ollama is already running (PID: $($OllamaProcess[0].Id))." -ForegroundColor Green
} else {
    $OllamaExe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
    if (Test-Path $OllamaExe) {
        Write-Host "[INFO] Starting Ollama in background ($OllamaExe serve)..." -ForegroundColor Yellow
        Start-Process -FilePath $OllamaExe -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 2
        Write-Host "[OK]   Ollama background process started." -ForegroundColor Green
    } else {
        Write-Host "[INFO] Ollama executable not found locally; continuing (Gemini fallback active)." -ForegroundColor DarkGray
    }
}

# 4. Open Browser to Frontend
Write-Host "[INFO] Launching browser to http://localhost:8000/ ..." -ForegroundColor Green
Start-Process "http://localhost:8000/"

# 5. Start FastAPI Backend
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  EvoRAG Backend Server is starting on port 8000" -ForegroundColor Green
Write-Host "  Press Ctrl+C to stop the server." -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""

& $PythonCmd api.py

