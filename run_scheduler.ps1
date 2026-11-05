# run_scheduler.ps1 — EvoRAG Step 1: News Scheduler / Knowledge-Base Updater
# Usage:
#   .\run_scheduler.ps1            # Starts the continuous 3-day APScheduler process
#   .\run_scheduler.ps1 -Once      # Runs check/ingestion once and exits
#   .\run_scheduler.ps1 -Force     # Forces immediate fetch and incremental update
#   .\run_scheduler.ps1 -CheckOnly # Only checks if an update is due

param (
    [switch]$Force,
    [switch]$Once,
    [switch]$CheckOnly
)

$ScriptDir = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   EvoRAG - News Ingestion & Knowledge Scheduler (Phase 6)   " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Directory : $ScriptDir" -ForegroundColor DarkGray
Write-Host "  Interval  : Every 3 days" -ForegroundColor DarkGray
Write-Host "  Model     : sentence-transformers/all-MiniLM-L6-v2" -ForegroundColor DarkGray
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Verify Python is available
$PythonCmd = "python"
if (-not (Get-Command $PythonCmd -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] 'python' was not found on PATH. Please ensure Python 3.12 is installed." -ForegroundColor Red
    exit 1
}

$pyVer = & $PythonCmd --version 2>&1
Write-Host "[INFO] Using Python environment: $pyVer" -ForegroundColor Green

# Determine arguments
$argsList = @("knowledge_scheduler.py")
if ($Force) {
    $argsList += "--force"
    Write-Host "[MODE] Forcing immediate fetch and incremental update..." -ForegroundColor Yellow
} elseif ($Once) {
    $argsList += "--once"
    Write-Host "[MODE] Running one-time ingestion check..." -ForegroundColor Yellow
} elseif ($CheckOnly) {
    $argsList += "--check-only"
    Write-Host "[MODE] Checking if index update is due..." -ForegroundColor Yellow
} else {
    Write-Host "[MODE] Starting APScheduler (3-day cycle + startup check)..." -ForegroundColor Green
}

Write-Host ""

# Run scheduler
& $PythonCmd $argsList

