<#
AEROVIGIL one-command targets for Windows (the same targets as the Makefile).

Usage (from the repo folder, in PowerShell):
    .\make.ps1 demo        open the offline demo dashboard (builds its data the first time)
    .\make.ps1 demo-data   (re)build the demo's precomputed scenarios (~10 min)

The demo runs on this computer only (http://localhost:8501) and needs no internet.
Close it with Ctrl+C in this window.
#>
param([Parameter(Position = 0)][string]$Target = "help")

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }
$Url = "http://localhost:8501"

function Invoke-DemoData {
    & $Py "src\aerovigil\demo\build.py"
    if ($LASTEXITCODE -ne 0) { throw "demo data build failed" }
}

function Start-Demo {
    if (-not (Test-Path "data\processed\demo\scenarios.parquet")) { Invoke-DemoData }
    # headless: no first-run e-mail question and no usage statistics; the browser is
    # opened here once the app answers
    $app = Start-Process -FilePath $Py -NoNewWindow -PassThru -ArgumentList @(
        "-m", "streamlit", "run", "app\app.py", "--server.headless", "true",
        "--server.port", "8501", "--browser.gatherUsageStats", "false")
    $ready = $false
    for ($i = 0; $i -lt 60 -and -not $app.HasExited; $i++) {
        Start-Sleep -Seconds 1
        try {
            $null = Invoke-WebRequest -UseBasicParsing "$Url/_stcore/health" -TimeoutSec 2
            $ready = $true
            break
        } catch { }
    }
    if (-not $ready) { throw "the demo did not start; see the messages above" }
    Write-Output "AEROVIGIL demo running at $Url  (Ctrl+C to stop)"
    Start-Process $Url
    try { Wait-Process -Id $app.Id } finally {
        if (-not $app.HasExited) { Stop-Process -Id $app.Id -Force }
    }
}

switch ($Target) {
    "demo-data" { Invoke-DemoData }
    "demo" { Start-Demo }
    default {
        Write-Output "Targets:"
        Write-Output "  .\make.ps1 demo        open the offline demo dashboard"
        Write-Output "  .\make.ps1 demo-data   (re)build the demo's precomputed scenarios"
    }
}
