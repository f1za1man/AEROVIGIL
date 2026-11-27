<#
AEROVIGIL one-command targets for Windows (the same targets as the Makefile).

Usage (from the repo folder, in PowerShell):
    .\make.ps1 help        list the targets
    .\make.ps1 download    fetch and check the dataset (2.6 GB, not redistributed)
    .\make.ps1 data        Modules 02-04: base tables, features, splits, attacks, windows
    .\make.ps1 train       Modules 05-10: every detector, trained and calibrated
    .\make.ps1 eval        Steps B-G: protocols, Stealth Frontier, explanations, edge,
                           fallback, then every table (~15 h)
    .\make.ps1 tables      rebuild every results table from saved results (minutes)
    .\make.ps1 figures     redraw the Stealth Frontier and fallback figures
    .\make.ps1 test        black, ruff and pytest
    .\make.ps1 smoke       tests plus tables and headline figure (works on a fresh clone)
    .\make.ps1 demo        open the offline demo dashboard (builds its data the first time)

Order for a full rebuild: data -> train -> eval (each reads the previous one's output).
The demo runs on this computer only (http://localhost:8501); close it with Ctrl+C.
#>
param([Parameter(Position = 0)][string]$Target = "help")

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }
$Url = "http://localhost:8501"
$Detectors = @("kalman", "velocity", "fused", "cusum", "consistency", "random_forest",
               "lstm_ae", "lstm_supervised")

function Invoke-Py {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PyArgs)
    Write-Output ">> python $($PyArgs -join ' ')"
    & $Py @PyArgs
    if ($LASTEXITCODE -ne 0) { throw "failed: python $($PyArgs -join ' ')" }
}

function Invoke-Download { Invoke-Py "src\aerovigil\data\download.py" }

function Invoke-Data {
    Invoke-Download
    Invoke-Py "src\aerovigil\data\inspect_raw_files.py"
    Invoke-Py -m aerovigil.data.inspect_dataset
    Invoke-Py "src\aerovigil\preprocessing\clean_flights.py"
    Invoke-Py "src\aerovigil\preprocessing\split_flights.py"
    Invoke-Py "src\aerovigil\data\check_raw_coverage.py"
    Invoke-Py "src\aerovigil\data\check_attack_end.py"
    Invoke-Py "src\aerovigil\data\build_raw_base.py"
    Invoke-Py "src\aerovigil\data\check_base_quality.py"
    Invoke-Py -m aerovigil.features.build_features_v2
    Invoke-Py -m aerovigil.preprocessing.normalize_features_v2
    Invoke-Py -m aerovigil.windowing.make_windows_v2
    Invoke-Py "src\aerovigil\windowing\window_separability.py"
    Invoke-Py "src\aerovigil\attacks\generate_v2.py"
    Invoke-Py "src\aerovigil\attacks\features_v2_m04.py"
    Invoke-Py "src\aerovigil\attacks\attack_separability.py"
    Invoke-Py "src\aerovigil\baselines\build_window_tables_v2.py" --sources "orig,syn"
    Invoke-Py "src\aerovigil\lstm\build_sequences_v2.py"
}

function Invoke-Train {
    foreach ($s in "kalman\calibrate_imu", "kalman\tune_noise", "baselines\rescore_v2",
                   "kalman\run_filter", "kalman\evaluate_detectors", "adaptive\search",
                   "lstm\train_v2", "lstm\evaluate_v2", "lstm\sequential_v2",
                   "consistency\evaluate", "fusion\signals", "fusion\fuse",
                   "fusion\adaptive", "fusion\threat_model") {
        Invoke-Py "src\aerovigil\$s.py"
    }
}

function Invoke-Eval {
    Invoke-Py "src\aerovigil\baselines\all_models_v2.py"
    Invoke-Py "src\aerovigil\baselines\all_models_v2.py" --full
    Invoke-Py "src\aerovigil\evaluation\physics_protocols.py"
    Invoke-Py "src\aerovigil\evaluation\lstm_protocols.py"
    Invoke-Py "src\aerovigil\evaluation\ablations.py"
    Invoke-Py "src\aerovigil\stealth\frontier.py"
    foreach ($d in $Detectors) { Invoke-Py "src\aerovigil\stealth\adaptive.py" --detector $d }
    Invoke-Py "src\aerovigil\stealth\adaptive.py" --summarise
    Invoke-Py "src\aerovigil\stealth\whitebox.py"
    Invoke-Py "src\aerovigil\stealth\harden.py" --model random_forest
    Invoke-Py "src\aerovigil\stealth\harden.py" --model lstm_supervised
    Invoke-Py "src\aerovigil\explain\run.py"
    Invoke-Py "src\aerovigil\edge\benchmark.py"
    Invoke-Py "src\aerovigil\mitigation\run.py"
    Invoke-Tables
}

function Invoke-Tables { Invoke-Py "src\aerovigil\evaluation\run_all.py" }

function Invoke-Figures {
    Invoke-Py "src\aerovigil\stealth\figures.py"
    Invoke-Py "src\aerovigil\mitigation\run.py" --figures
}

function Invoke-Test {
    Invoke-Py -m black --check --line-length 88 src\aerovigil tests app
    Invoke-Py -m ruff check src\aerovigil tests app
    Invoke-Py -m pytest tests -q
}

function Invoke-DemoData { Invoke-Py "src\aerovigil\demo\build.py" }

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
    "download" { Invoke-Download }
    "data" { Invoke-Data }
    "train" { Invoke-Train }
    "eval" { Invoke-Eval }
    "tables" { Invoke-Tables }
    "figures" { Invoke-Figures }
    "test" { Invoke-Test }
    "smoke" {
        # a fresh clone has no dataset: data tests skip instead of failing
        $env:AEROVIGIL_ALLOW_MISSING_DATA = "1"
        $t = Get-Date
        Invoke-Test
        Invoke-Tables
        Invoke-Py "src\aerovigil\stealth\figures.py"
        Write-Output ("smoke finished in {0:N1} min" -f ((Get-Date) - $t).TotalMinutes)
    }
    "demo-data" { Invoke-DemoData }
    "demo" { Start-Demo }
    default {
        # the comment block at the top of this file lists the targets
        $text = Get-Content $PSCommandPath -Raw
        $text.Substring(2, $text.IndexOf("#>") - 2).Trim() | Write-Output
    }
}
