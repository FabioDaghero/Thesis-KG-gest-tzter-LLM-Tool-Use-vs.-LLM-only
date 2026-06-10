# run_all_models.ps1 — Benchmark-Runner (Windows)
# Aufruf: .\run_all_models.ps1 [-Repeat N] [-OnlyA|-OnlyB] [-SkipPull]
#         [-Benchmark PFAD] [-ResultsDir PFAD]
# Voraussetzungen: Ollama (11434), Fuseki (3030), pip install -r requirements.txt

param(
    [switch]$SkipPull,
    [switch]$OnlyA,
    [switch]$OnlyB,
    [int]$Repeat = 3,
    [string]$Benchmark = "data/benchmark_v1.yaml",
    [string]$ResultsDir = "results/main_benchmark"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$MODELS = @(
    "llama3.1:8b-instruct-q4_K_M"
)
$RunA = -not $OnlyB
$RunB = -not $OnlyA

function Log($msg) {
    Write-Host "[$([DateTime]::Now.ToString('HH:mm:ss'))] $msg" -ForegroundColor Cyan
}

function Sep {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor DarkGray
}

# Dienste pruefen
Sep
Log "Konfiguration:"
Log "  Benchmark  : $Benchmark"
Log "  ResultsDir : $ResultsDir"
Log "  Repeat     : $Repeat"
Log "  Modelle    : $($MODELS -join ', ')"
Log "  System A   : $RunA  |  System B: $RunB"
Sep
Log "Pruefe Dienste..."

try {
    $null = Invoke-RestMethod "http://localhost:11434/" -TimeoutSec 3
    Log "OK: Ollama erreichbar."
} catch {
    Write-Error "Ollama ist nicht erreichbar (http://localhost:11434). Starte: ollama serve"
    exit 1
}

if ($RunB) {
    try {
        $null = Invoke-RestMethod "http://localhost:3030/" -TimeoutSec 3
        Log "OK: Fuseki erreichbar."
    } catch {
        Write-Error "Fuseki ist nicht erreichbar (http://localhost:3030). Starte den Docker-Container."
        exit 1
    }
}

# Ausgabeordner erstellen
if (-not (Test-Path $ResultsDir)) {
    New-Item -ItemType Directory -Path $ResultsDir | Out-Null
    Log "Ausgabeordner erstellt: $ResultsDir"
}

# Modelle ziehen
if (-not $SkipPull) {
    Sep
    Log "Ziehe Modelle (kann einige Minuten dauern)..."
    foreach ($model in $MODELS) {
        Log "  ollama pull $model"
        ollama pull $model
        if ($LASTEXITCODE -ne 0) { Write-Error "ollama pull fehlgeschlagen fuer $model"; exit 1 }
        Log "  -> OK: $model"
    }
} else {
    Log "Modell-Pull uebersprungen (-SkipPull)."
}

# Benchmark-Laeufe (je Modell $Repeat-mal wiederholen)
Sep
Log "Starte Benchmark-Laeufe ($Repeat Wiederholungen pro Modell)..."

foreach ($model in $MODELS) {
    Sep
    Log "Modell: $model"

    for ($run = 1; $run -le $Repeat; $run++) {
        Sep
        Log "  Wiederholung $run/$Repeat"

        if ($RunA) {
            Log "    System A (LLM-only)..."
            python system_a.py `
                --model $model `
                --benchmark $Benchmark `
                --results-dir $ResultsDir
            if ($LASTEXITCODE -ne 0) {
                Write-Error "System A fehlgeschlagen (Modell=$model, Lauf=$run)"
                exit 1
            }
            Log "    System A fertig."
        }

        if ($RunB) {
            Log "    System B (LLM+KG)..."
            python system_b.py `
                --model $model `
                --benchmark $Benchmark `
                --results-dir $ResultsDir
            if ($LASTEXITCODE -ne 0) {
                Write-Error "System B fehlgeschlagen (Modell=$model, Lauf=$run)"
                exit 1
            }
            Log "    System B fertig."
        }

        # Pause gegen Timestamp-Kollisionen
        if ($run -lt $Repeat) { Start-Sleep -Seconds 2 }
    }
}

# Auswertung
Sep
Log "Starte Auswertung..."
python eval.py --dir $ResultsDir --compare

Sep
Log "Fertig. Ergebnisse in $ResultsDir"
Log "Eval-JSONs (eval_*.json) fuer Kapitel 5.3 verwenden."
Write-Host ""
