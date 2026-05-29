# run_all_models.ps1 — Benchmark-Runner (Windows)
# Aufruf: .\run_all_models.ps1 [-Repeat N] [-OnlyA|-OnlyB] [-SkipPull]
#         [-Benchmark PFAD] [-ResultsDir PFAD]
# Voraussetzungen: Ollama (11434), Fuseki (3030), pip install -r requirements.txt

param(
    [switch]$SkipPull,
    [switch]$OnlyA,
    [switch]$OnlyB
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$MODELS = @(
    "mistral",
    "qwen2.5:7b-instruct",
    "llama3.1:8b-instruct-q4_K_M",
    "qwen2.5-coder:7b-instruct",
    "gemma2:9b",
    "deepseek-r1:7b"
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

# Benchmark-Laeufe
Sep
Log "Starte Benchmark-Laeufe..."

foreach ($model in $MODELS) {
    Sep
    Log "Modell: $model"

    if ($RunA) {
        Log "  System A (LLM-only)..."
        python system_a.py --model $model
        if ($LASTEXITCODE -ne 0) { Write-Error "System A fehlgeschlagen fuer $model"; exit 1 }
        Log "  System A fertig."
    }

    if ($RunB) {
        Log "  System B (LLM+KG)..."
        python system_b.py --model $model
        if ($LASTEXITCODE -ne 0) { Write-Error "System B fehlgeschlagen fuer $model"; exit 1 }
        Log "  System B fertig."
    }
}

# Auswertung
Sep
Log "Starte Auswertung..."
python eval.py --all --compare

Sep
Log "Fertig. Ergebnisse in thesis-code/results/"
Log "Eval-JSONs (eval_*.json) fuer Kapitel 5.3 verwenden."
Write-Host ""
