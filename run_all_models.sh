#!/usr/bin/env bash
# run_all_models.sh — Benchmark-Runner (Linux/macOS)
# Aufruf: bash run_all_models.sh [--repeat N] [--only-a|--only-b] [--skip-pull]
#         [--benchmark PFAD] [--results-dir PFAD]
# Voraussetzungen: Ollama (11434), Fuseki (3030), pip install -r requirements.txt

set -euo pipefail
cd "$(dirname "$0")"

# Konfiguration
MODELS=("llama3.1:8b-instruct-q4_K_M")

SKIP_PULL=false
RUN_A=true
RUN_B=true
REPEAT=3
BENCHMARK="data/benchmark_v1.yaml"
# Ziel: aktueller Lauf; eingefrorene v1.0/v1.1-Ordner nicht ueberschreiben
RESULTS_DIR="results/main_benchmark_v1.2"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-pull)   SKIP_PULL=true; shift ;;
    --only-a)      RUN_B=false; shift ;;
    --only-b)      RUN_A=false; shift ;;
    --repeat)      REPEAT="$2"; shift 2 ;;
    --benchmark)   BENCHMARK="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    *) echo "Unbekannte Option: $1"; exit 1 ;;
  esac
done

# Hilfsfunktionen
log() { echo "[$(date '+%H:%M:%S')] $*"; }
sep() { echo ""; echo "====================================================================="; }

check_ollama() {
  if ! curl -sf http://localhost:11434/ > /dev/null 2>&1; then
    echo "FEHLER: Ollama ist nicht erreichbar (http://localhost:11434)."
    echo "        Starte Ollama mit: ollama serve"
    exit 1
  fi
}

check_fuseki() {
  if ! curl -sf http://localhost:3030/ > /dev/null 2>&1; then
    echo "FEHLER: Fuseki ist nicht erreichbar (http://localhost:3030)."
    echo "        Starte Fuseki z.B. mit: docker start fuseki (siehe fuseki_setup.md)"
    exit 1
  fi
}

# Vorbedingungen
sep
log "Konfiguration:"
log "  Benchmark  : $BENCHMARK"
log "  ResultsDir : $RESULTS_DIR"
log "  Repeat     : $REPEAT"
log "  Modelle    : ${MODELS[*]}"
log "  System A   : $RUN_A  |  System B: $RUN_B"
sep
log "Pruefe Dienste..."
check_ollama
if $RUN_B; then check_fuseki; fi
log "OK: alle benoetigten Dienste laufen."

mkdir -p "$RESULTS_DIR"

# Modelle ziehen
if ! $SKIP_PULL; then
  sep
  log "Ziehe Modelle..."
  for model in "${MODELS[@]}"; do
    log "  ollama pull $model"
    ollama pull "$model"
    log "  -> OK: $model"
  done
else
  log "Modell-Pull uebersprungen (--skip-pull)."
fi

# Benchmark-Laeufe (je Modell REPEAT-mal wiederholen)
sep
log "Starte Benchmark-Laeufe ($REPEAT Wiederholungen pro Modell)..."

for model in "${MODELS[@]}"; do
  sep
  log "Modell: $model"

  for ((run=1; run<=REPEAT; run++)); do
    sep
    log "  Wiederholung $run/$REPEAT"

    if $RUN_A; then
      log "    System A (LLM-only)..."
      python3 system_a.py --model "$model" --benchmark "$BENCHMARK" --results-dir "$RESULTS_DIR"
      log "    System A fertig."
    fi

    if $RUN_B; then
      log "    System B (LLM+KG)..."
      python3 system_b.py --model "$model" --benchmark "$BENCHMARK" --results-dir "$RESULTS_DIR"
      log "    System B fertig."
    fi

    # Kurze Pause gegen Timestamp-Kollisionen
    if (( run < REPEAT )); then sleep 2; fi
  done
done

# Auswertung
sep
log "Starte Auswertung..."
python3 eval.py --dir "$RESULTS_DIR" --compare

sep
log "Fertig. Ergebnisse in $RESULTS_DIR"
log "Eval-JSONs (eval_*.json) fuer Kapitel 5.3 verwenden."
echo ""
