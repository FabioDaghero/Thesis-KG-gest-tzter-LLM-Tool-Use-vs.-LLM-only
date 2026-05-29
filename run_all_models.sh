#!/usr/bin/env bash
# run_all_models.sh — Benchmark-Runner (Linux/macOS)
# Aufruf: bash run_all_models.sh [--repeat N] [--only-a|--only-b] [--skip-pull]
#         [--benchmark PFAD] [--results-dir PFAD]
# Voraussetzungen: Ollama (11434), Fuseki (3030), pip install -r requirements.txt

set -euo pipefail
cd "$(dirname "$0")"

# Konfiguration
MISTRAL="mistral"
QWEN="qwen2.5:7b-instruct"
LLAMA="llama3.1:8b-instruct-q4_K_M"
CODER="qwen2.5-coder:7b-instruct"
GEMMA="gemma2:9b"
DEEPSEEK="deepseek-r1:7b"

MODELS=("$MISTRAL" "$QWEN" "$LLAMA" "$CODER" "$GEMMA" "$DEEPSEEK")

SKIP_PULL=false
RUN_A=true
RUN_B=true

for arg in "$@"; do
  case "$arg" in
    --skip-pull) SKIP_PULL=true ;;
    --only-a)    RUN_B=false ;;
    --only-b)    RUN_A=false ;;
  esac
done

# Hilfsfunktionen
log() { echo "[$(date '+%H:%M:%S')] $*"; }
sep() { echo ""; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }

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
    echo "        Starte Fuseki z.B. mit: docker start fuseki"
    echo "        (oder fuelle fuseki_setup.md)"
    exit 1
  fi
}

# Vorbedingungen pruefen
sep
log "Pruefe Dienste..."
check_ollama
if $RUN_B; then check_fuseki; fi
log "OK: alle benoetigten Dienste laufen."

# Modelle ziehen
if ! $SKIP_PULL; then
  sep
  log "Ziehe Modelle (kann einige Minuten dauern)..."
  for model in "${MODELS[@]}"; do
    log "  ollama pull $model"
    ollama pull "$model"
    log "  -> OK: $model"
  done
else
  log "Modell-Pull uebersprungen (--skip-pull)."
fi

# Benchmark laufen lassen
sep
log "Starte Benchmark-Laeufe..."
echo ""

for model in "${MODELS[@]}"; do
  sep
  log "Modell: $model"

  if $RUN_A; then
    log "  System A (LLM-only)..."
    python3 system_a.py --model "$model"
    log "  System A fertig."
  fi

  if $RUN_B; then
    log "  System B (LLM+KG)..."
    python3 system_b.py --model "$model"
    log "  System B fertig."
  fi
done

# Auswertung
sep
log "Starte Auswertung aller Laeufe..."
echo ""

if $RUN_A && $RUN_B; then
  python3 eval.py --all --compare
elif $RUN_A; then
  python3 eval.py --all --compare
else
  python3 eval.py --all --compare
fi

sep
log "Fertig. Alle Ergebnisse liegen in thesis-code/results/."
log "Eval-JSONs (eval_*.json) fuer Kap. 5.3 nutzbar."
echo ""
