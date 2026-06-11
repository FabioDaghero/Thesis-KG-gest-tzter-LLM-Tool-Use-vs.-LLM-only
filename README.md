# thesis-code – Bachelor-Thesis Code

Lauffähiger Code zur Forschungsfrage „KG-gestützter LLM-Tool-Use vs. LLM-only"
(Bachelorarbeit Fabio Daghero, Hochschule Offenburg 2026).

## Voraussetzungen

- Python 3.10+
- [Ollama](https://ollama.com) auf `http://localhost:11434`
- [Apache Jena Fuseki](https://jena.apache.org/documentation/fuseki2/) auf
  `http://localhost:3030` mit geladenem `snapshot.ttl` (Dataset: `battery`,
  1431 Tripel — dokumentiert in `data/snapshot_info.yaml`)
- Dependencies:
  ```
  pip install -r requirements.txt
  ```

## Verzeichnisstruktur

```
thesis-code/
  data/
    benchmark_v1.yaml        Hauptbenchmark: 48 Fragen (K1/K2/K3/N à 12),
                             Gold-SPARQL, Gold-Evidenz, Ground Truth
    benchmark.yaml           v0-Pilot: 12 Fragen, 3 Modelle (Modellauswahl)
    snapshot_info.yaml       KG-Snapshot-Dokumentation (Version, Klassen, IRIs)
    results_summary.yaml     Aggregierte Auswertung (generiert, s. unten)
    gold_sparql_results/     Gold-Resultsets als CSV (23 Dateien)
  results/                   v0-Pilot-Läufe
  results/main_benchmark/        Hauptbenchmark, Prompt v1.0 (3 Runs A+B)
  results/main_benchmark_v1.1/   Prompt v1.1 — Kontaminationsbereinigung
  results/main_benchmark_v1.2/   Prompt v1.2 — hartes SPARQL-Limit (final)
  system_a.py                System A – LLM-only (Baseline, kein Schema-Wissen)
  system_b.py                System B – LLM+KG via SPARQL-Tool-Use gegen Fuseki
  eval.py                    Evaluierung gegen Ground Truth
  make_results_summary.py    Summary-Generierung + Ablationsvergleich
  run_all_models.ps1         Benchmark-Runner (Windows)
```

## Prompt-Versionen (System B)

| Version | Änderung |
|---|---|
| v1.0 | Eingefroren vor Hauptbenchmark |
| v1.1 | Benchmark-Entitäten aus Query-Patterns entfernt (Kontaminationskontrolle) |
| v1.2 | Hartes SPARQL-Limit: max. 3 Calls, weitere werden abgewiesen (`n_rejected_sparql`) |

Die Version wird in jedem Ergebnis-JSON als `prompt_version` geloggt.
System A blieb durchgehend auf v1.0 (deterministisch, von B-Änderungen unberührt).

## Hauptbenchmark reproduzieren

```powershell
# 3 Wiederholungsläufe, System A + B, Llama 3.1 8B
.\run_all_models.ps1 -ResultsDir results/main_benchmark_v1.2

# nur System B (A ist deterministisch und liegt bereits vor)
.\run_all_models.ps1 -OnlyB -ResultsDir results/main_benchmark_v1.2
```

## Evaluierung und Auswertung

```powershell
# Alle Ergebnisse eines Ordners evaluieren (schreibt eval_*.json)
python eval.py --dir results/main_benchmark_v1.2

# Summary generieren (Mittel über alle Runs, per-Klasse, per-Frage)
python make_results_summary.py --dir results/main_benchmark_v1.2 `
    --a-dir results/main_benchmark_v1.1

# Ablation: Prompt-Versionen vergleichen
python make_results_summary.py --compare results/main_benchmark `
    results/main_benchmark_v1.1 results/main_benchmark_v1.2
```

## Schneller Sanity-Lauf

```powershell
python system_a.py --model llama3.1:8b-instruct-q4_K_M --only K1.3,N1
python system_b.py --model llama3.1:8b-instruct-q4_K_M --only K1.3,N1
```

- K1.3 („Wie viele LG-INR-Zellen?") → erwartet: 88
- N1 („Nennspannung von healthbatt_PM1?") → erwartet: `null` / status=unknown

## v0-Pilot (Modellauswahl)

Der Pilot (12 Fragen, 6 Modelle) diente der Modellauswahl. Zentrales
Ergebnis: Nur Llama 3.1 8B erzeugt zuverlässig syntaktisch korrektes
SPARQL (91,7 % vs. ≤16,7 % bei Mistral/Qwen). Der Hauptbenchmark
fokussiert daher auf `llama3.1:8b-instruct-q4_K_M`.

```powershell
.\run_all_models.ps1 -Benchmark data/benchmark.yaml -ResultsDir results
```
