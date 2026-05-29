# thesis-code – Bachelor-Thesis Code

Lauffähiger Code zur Forschungsfrage „KG-gestützter LLM-Tool-Use vs. LLM-only" (Bachelorarbeit Fabio Daghero, Hochschule Offenburg 2026).

## Voraussetzungen

- Python 3.10+
- [Ollama](https://ollama.com) auf `http://localhost:11434`
- [Apache Jena Fuseki](https://jena.apache.org/documentation/fuseki2/) auf `http://localhost:3030` mit geladenem `snapshot.ttl` (Dataset: `battery`)
- Dependencies:
  ```
  pip install -r requirements.txt
  ```

## Verzeichnisstruktur

```
thesis-code/
  data/
    benchmark.yaml         12 Fragen + Ground Truth (Klassen K1–K3, Negativfälle N1–N3)
  results/                 JSON-Logs pro Run (wird automatisch erstellt, nicht im Repo)
  system_a.py              System A – LLM-only
  system_b.py              System B – LLM+KG via Tool-Use gegen Fuseki
  eval.py                  Evaluierung: vergleicht results/*.json gegen Ground Truth
  run_all_models.ps1       Alle Modelle sequenziell laufen lassen (Windows)
  requirements.txt
```

## Unterstützte Modelle

Beliebiges Ollama-Modell über `--model`. Getestete Modelle:

| Modell | System B Faktualität |
|---|---|
| `llama3.1:8b-instruct-q4_K_M` | 83,3 % |
| `qwen2.5-coder:7b-instruct` | 41,7 % |
| `mistral` | 33,3 % |
| `qwen2.5:7b-instruct` | 33,3 % |
| `gemma2:9b` | 25,0 % |
| `deepseek-r1:7b` | 16,7 % |

## Schneller Sanity-Lauf

```powershell
# System A – zwei Fragen, Standardmodell
python system_a.py --model llama3.1:8b-instruct-q4_K_M --only K1.3,N1

# System B – zwei Fragen
python system_b.py --model llama3.1:8b-instruct-q4_K_M --only K1.3,N1
```

- K1.3 („Wie viele LG-INR-Zellen?") → erwartete Antwort: 88
- N1 („Nennspannung von PM1?") → erwartete Antwort: nicht im Wissensgraph

## Voller Benchmark-Lauf (ein Modell)

```powershell
python system_a.py --model llama3.1:8b-instruct-q4_K_M
python system_b.py --model llama3.1:8b-instruct-q4_K_M
```

Ergebnisse landen in `results/system_a_<modell>_<timestamp>.json` bzw. `system_b_...`.

## Alle Modelle auf einmal (Windows)

```powershell
.\run_all_models.ps1
```

Das Skript führt System A und System B für alle sechs Modelle nacheinander aus.

## Evaluierung

```powershell
# Einzelne Ergebnisdatei auswerten
python eval.py results/system_b_llama3.1-8b-instruct-q4_K_M_<timestamp>.json

# Zwei Runs vergleichen (System A vs. System B)
python eval.py results/system_a_<ts>.json --compare results/system_b_<ts>.json
```

Ausgabe: Faktualitätsquote (strict/lenient), Latenz, Token-Verbrauch, SPARQL-Fehlerquote.
