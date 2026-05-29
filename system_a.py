"""System A: LLM-only-Baseline.

Schickt jede Benchmark-Frage einzeln an Ollama und speichert die
JSON-Antworten unter results/.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
import yaml

OLLAMA_URL = "http://localhost:11434/api/generate"

SYSTEM_PROMPT = """Du bist ein praeziser Assistent fuer Fragen zur
Battery Knowledge Base des Fraunhofer ISC. Du beantwortest Fragen zu
einer kommerziellen Lithium-Ionen-Zelle (LG INR18650 MH1) und ihren
durchgefuehrten Tests, Halbzellen, Cell Openings und Post-Mortem-
Analysen.

Wenn du eine Antwort nicht zuverlaessig weisst, antworte mit dem JSON-
Feld answer = null und confidence <= 0.2. Erfinde keine Werte.

Antworte AUSSCHLIESSLICH im folgenden JSON-Format (ohne weitere Texte
ausserhalb des JSON-Blocks):

{
  "answer": <string | number | array | null>,
  "confidence": <float zwischen 0 und 1>,
  "reasoning": "<ein kurzer Satz, max. 200 Zeichen>"
}
"""


def build_prompt(frage: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nFrage: {frage}\n\nAntwort (JSON):"


def call_ollama(model: str, prompt: str, timeout: int = 120) -> dict:
    """Ruft Ollama auf und gibt das geparste Response-Objekt zurueck."""
    started = time.perf_counter()
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 400},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    raw = resp.json()
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return {
        "raw_response": raw.get("response", ""),
        "tokens_prompt": raw.get("prompt_eval_count", 0),
        "tokens_completion": raw.get("eval_count", 0),
        "latency_ms": elapsed_ms,
    }


def extract_json(text: str) -> dict | None:
    """Ersten {...}-Block aus dem Modell-Output parsen."""
    # <think>-Block entfernen (Reasoning-Modelle)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="System A: LLM-only-Runner")
    parser.add_argument("--model", default="mistral", help="Ollama-Modellname")
    parser.add_argument("--only", default=None,
                        help="Komma-separierte IDs, z.B. 'K1.1,N2'")
    args = parser.parse_args()

    root = Path(__file__).parent
    benchmark_path = root / "data" / "benchmark.yaml"
    if not benchmark_path.exists():
        print(f"FEHLER: {benchmark_path} fehlt.", file=sys.stderr)
        return 1

    with benchmark_path.open(encoding="utf-8") as f:
        benchmark = yaml.safe_load(f)

    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        benchmark = [q for q in benchmark if q["id"] in wanted]

    print(f"System A laeuft. Modell={args.model}, Fragen={len(benchmark)}")
    results = []
    for q in benchmark:
        print(f"  [{q['id']}] {q['frage'][:70]} ...", end=" ", flush=True)
        try:
            out = call_ollama(args.model, build_prompt(q["frage"]))
        except requests.RequestException as exc:
            print(f"FEHLER: {exc}")
            results.append({"id": q["id"], "error": str(exc)})
            continue

        parsed = extract_json(out["raw_response"])
        results.append({
            "id": q["id"],
            "klasse": q["klasse"],
            "frage": q["frage"],
            "ground_truth": q["ground_truth"],
            "model_answer": parsed,
            "raw_response": out["raw_response"],
            "tokens_prompt": out["tokens_prompt"],
            "tokens_completion": out["tokens_completion"],
            "latency_ms": out["latency_ms"],
        })
        print(f"{out['latency_ms']} ms")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = args.model.replace(":", "-").replace("/", "-")
    out_path = root / "results" / f"system_a_{model_safe}_{ts}.json"
    out_path.write_text(
        json.dumps({"model": args.model, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Fertig. Resultate: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
