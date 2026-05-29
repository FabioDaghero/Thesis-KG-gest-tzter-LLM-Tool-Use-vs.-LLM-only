"""System B: LLM + KG via SPARQL-Tool-Use.

Multi-Turn-Loop gegen Ollama; Queries laufen gegen den lokalen
Fuseki-Endpoint. JSON-Protokoll mit den Aktionen sparql/answer.
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
FUSEKI_URL = "http://localhost:3030/battery/sparql"
MAX_TURNS = 4
MAX_RESULT_ROWS = 50
HB_PREFIX = "https://healthbatt.projects01.open-semantic-lab.org/id/"

SCHEMA_BLOCK = """
SNAPSHOT-SCHEMA (Battery Knowledge Base, Fraunhofer ISC, healthbatt-Sub-Wiki):

Prefix in allen Queries verwenden:
    PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX hb:   <https://healthbatt.projects01.open-semantic-lab.org/id/>

Klassen (Category-IRIs):
    hb:Category-3AOSW60482870f8fc406782f60295bf7a09e3  -- LG INR18650 MH1 (Battery Cell, 88 Inst.)
    hb:Category-3AOSW6f39d77241e24a33ab6d036dfac03ace  -- Electrochemical Test (125)
    hb:Category-3AOSWdda41d4a4ec0421babe0295c6edcb5df  -- Test Procedure (19)
    hb:Category-3AOSWa8481aad04e84d5f875ecdab19407e6f  -- Half Cell ISC (25)
    hb:Category-3AOSWd5c685a7b3c74241af0311176d0205f6  -- Cell Opening (7)
    hb:Category-3AOSW47c67760dd164c82b570f58c8269b373  -- Post-Mortem Experiment (7)
    hb:Category-3AOSWa2d79bb7ad78412b80f7f98482371096  -- Electrode Thickness Measurement (12)
    hb:Category-3AOSW5bee2b0677e84c059bbdc5a6db7c0d92  -- Electrode Weight Measurement (12)

Relationen (Property-IRIs):
    rdf:type                              -- Klassenzugehoerigkeit
    rdfs:label                            -- menschenlesbarer Name
    hb:Property-3AHasDut                  -- Test -> Cell (zentrale Multi-Hop-Achse)
    hb:Property-3AHasProcedure            -- Test -> Test Procedure
    hb:Property-3AHasActionee             -- Test -> Person, die ihn durchfuehrte
    hb:Property-3AHasStartDateAndTime-23aux  -- Test -> Zeitstempel (Literal)
    hb:Property-3AHasTemperature          -- Test -> Temperatur (als URI .../id/N)
    hb:Property-3AHasSerialNumber         -- Cell -> physische Seriennummer

WICHTIGE QUIRKS:
- HasTemperature liefert eine URI, NICHT ein Literal. Den Wert mit
    BIND(REPLACE(STR(?tempUri), '.*/', '') AS ?temp)
  extrahieren und mit FILTER(?temp = '25') vergleichen.
- 'gealtert' steckt nur in rdfs:label, nicht als Tripel. Word-Boundary noetig:
    FILTER( REGEX(?label, '(^|[^a-zA-Z])gealtert', 'i') || REGEX(?label, 'aged', 'i') )
- IsBasedOn ist Wiki-Versionierung, KEINE Halbzelle-zu-Cell-Beziehung -> nicht verwenden.
- HasDut zeigt vom Test auf die Cell (umgekehrt zur intuitiven Richtung).

QUERY-PATTERNS:
  A) Test ueber sein Label finden:
      ?test rdfs:label '<Test-Label>' ; rdf:type hb:Category-3AOSW6f39d77241e24a33ab6d036dfac03ace .
  B) Cell ueber Test (HasDut) finden (Multi-Hop):
      ?test rdfs:label '<Test-Label>' ; hb:Property-3AHasDut ?cell .
      ?cell rdfs:label ?cellLabel .
  C) Tests einer benannten Cell (umgekehrt!):
      ?cell rdfs:label '<Cell-Label>' .
      ?test hb:Property-3AHasDut ?cell .
  D) Procedure ueber Label, dann Tests/Cells:
      ?proc rdfs:label 'hartm_HealthBatt_Cycling' .
      ?test hb:Property-3AHasProcedure ?proc ; hb:Property-3AHasDut ?cell .
  E) COUNT mit Aggregat -- Klammern sind PFLICHT:
      SELECT (COUNT(?x) AS ?n) WHERE BODY
  F) REGEX-Filter auf Labels -- Label erst binden:
      ?cell rdfs:label ?label .
      FILTER( REGEX(?label, 'aged', 'i') )
  G) Instanzen einer Klasse zaehlen -- rdf:type verwenden, NICHT rdfs:label:
      SELECT (COUNT(?inst) AS ?n) WHERE {
        ?inst rdf:type hb:Category-3A... .
      }
      WICHTIG: Die Variable in COUNT(?inst) MUSS dieselbe sein wie in WHERE (?inst).
      FALSCH: SELECT (COUNT(?x) AS ?n) WHERE { ?inst rdf:type ... } -- ?x ist ungebunden!

ANTI-PATTERNS:
  - Erfinde NIEMALS IRIs wie <http://example.com/...>. Suche stattdessen ueber rdfs:label.
  - HasSerialNumber ist NUR an Cells, nicht an Tests. Test-Namen stehen in rdfs:label.
  - REGEX(?cell rdfs:label, ...) ist falsch. Bind erst.
  - PREFIX-Deklarationen kommen IMMER VOR SELECT, nie danach.
"""


SYSTEM_PROMPT = (
    "Du bist ein praeziser Assistent fuer Fragen zur Battery Knowledge Base "
    "des Fraunhofer ISC. Du hast Zugriff auf einen lokalen SPARQL-Endpoint, der einen "
    "Snapshot mit 1431 Tripeln enthaelt.\n"
    + SCHEMA_BLOCK +
    "\n\n"
    "Du gibst PRO ANTWORT GENAU EIN EINZIGES JSON-Objekt aus -- niemals zwei.\n"
    "Keine Markdown-Code-Fences, kein Fliesstext drumherum. Zwei Aktionen sind erlaubt:\n"
    "\n"
    "  1) Eine SPARQL-Abfrage absetzen:\n"
    "     OBJ_OPEN \"action\": \"sparql\", \"query\": \"<dein SPARQL-Text>\" OBJ_CLOSE\n"
    "\n"
    "  2) Die finale Antwort geben (NUR wenn du das Query-Ergebnis schon gesehen hast):\n"
    "     OBJ_OPEN \"action\": \"answer\", \"answer\": <wert>, "
    "\"confidence\": <float>, \"reasoning\": \"<max 200 Zeichen>\" OBJ_CLOSE\n"
    "\n"
    "REGELN:\n"
    "- Sobald du sparql ausgibst, HOERST DU SOFORT NACH DEM SCHLIESSENDEN OBJ_CLOSE AUF.\n"
    "- NIEMALS BEIDE AKTIONEN in derselben Antwort kombinieren.\n"
    "- answer ist verboten, solange du noch kein Query-Ergebnis erhalten hast.\n"
    "- Wenn das Query-Ergebnis leer ist (n=0), antworte mit answer=null, "
    "confidence<=0.3 und sage in reasoning, dass die Information nicht im Snapshot vorhanden ist.\n"
    "- Erfinde keine Werte.\n"
    "\n"
    "JSON-OUTPUT-REGELN:\n"
    "- Verwende in SPARQL-String-Literalen NUR einfache Apostrophes, "
    "NIEMALS doppelte Anfuehrungszeichen. Beispiel:\n"
    "    OK : FILTER(?label = 'HealthBatt MT_05')\n"
    "    NICHT: FILTER(?label = \"HealthBatt MT_05\")\n"
    "- Pruefe vor dem Senden: dein Output ist EIN einzelnes JSON-Objekt mit schliessendem OBJ_CLOSE.\n"
    "- Wenn du als Observation eine Nachricht erhältst, die mit 'SPARQL-FEHLER' beginnt, "
    "bedeutet das, dass deine letzte Query abgelehnt wurde. Gib SOFORT eine korrigierte "
    "Query mit action=sparql aus. Haeufige Ursachen: fehlende SELECT-Klausel, ungebundene "
    "Variable in COUNT (z.B. COUNT(?x) obwohl ?x im WHERE-Teil nicht vorkommt), "
    "falsche Syntax. Lies den Fehlertext und behebe den konkreten Fehler.\n"
).replace("OBJ_OPEN", "{").replace("OBJ_CLOSE", "}")


def run_sparql(query: str) -> dict:
    """Setzt die Query an Fuseki ab. Gibt strukturierte Antwort zurueck."""
    try:
        resp = requests.post(
            FUSEKI_URL,
            data={"query": query},
            headers={"Accept": "application/sparql-results+json"},
            timeout=30,
        )
        if resp.status_code >= 400:
            return {"ok": False, "status": resp.status_code,
                    "error": f"HTTP {resp.status_code}",
                    "error_detail": resp.text[:800]}
        data = resp.json()
        bindings = data.get("results", {}).get("bindings", [])
        vars_ = data.get("head", {}).get("vars", [])
        rows = [{v: b.get(v, {}).get("value") for v in vars_} for b in bindings]
        return {"ok": True, "n": len(rows), "rows": rows[:MAX_RESULT_ROWS],
                "truncated": len(rows) > MAX_RESULT_ROWS}
    except requests.RequestException as exc:
        return {"ok": False, "error": str(exc)}
    except json.JSONDecodeError:
        return {"ok": False, "error": "non-JSON response from Fuseki"}


def resolve_uris_to_labels(rows: list) -> list:
    """HB-URIs in Ergebnis-Rows durch rdfs:label ersetzen (Batch-Lookup)."""
    uris = set()
    for row in rows:
        for val in row.values():
            if isinstance(val, str) and val.startswith(HB_PREFIX):
                uris.add(val)
    if not uris:
        return rows

    values_clause = " ".join(f"<{u}>" for u in uris)
    label_query = (
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
        "SELECT ?u ?lbl WHERE {\n"
        f"  VALUES ?u {{ {values_clause} }}\n"
        "  ?u rdfs:label ?lbl .\n"
        "}"
    )
    result = run_sparql(label_query)
    if not result.get("ok"):
        return rows  # Fallback: unveraendert zurueck

    uri_to_label: dict[str, str] = {}
    for r in result.get("rows", []):
        if r.get("u") and r.get("lbl"):
            uri_to_label[r["u"]] = r["lbl"]

    resolved = []
    for row in rows:
        new_row = {}
        for k, v in row.items():
            new_row[k] = uri_to_label.get(v, v) if isinstance(v, str) else v
        resolved.append(new_row)
    return resolved


def call_ollama_chat(messages: list, model: str):
    """Sendet eine Multi-Message-Konversation an Ollama."""
    started = time.perf_counter()
    sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
    convo = "\n\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in messages if m["role"] != "system"
    )
    full_prompt = f"{convo}\n\nASSISTANT:"
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "system": sys_msg,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 500},
        },
        timeout=180,
    )
    resp.raise_for_status()
    raw = resp.json()
    return (
        raw.get("response", ""),
        raw.get("prompt_eval_count", 0),
        raw.get("eval_count", 0),
        int((time.perf_counter() - started) * 1000),
    )


def _try_parse(candidate: str):
    try:
        return json.loads(candidate, strict=False)
    except json.JSONDecodeError:
        return None


def _fallback_extract(text: str):
    """Regex-basiert: extrahiert action + query/answer ohne strict-JSON."""
    m_action = re.search(r'"action"\s*:\s*"(\w+)"', text)
    if not m_action:
        return None
    action = m_action.group(1)
    if action == "sparql":
        m_start = re.search(r'"query"\s*:\s*"', text)
        if not m_start:
            return None
        body = text[m_start.end():]
        # letztes " im Body finden, das vor optionalem }-Whitespace-Ende steht
        # Strategie: das laenge Ende ruckwarts trimmen, dann letztes "
        trimmed = body.rstrip()
        if trimmed.endswith("}"):
            trimmed = trimmed[:-1].rstrip()
        if not trimmed.endswith('"'):
            return None
        query = trimmed[:-1]
        # Entschaerfen: bekannte JSON-Escapes deserialisieren
        query = query.replace('\\n', '\n').replace('\\t', '\t')
        query = query.replace('\\"', '"').replace("\\'", "'")
        return {"action": "sparql", "query": query.strip()}
    if action == "answer":
        ans_m = re.search(r'"answer"\s*:\s*(.+?)(?:,\s*"confidence"|\})', text, re.DOTALL)
        conf_m = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
        reas_m = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
        if ans_m:
            ans_raw = ans_m.group(1).strip().rstrip(',').strip()
            try:
                ans_val = json.loads(ans_raw)
            except json.JSONDecodeError:
                ans_val = ans_raw.strip('"').strip("'")
            return {
                "action": "answer",
                "answer": ans_val,
                "confidence": float(conf_m.group(1)) if conf_m else None,
                "reasoning": reas_m.group(1) if reas_m else "",
            }
    return None


def extract_json(text: str):
    """Robustes Parsen in drei Stufen."""
    if not text:
        return None
    # <think>-Block entfernen (Reasoning-Modelle)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```(?:json|sparql)?\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    # Stufe 1: Klammer-Zaehlung
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_string:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_string = False
            else:
                if c == '"':
                    in_string = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        parsed = _try_parse(candidate)
                        if parsed is not None:
                            return parsed
                        break
        start = text.find("{", start + 1)
    # Stufe 2: Regex-Fallback
    return _fallback_extract(text)


def answer_one(question: str, model: str) -> dict:
    """Multi-Turn-Loop fuer eine Frage."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Frage: {question}"},
    ]
    sparql_log = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_latency_ms = 0
    raw_history = []

    for turn in range(MAX_TURNS):
        text, pt, ct, lat = call_ollama_chat(messages, model)
        total_prompt_tokens += pt
        total_completion_tokens += ct
        total_latency_ms += lat
        raw_history.append(text)
        parsed = extract_json(text)
        messages.append({"role": "assistant", "content": text})

        if parsed is None:
            return {
                "final": None,
                "error": "Model output not parseable as JSON",
                "raw_history": raw_history,
                "sparql_queries": sparql_log,
                "turns": turn + 1,
                "tokens_prompt": total_prompt_tokens,
                "tokens_completion": total_completion_tokens,
                "latency_ms": total_latency_ms,
            }

        action = parsed.get("action")
        if action == "answer":
            return {
                "final": parsed,
                "raw_history": raw_history,
                "sparql_queries": sparql_log,
                "turns": turn + 1,
                "tokens_prompt": total_prompt_tokens,
                "tokens_completion": total_completion_tokens,
                "latency_ms": total_latency_ms,
            }
        if action == "sparql":
            query = parsed.get("query", "").strip()
            result = run_sparql(query)
            sparql_log.append({"query": query, "result_summary": {
                "ok": result.get("ok"),
                "n": result.get("n"),
                "error": result.get("error"),
                "error_detail": result.get("error_detail"),
            }})
            if result.get("ok"):
                # URIs durch lesbare Labels ersetzen, bevor das Ergebnis ans Modell geht
                if result.get("rows"):
                    result["rows"] = resolve_uris_to_labels(result["rows"])
                obs = json.dumps(result, ensure_ascii=False)[:2000]
                messages.append({"role": "user", "content": f"SPARQL-Ergebnis: {obs}"})
            else:
                # Fehlermeldung mit konkretem Detail, damit das Modell den Fehler beheben kann
                detail = result.get("error_detail") or result.get("error") or "unbekannter Fehler"
                obs = f"SPARQL-FEHLER (bitte Query korrigieren): {detail[:600]}"
                messages.append({"role": "user", "content": obs})
            continue
        return {
            "final": parsed,
            "error": f"unknown action: {action}",
            "raw_history": raw_history,
            "sparql_queries": sparql_log,
            "turns": turn + 1,
            "tokens_prompt": total_prompt_tokens,
            "tokens_completion": total_completion_tokens,
            "latency_ms": total_latency_ms,
        }

    return {
        "final": None,
        "error": f"max turns ({MAX_TURNS}) reached without answer",
        "raw_history": raw_history,
        "sparql_queries": sparql_log,
        "turns": MAX_TURNS,
        "tokens_prompt": total_prompt_tokens,
        "tokens_completion": total_completion_tokens,
        "latency_ms": total_latency_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="System B: LLM+KG via Tool-Use")
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--only", default=None,
                        help="Komma-separierte IDs, z.B. 'K1.1,N2'")
    args = parser.parse_args()

    root = Path(__file__).parent
    benchmark_path = root / "data" / "benchmark.yaml"
    with benchmark_path.open(encoding="utf-8") as f:
        benchmark = yaml.safe_load(f)

    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        benchmark = [q for q in benchmark if q["id"] in wanted]

    health = run_sparql("ASK { ?s ?p ?o }")
    if not health.get("ok"):
        print(f"FEHLER: Fuseki nicht erreichbar -- {health.get('error')}", file=sys.stderr)
        return 1

    print(f"System B laeuft. Modell={args.model}, Fragen={len(benchmark)}")
    results = []
    for q in benchmark:
        print(f"  [{q['id']}] {q['frage'][:60]}", flush=True)
        out = answer_one(q["frage"], args.model)
        results.append({
            "id": q["id"],
            "klasse": q["klasse"],
            "frage": q["frage"],
            "ground_truth": q["ground_truth"],
            "model_answer": out.get("final"),
            "sparql_queries": out.get("sparql_queries"),
            "raw_history": out.get("raw_history"),
            "turns": out.get("turns"),
            "tokens_prompt": out.get("tokens_prompt"),
            "tokens_completion": out.get("tokens_completion"),
            "latency_ms": out.get("latency_ms"),
            "error": out.get("error"),
        })
        n_calls = len(out.get("sparql_queries") or [])
        print(f"    -> {out.get('turns')} Turns, {n_calls} SPARQL-Calls, {out.get('latency_ms')} ms")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = args.model.replace(":", "-").replace("/", "-")
    out_path = root / "results" / f"system_b_{model_safe}_{ts}.json"
    out_path.write_text(
        json.dumps({"model": args.model, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Fertig. Resultate: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
