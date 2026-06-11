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
PROMPT_VERSION = "v1.2"       # wird im Ergebnis-JSON geloggt
MAX_SPARQL_CALLS = 3           # Maximale SPARQL-Calls pro Frage
MAX_TURNS = 10                 # Sicherheits-Turnlimit (SPARQL-Limit greift zuerst)
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
- Eigenschaften wie Alterungszustand stecken nur im rdfs:label, nicht als
  eigenes Tripel. Bei Wortsuche im Label Word-Boundary beachten:
    FILTER( REGEX(?label, '(^|[^a-zA-Z])<suchwort>', 'i') )
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
      ?proc rdfs:label '<Procedure-Label>' .
      ?test hb:Property-3AHasProcedure ?proc ; hb:Property-3AHasDut ?cell .
  E) COUNT mit Aggregat -- Klammern sind PFLICHT:
      SELECT (COUNT(?x) AS ?n) WHERE BODY
  F) REGEX-Filter auf Labels -- Label erst binden:
      ?cell rdfs:label ?label .
      FILTER( REGEX(?label, '<suchwort>', 'i') )
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


# kein f-String: geschweifte Klammern via OBJ_OPEN/OBJ_CLOSE
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
    "  2) Die finale Antwort geben (NUR wenn du das Query-Ergebnis gesehen hast):\n"
    "     OBJ_OPEN\n"
    "       \"action\": \"answer\",\n"
    "       \"answer\": <string | number | array | null>,\n"
    "       \"status\": \"<supported|partially_supported|unknown|unsupported|error>\",\n"
    "       \"confidence\": <float 0-1>,\n"
    "       \"reasoning\": \"<max. 200 Zeichen>\",\n"
    "       \"conditions\": [\"<optionale Bedingung, z.B. lokaler Snapshot, Temp=25C>\"],\n"
    "       \"evidence\": OBJ_OPEN \"query_id\": \"sparql_call_1\", "
    "\"n_bindings\": <int>, \"used_values\": [\"<Label>\"] OBJ_CLOSE,\n"
    "       \"limitations\": \"<optional, nur bei unknown oder unsupported>\"\n"
    "     OBJ_CLOSE\n"
    "\n"
    "STATUS-REGELN:\n"
    "- status=supported:           Antwort vollstaendig durch KG-Daten belegt.\n"
    "- status=partially_supported: KG hat Teilergebnis; Antwort unvollstaendig.\n"
    "- status=unknown:             SPARQL lieferte n=0 (Entitaet/Relation nicht im KG).\n"
    "                              Setze answer=null, limitations erklaert den Grund.\n"
    "- status=unsupported:         Frage ausserhalb des Scope dieser KG.\n"
    "                              Setze answer=null, limitations erklaert den Grund.\n"
    "- status=error:               Maximale SPARQL-Versuche ohne verwertbares Ergebnis.\n"
    "\n"
    "REGELN:\n"
    "- Sobald du sparql ausgibst, HOERST DU SOFORT NACH DEM SCHLIESSENDEN OBJ_CLOSE AUF.\n"
    "- NIEMALS BEIDE AKTIONEN in derselben Antwort kombinieren.\n"
    "- answer ist verboten, solange du noch kein Query-Ergebnis erhalten hast.\n"
    "- Wenn das Query-Ergebnis leer ist (n=0): status=unknown, answer=null,\n"
    "  confidence<=0.3, limitations='Kein Treffer im Snapshot'. "
    "Das ist kein Fehler -- gib direkt action=answer aus.\n"
    "- Wenn du die Nachricht 'PFLICHT: [...] action=answer' erhaeltst, gib sofort\n"
    "  action=answer aus. Falls kein Ergebnis vorlag: status=error.\n"
    "- Du darfst pro Frage MAXIMAL 3 SPARQL-Abfragen ausfuehren. Weitere\n"
    "  Abfragen werden abgelehnt. Plane deine Queries entsprechend.\n"
    "- Erfinde keine Werte.\n"
    "\n"
    "JSON-OUTPUT-REGELN:\n"
    "- Verwende in SPARQL-String-Literalen NUR einfache Apostrophes, "
    "NIEMALS doppelte Anfuehrungszeichen. Beispiel:\n"
    "    OK : FILTER(?label = '<Label>')\n"
    "    NICHT: FILTER(?label = \"<Label>\")\n"
    "- Pruefe vor dem Senden: dein Output ist EIN einzelnes JSON-Objekt "
    "mit schliessendem OBJ_CLOSE.\n"
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
                    "error": "HTTP {}".format(resp.status_code),
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

    values_clause = " ".join("<{}>".format(u) for u in uris)
    label_query = (
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
        "SELECT ?u ?lbl WHERE {\n"
        "  VALUES ?u { " + values_clause + " }\n"
        "  ?u rdfs:label ?lbl .\n"
        "}"
    )
    result = run_sparql(label_query)
    if not result.get("ok"):
        return rows  # Fallback: unveraendert zurueck

    uri_to_label: dict = {}
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
        "{}: {}".format(m["role"].upper(), m["content"])
        for m in messages if m["role"] != "system"
    )
    full_prompt = "{}\n\nASSISTANT:".format(convo)
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
        trimmed = body.rstrip()
        if trimmed.endswith("}"):
            trimmed = trimmed[:-1].rstrip()
        if not trimmed.endswith('"'):
            return None
        query = trimmed[:-1]
        query = query.replace('\\n', '\n').replace('\\t', '\t')
        query = query.replace('\\"', '"').replace("\\'", "'")
        return {"action": "sparql", "query": query.strip()}
    if action == "answer":
        # status und limitations zusaetzlich extrahieren
        ans_m = re.search(
            r'"answer"\s*:\s*(.+?)(?:,\s*"(?:status|confidence)"|\})', text, re.DOTALL
        )
        status_m = re.search(r'"status"\s*:\s*"([^"]+)"', text)
        conf_m = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
        reas_m = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
        lim_m = re.search(r'"limitations"\s*:\s*"([^"]*)"', text)
        if ans_m:
            ans_raw = ans_m.group(1).strip().rstrip(',').strip()
            try:
                ans_val = json.loads(ans_raw)
            except json.JSONDecodeError:
                ans_val = ans_raw.strip('"').strip("'")
            return {
                "action": "answer",
                "answer": ans_val,
                "status": status_m.group(1) if status_m else None,
                "confidence": float(conf_m.group(1)) if conf_m else None,
                "reasoning": reas_m.group(1) if reas_m else "",
                "conditions": [],
                "evidence": {},
                "limitations": lim_m.group(1) if lim_m else None,
            }
    return None


def extract_json(text: str):
    """Robustes Parsen in drei Stufen."""
    if not text:
        return None
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


# Hilfsfunktion fuer Logging-Felder

def _sparql_log_stats(sparql_log: list) -> dict:
    """Berechnet die vier neuen B3-Logging-Felder aus dem SPARQL-Verlauf."""
    n = len(sparql_log)
    if n == 0:
        return {
            "n_sparql_calls": 0,
            "first_query_ok": None,
            "final_query_ok": None,
            "repaired_after_error": False,
        }
    first_ok = sparql_log[0]["result_summary"].get("ok", False)
    final_ok = sparql_log[-1]["result_summary"].get("ok", False)
    # repaired: Fehler vor der letzten Query
    repaired = (
        any(
            not sparql_log[i]["result_summary"].get("ok", False)
            for i in range(n - 1)
        )
        if n > 1 else False
    )
    return {
        "n_sparql_calls": n,
        "first_query_ok": first_ok,
        "final_query_ok": final_ok,
        "repaired_after_error": repaired,
    }


# Haupt-Loop

def answer_one(question: str, model: str) -> dict:
    """Multi-Turn-Loop fuer eine Frage."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Frage: {}".format(question)},
    ]
    sparql_log: list = []
    n_sparql_calls = 0
    n_rejected = 0  # abgewiesene Query-Versuche nach Limit
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_latency_ms = 0
    raw_history: list = []

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
                **_sparql_log_stats(sparql_log),
            }

        action = parsed.get("action")

        if action == "answer":
            return {
                "final": parsed,
                "n_rejected_sparql": n_rejected,
                "raw_history": raw_history,
                "sparql_queries": sparql_log,
                "turns": turn + 1,
                "tokens_prompt": total_prompt_tokens,
                "tokens_completion": total_completion_tokens,
                "latency_ms": total_latency_ms,
                **_sparql_log_stats(sparql_log),
            }

        if action == "sparql":
            # hartes Limit: Query nicht ausfuehren, Antwort einfordern
            if n_sparql_calls >= MAX_SPARQL_CALLS:
                n_rejected += 1
                messages.append({
                    "role": "user",
                    "content": (
                        "ABGELEHNT: Das Limit von {} SPARQL-Abfragen ist erreicht. "
                        "Deine Query wurde NICHT ausgefuehrt. Gib JETZT action=answer "
                        "aus. Nutze die bisherigen Ergebnisse. Waren alle Ergebnisse "
                        "leer (n=0), antworte mit answer=null, status=unknown, "
                        "limitations='Kein Treffer im Snapshot'."
                    ).format(MAX_SPARQL_CALLS),
                })
                continue
            query = parsed.get("query", "").strip()
            result = run_sparql(query)
            n_sparql_calls += 1

            # call_index im Log festhalten
            sparql_log.append({
                "call_index": n_sparql_calls,
                "query": query,
                "result_summary": {
                    "ok": result.get("ok"),
                    "n": result.get("n"),
                    "error": result.get("error"),
                    "error_detail": result.get("error_detail"),
                },
            })

            if result.get("ok"):
                # n=0 ist kein Fehler
                if result.get("rows"):
                    result["rows"] = resolve_uris_to_labels(result["rows"])
                obs = json.dumps(result, ensure_ascii=False)[:2000]
                messages.append({
                    "role": "user",
                    "content": "SPARQL-Ergebnis (sparql_call_{}): {}".format(
                        n_sparql_calls, obs),
                })
            else:
                # Self-Correction nur bei technischen Fehlern
                detail = result.get("error_detail") or result.get("error") or "unbekannter Fehler"
                obs = "SPARQL-FEHLER (bitte Query korrigieren): {}".format(detail[:600])
                messages.append({"role": "user", "content": obs})

            # nach Limit: Antwort einfordern
            if n_sparql_calls >= MAX_SPARQL_CALLS:
                messages.append({
                    "role": "user",
                    "content": (
                        "PFLICHT: Du hast {} SPARQL-Abfragen ausgefuehrt "
                        "(Maximum erreicht). Gib JETZT deine finale Antwort mit "
                        "action=answer aus. Keine weiteren sparql-Aktionen erlaubt."
                    ).format(MAX_SPARQL_CALLS),
                })
            continue

        # Unbekannte Aktion
        return {
            "final": parsed,
            "error": "unknown action: {}".format(action),
            "raw_history": raw_history,
            "sparql_queries": sparql_log,
            "turns": turn + 1,
            "tokens_prompt": total_prompt_tokens,
            "tokens_completion": total_completion_tokens,
            "latency_ms": total_latency_ms,
            **_sparql_log_stats(sparql_log),
        }

    # Timeout
    return {
        "final": None,
        "error": "max turns ({}) reached without answer".format(MAX_TURNS),
        "n_rejected_sparql": n_rejected,
        "raw_history": raw_history,
        "sparql_queries": sparql_log,
        "turns": MAX_TURNS,
        "tokens_prompt": total_prompt_tokens,
        "tokens_completion": total_completion_tokens,
        "latency_ms": total_latency_ms,
        **_sparql_log_stats(sparql_log),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="System B: LLM+KG via Tool-Use")
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--only", default=None,
                        help="Komma-separierte IDs, z.B. 'K1.1,N2'")
    parser.add_argument("--benchmark", default=None,
                        help="Pfad zur Benchmark-YAML (Default: data/benchmark_v1.yaml)")
    parser.add_argument("--results-dir", default=None,
                        help="Ausgabeordner fuer JSON-Resultate (Default: results/)")
    args = parser.parse_args()

    root = Path(__file__).parent
    bm_file = args.benchmark if args.benchmark else str(root / "data" / "benchmark_v1.yaml")
    benchmark_path = Path(bm_file)
    if not benchmark_path.is_absolute():
        benchmark_path = root / benchmark_path
    with benchmark_path.open(encoding="utf-8") as f:
        benchmark = yaml.safe_load(f)

    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        benchmark = [q for q in benchmark if q["id"] in wanted]

    health = run_sparql("ASK { ?s ?p ?o }")
    if not health.get("ok"):
        print("FEHLER: Fuseki nicht erreichbar -- {}".format(health.get("error")),
              file=sys.stderr)
        return 1

    print("System B laeuft. Modell={}, Fragen={}".format(args.model, len(benchmark)))
    results = []
    for q in benchmark:
        print("  [{}] {}".format(q["id"], q["frage"][:60]), flush=True)
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
            # neue Logging-Felder
            "n_sparql_calls": out.get("n_sparql_calls"),
            "first_query_ok": out.get("first_query_ok"),
            "final_query_ok": out.get("final_query_ok"),
            "repaired_after_error": out.get("repaired_after_error"),
            "n_rejected_sparql": out.get("n_rejected_sparql"),
            "tokens_prompt": out.get("tokens_prompt"),
            "tokens_completion": out.get("tokens_completion"),
            "latency_ms": out.get("latency_ms"),
            "error": out.get("error"),
        })
        n_calls = out.get("n_sparql_calls", 0)
        repaired = out.get("repaired_after_error", False)
        print("    -> {} Turns, {} SPARQL-Calls, repaired={}, {} ms".format(
            out.get("turns"), n_calls, repaired, out.get("latency_ms")))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = args.model.replace(":", "-").replace("/", "-")
    results_dir = Path(args.results_dir) if args.results_dir else root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "system_b_{}_{}.json".format(model_safe, ts)
    out_path.write_text(
        json.dumps({
            "model": args.model,
            "prompt_version": PROMPT_VERSION,
            "benchmark": str(benchmark_path),
            "results": results,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Fertig. Resultate: {}".format(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
