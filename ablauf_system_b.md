# Ablauf der Abfragen in System B (Schritt für Schritt)

Dieses Dokument geht den Weg einer einzelnen Frage durch System B
(`system_b.py`) der Reihe nach durch. Immer zuerst die Erklärung, was
passiert, dann der zugehörige Code-Ausschnitt.

System B ist der Kern der Arbeit: Das LLM beantwortet eine Frage nicht
aus seinem eigenen Wissen, sondern formuliert dafür SPARQL-Abfragen
gegen den lokalen Wissensgraphen. Das geschieht in einer Schleife über
mehrere Runden (Turns), bis das Modell eine finale Antwort liefert.

---

## Schritt 1: Konfiguration und feste Grenzen

Ganz oben in der Datei stehen die Adressen der beiden Dienste und die
Grenzwerte. `OLLAMA_URL` ist das lokale Sprachmodell, `FUSEKI_URL` der
SPARQL-Endpoint mit dem Wissensgraphen. Wichtig sind die zwei Limits:
maximal drei SPARQL-Abfragen pro Frage und maximal zehn Runden als
Sicherheitsnetz.

```python
OLLAMA_URL = "http://localhost:11434/api/generate"
FUSEKI_URL = "http://localhost:3030/battery/sparql"
PROMPT_VERSION = "v1.2"
MAX_SPARQL_CALLS = 3      # Maximale SPARQL-Calls pro Frage
MAX_TURNS = 10            # Sicherheits-Turnlimit
MAX_RESULT_ROWS = 50
HB_PREFIX = "https://healthbatt.projects01.open-semantic-lab.org/id/"
```

---

## Schritt 2: Das Modell bekommt das Schema mit (System-Prompt)

Bevor irgendeine Frage gestellt wird, wird dem Modell ein fester
System-Prompt mitgegeben. Darin steht der `SCHEMA_BLOCK`: die Klassen,
die Relationen und die Besonderheiten des Graphen (die sogenannte
Schema Card). Ohne diese Informationen könnte das Modell keine
korrekten IRIs benutzen. Hier ein Ausschnitt der Klassen und Quirks.

```python
SCHEMA_BLOCK = """
Klassen (Category-IRIs):
    hb:Category-3AOSW60482870... -- LG INR18650 MH1 (Battery Cell, 88 Inst.)
    hb:Category-3AOSW6f39d772... -- Electrochemical Test (125)
    hb:Category-3AOSWdda41d4a... -- Test Procedure (19)
    ...
WICHTIGE QUIRKS:
- HasTemperature liefert eine URI, NICHT ein Literal. Den Wert mit
    BIND(REPLACE(STR(?tempUri), '.*/', '') AS ?temp) extrahieren.
- HasDut zeigt vom Test auf die Cell (umgekehrt zur intuitiven Richtung).
"""
```

Der System-Prompt legt außerdem fest, dass das Modell pro Runde genau
ein JSON-Objekt ausgeben muss, mit einer von zwei Aktionen: entweder
eine Abfrage absetzen oder die finale Antwort geben.

```python
#   1) Eine SPARQL-Abfrage absetzen:
#      { "action": "sparql", "query": "<dein SPARQL-Text>" }
#
#   2) Die finale Antwort geben (NUR wenn du das Query-Ergebnis gesehen hast):
#      { "action": "answer", "answer": ..., "status": ..., "confidence": ... }
```

---

## Schritt 3: Eine Frage startet (Aufbau der Konversation)

Die Funktion `answer_one` behandelt genau eine Frage. Sie baut die
Startkonversation auf: die System-Nachricht (mit Schema) und die
eigentliche Frage als Benutzer-Nachricht. Außerdem werden mehrere
Zähler angelegt, die später protokollieren, was passiert ist.

```python
def answer_one(question: str, model: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Frage: {}".format(question)},
    ]
    sparql_log: list = []
    n_sparql_calls = 0
    n_rejected = 0          # nach Limit abgewiesene Versuche
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_latency_ms = 0
    raw_history: list = []
```

---

## Schritt 4: Die Runden-Schleife beginnt, das Modell antwortet

Jetzt startet die Schleife. In jeder Runde wird die komplette bisherige
Konversation an das Modell geschickt. Die Antwort des Modells (`text`)
wird gespeichert, die Tokens und die Dauer aufaddiert.

```python
    for turn in range(MAX_TURNS):
        text, pt, ct, lat = call_ollama_chat(messages, model)
        total_prompt_tokens += pt
        total_completion_tokens += ct
        total_latency_ms += lat
        raw_history.append(text)
        parsed = extract_json(text)
        messages.append({"role": "assistant", "content": text})
```

---

## Schritt 5: Wie der Modell-Aufruf genau funktioniert

`call_ollama_chat` setzt die Nachrichten zu einem einzigen Prompt
zusammen und schickt sie an Ollama. Wichtig ist `temperature: 0.0`:
Damit antwortet das Modell möglichst deterministisch, was die Läufe
besser wiederholbar macht. Zurück kommen der Antworttext sowie die
Token-Zahlen und die Dauer.

```python
def call_ollama_chat(messages: list, model: str):
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
    return (raw.get("response", ""), raw.get("prompt_eval_count", 0),
            raw.get("eval_count", 0), int((time.perf_counter() - started) * 1000))
```

---

## Schritt 6: Die Modell-Antwort wird zu JSON geparst

Das Modell gibt Text aus, der ein JSON-Objekt enthalten soll. Da kleine
Modelle dabei oft schludern, parst `extract_json` in mehreren Stufen.
Stufe 1 sucht die erste öffnende Klammer und zählt Klammern mit, bis das
Objekt vollständig ist, und versucht es zu parsen.

```python
def extract_json(text: str):
    if not text:
        return None
    text = re.sub(r"```(?:json|sparql)?\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            ...
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
```

Schlägt das saubere Parsen fehl, greift Stufe 2 (`_fallback_extract`):
Sie zieht mit regulären Ausdrücken die `action` und je nach Fall die
`query` oder die `answer`-Felder direkt aus dem Text. So überlebt das
System auch leicht kaputtes JSON.

```python
def _fallback_extract(text: str):
    m_action = re.search(r'"action"\s*:\s*"(\w+)"', text)
    if not m_action:
        return None
    action = m_action.group(1)
    if action == "sparql":
        ...
        return {"action": "sparql", "query": query.strip()}
    if action == "answer":
        ...
        return {"action": "answer", "answer": ans_val, "status": ...}
```

---

## Schritt 7: Falls gar nichts Parsbares zurückkommt

Liefert das Modell etwas, das auch der Fallback nicht versteht, bricht
die Funktion für diese Frage ab und gibt einen Fehler zurück, samt allen
bisher gesammelten Protokolldaten.

```python
        if parsed is None:
            return {
                "final": None,
                "error": "Model output not parseable as JSON",
                "raw_history": raw_history,
                "sparql_queries": sparql_log,
                "turns": turn + 1,
                ...
            }
```

---

## Schritt 8: Verzweigung nach der Aktion

Aus dem geparsten JSON wird die `action` gelesen. Davon hängt alles
Weitere ab. Liefert das Modell `answer`, ist die Frage fertig und das
Ergebnis wird zurückgegeben, inklusive der zusammengefassten
SPARQL-Statistik (`_sparql_log_stats`).

```python
        action = parsed.get("action")

        if action == "answer":
            return {
                "final": parsed,
                "n_rejected_sparql": n_rejected,
                "raw_history": raw_history,
                "sparql_queries": sparql_log,
                "turns": turn + 1,
                ...
                **_sparql_log_stats(sparql_log),
            }
```

---

## Schritt 9: Aktion "sparql" -- zuerst das Limit prüfen

Will das Modell eine Abfrage absetzen, wird zuerst geprüft, ob das Limit
von drei Abfragen schon erreicht ist. Wenn ja, wird die Query NICHT
ausgeführt; stattdessen wird das Modell aufgefordert, jetzt eine Antwort
zu geben. Genau das ist das harte SPARQL-Limit aus Version v1.2.

```python
        if action == "sparql":
            if n_sparql_calls >= MAX_SPARQL_CALLS:
                n_rejected += 1
                messages.append({
                    "role": "user",
                    "content": (
                        "ABGELEHNT: Das Limit von {} SPARQL-Abfragen ist erreicht. "
                        "Deine Query wurde NICHT ausgefuehrt. Gib JETZT action=answer aus. ..."
                    ).format(MAX_SPARQL_CALLS),
                })
                continue
```

---

## Schritt 10: Die Query wird ausgeführt und protokolliert

Ist das Limit noch nicht erreicht, wird die Query an Fuseki geschickt
(`run_sparql`), der Zähler erhöht und der Vorgang im `sparql_log`
festgehalten: welche Query, ob sie geklappt hat und wie viele Zeilen
zurückkamen.

```python
            query = parsed.get("query", "").strip()
            result = run_sparql(query)
            n_sparql_calls += 1

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
```

Die eigentliche Ausführung steckt in `run_sparql`: Sie schickt die Query
per HTTP an Fuseki, fängt HTTP-Fehler ab und wandelt das Ergebnis in
eine einfache Liste von Zeilen um.

```python
def run_sparql(query: str) -> dict:
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
```

---

## Schritt 11: Erfolgreiche Query -- Ergebnis zurück an das Modell

Hat die Query geklappt, werden zuerst rohe IRIs durch lesbare Labels
ersetzt (`resolve_uris_to_labels`), damit das Modell mit Namen statt
kryptischen URIs arbeitet. Dann wird das Ergebnis als neue
Benutzer-Nachricht zurück in die Konversation gehängt. Ein leeres
Ergebnis (n=0) gilt ausdrücklich NICHT als Fehler.

```python
            if result.get("ok"):
                if result.get("rows"):
                    result["rows"] = resolve_uris_to_labels(result["rows"])
                obs = json.dumps(result, ensure_ascii=False)[:2000]
                messages.append({
                    "role": "user",
                    "content": "SPARQL-Ergebnis (sparql_call_{}): {}".format(
                        n_sparql_calls, obs),
                })
```

Das Ersetzen der IRIs durch Labels macht `resolve_uris_to_labels` mit
einem einzigen Sammel-Lookup: Es sammelt alle vorkommenden HB-URIs und
holt deren `rdfs:label` in einer einzigen zusätzlichen Query.

```python
def resolve_uris_to_labels(rows: list) -> list:
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
    ...
```

---

## Schritt 12: Fehlgeschlagene Query -- Selbstkorrektur

War die Query technisch fehlerhaft (zum Beispiel Syntaxfehler oder
HTTP 400), bekommt das Modell den Fehlertext zurück, eingeleitet mit
"SPARQL-FEHLER". Der System-Prompt weist das Modell an, in diesem Fall
sofort eine korrigierte Query zu liefern. Das ist der
Self-Correction-Mechanismus.

```python
            else:
                detail = (result.get("error_detail") or result.get("error")
                          or "unbekannter Fehler")
                obs = "SPARQL-FEHLER (bitte Query korrigieren): {}".format(detail[:600])
                messages.append({"role": "user", "content": obs})
```

---

## Schritt 13: Beim Erreichen des Limits Antwort einfordern

Wurde mit dieser Query das Limit gerade erreicht, wird zusätzlich eine
Pflicht-Aufforderung angehängt: Das Modell muss jetzt antworten, keine
weiteren Abfragen mehr.

```python
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
```

Danach beginnt die Schleife von vorne (Schritt 4): Das Modell bekommt
die erweiterte Konversation samt Query-Ergebnis und entscheidet erneut,
ob es eine weitere Abfrage braucht oder antwortet.

---

## Schritt 14: Notausgang -- maximale Rundenzahl erreicht

Sollte das Modell nach zehn Runden immer noch keine Antwort geben
(in der Praxis greift vorher das SPARQL-Limit), bricht die Schleife mit
einem Timeout-Fehler ab.

```python
    return {
        "final": None,
        "error": "max turns ({}) reached without answer".format(MAX_TURNS),
        "n_rejected_sparql": n_rejected,
        "raw_history": raw_history,
        "sparql_queries": sparql_log,
        "turns": MAX_TURNS,
        ...
    }
```

---

## Schritt 15: Drumherum -- von der Frageliste bis zur Ergebnisdatei

`main` bindet alles zusammen: Benchmark laden, prüfen ob Fuseki
überhaupt erreichbar ist (ein einfaches ASK als Health-Check), dann jede
Frage durch `answer_one` schicken.

```python
    health = run_sparql("ASK { ?s ?p ?o }")
    if not health.get("ok"):
        print("FEHLER: Fuseki nicht erreichbar ...", file=sys.stderr)
        return 1

    for q in benchmark:
        out = answer_one(q["frage"], args.model)
        results.append({
            "id": q["id"],
            "klasse": q["klasse"],
            "frage": q["frage"],
            "ground_truth": q["ground_truth"],
            "model_answer": out.get("final"),
            "sparql_queries": out.get("sparql_queries"),
            ...
        })
```

Am Ende werden alle Ergebnisse zusammen mit dem Modellnamen und der
Prompt-Version in eine JSON-Datei geschrieben, die später von `eval.py`
ausgewertet wird.

```python
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
```

---

## Der Ablauf in einem Satz

Für jede Frage baut System B eine Konversation auf, lässt das Modell in
einer Schleife entweder eine SPARQL-Abfrage formulieren oder antworten,
führt die Abfragen gegen Fuseki aus, gibt die Ergebnisse (oder
Fehlermeldungen zur Selbstkorrektur) zurück und stoppt spätestens nach
drei Abfragen, woraufhin das Modell seine finale, evidenzgestützte
Antwort liefert.
