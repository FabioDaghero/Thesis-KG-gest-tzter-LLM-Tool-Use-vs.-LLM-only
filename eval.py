"""Evaluierung der System-A/B-Ergebnisse gegen die Ground Truth.

Aufruf: python eval.py [--dir VERZEICHNIS | --file DATEI | --all] [--compare]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


# Pfade
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
# Ground Truth steckt in den Result-JSONs selbst


# Hilfsfunktionen

def flatten_answer(answer: Any) -> str:
    """Normalisiert eine Modellantwort zu einem flachen String."""
    if answer is None:
        return ""
    if isinstance(answer, (int, float)):
        return str(answer)
    if isinstance(answer, str):
        return answer
    if isinstance(answer, list):
        parts = []
        for item in answer:
            if isinstance(item, dict):
                parts.extend(str(v) for v in item.values())
            else:
                parts.append(str(item))
        return " | ".join(parts)
    if isinstance(answer, dict):
        return " | ".join(str(v) for v in answer.values())
    return str(answer)


# exakte Werte ("0", "null", ...) separat; "0" nicht als Teilstring
NEGATIVE_KEYWORDS = (
    "nicht", "keine", "kein", "außerhalb", "not ", "no ", "outside",
    "null", "none", "leer", "empty", "unbekannt", "unknown",
    "not in", "not found",
)

def is_null_answer(answer: Any) -> bool:
    """Fallback ohne status-Feld: Antwort bedeutet 'nichts gefunden'."""
    if answer is None:
        return True
    s = flatten_answer(answer).strip().lower()
    if not s or s in ("null", "none", "0", "[]", ""):
        return True
    for kw in NEGATIVE_KEYWORDS:
        if kw in s:
            return True
    return False


def contains_uri(answer: Any) -> bool:
    """True wenn die Antwort eine HTTP-IRI enthält (Granularitätsproblem)."""
    s = flatten_answer(answer)
    return bool(re.search(r"https?://\S+", s))


def match_single(model_str: str, expected: Any) -> bool:
    """Exakter Teilstring-Match (case-insensitiv) für einen Erwartungswert."""
    if not model_str:
        return False
    needle = str(expected).lower().strip()
    haystack = model_str.lower()
    return needle in haystack


def match_list(model_str: str, expected_list: list) -> tuple[int, int]:
    """Gibt (treffer, gesamt) zurück."""
    total = len(expected_list)
    hits = sum(1 for e in expected_list if match_single(model_str, e))
    return hits, total


# Haupt-Matching-Logik

def evaluate_item(item: dict) -> dict:
    """Einzelne Benchmark-Frage auswerten."""
    qid = item["id"]
    klasse = item["klasse"]
    gt = item["ground_truth"]
    expected_answer = gt["answer"]

    # COUNT-Fragen haben expected_rows/expected_value statt expected_n
    expected_n = gt.get("expected_n")           # None bei COUNT-Fragen (K3.3)
    expected_rows = gt.get("expected_rows")     # nur COUNT-Fragen
    expected_value = gt.get("expected_value")   # konkreter Zahlenwert bei COUNT

    model_answer = item.get("model_answer") or {}
    raw_answer = model_answer.get("answer") if isinstance(model_answer, dict) else None
    # status-Feld aus model_answer (erweitertes Schema)
    model_status = model_answer.get("status") if isinstance(model_answer, dict) else None
    confidence = model_answer.get("confidence") if isinstance(model_answer, dict) else None
    model_str = flatten_answer(raw_answer)

    # evidence aus model_answer (System B liefert n_bindings)
    evidence = (model_answer.get("evidence") or {}) if isinstance(model_answer, dict) else {}
    evidence_n_bindings = evidence.get("n_bindings") if isinstance(evidence, dict) else None

    # System B: SPARQL-Metriken (erste Query)
    sparql_queries = item.get("sparql_queries", [])
    sparql_ok = False
    sparql_n = None
    sparql_error = None
    sparql_n_correct = None
    if sparql_queries:
        rs = sparql_queries[0].get("result_summary", {})
        sparql_ok = rs.get("ok", False)
        sparql_n = rs.get("n")
        sparql_error = rs.get("error")
        if sparql_n is not None:
            # COUNT-Frage vs. Listen-Frage
            if expected_rows is not None:
                sparql_n_correct = (sparql_n == expected_rows)
            elif expected_n is not None:
                sparql_n_correct = (sparql_n == expected_n)

    # neue Logging-Felder (aus System B v0.2+)
    n_sparql_calls = item.get("n_sparql_calls")
    first_query_ok = item.get("first_query_ok")
    final_query_ok = item.get("final_query_ok")
    repaired_after_error = item.get("repaired_after_error")

    # Tokens und Latenz
    tokens_prompt = item.get("tokens_prompt", 0) or 0
    tokens_completion = item.get("tokens_completion", 0) or 0
    tokens_total = tokens_prompt + tokens_completion
    latency_ms = item.get("latency_ms", 0) or 0
    turns = item.get("turns")

    # URI-Flag (Granularitätsproblem)
    has_uri = contains_uri(raw_answer)

    # Korrektheitsentscheidung
    correct = False
    partial = False

    if expected_n == 0:
        # N-Fragen — C2: status-Feld vorrangig nutzen
        if model_status is not None:
            correct = model_status in ("unknown", "unsupported")
            # status=error gilt nicht als korrekte Negativantwort
        else:
            # Fallback: Keyword-Matching (ältere Results ohne status-Feld)
            correct = is_null_answer(raw_answer)

    elif expected_rows is not None:
        # COUNT-Fragen — expected_value gegen model_answer prüfen
        check_val = expected_value if expected_value is not None else expected_answer
        correct = match_single(model_str, check_val)
        if not correct and sparql_n_correct:
            # SPARQL-Struktur korrekt, aber Wert falsch → teilweise
            partial = True

    elif isinstance(expected_answer, list):
        hits, total = match_list(model_str, expected_answer)
        if hits == total and total > 0:
            correct = True
        elif hits > 0:
            partial = True
        elif sparql_n_correct:
            partial = True

    else:
        # Einzelner Wert (String oder Zahl)
        correct = match_single(model_str, expected_answer)
        if not correct and sparql_n_correct:
            partial = True

    # Korrekt aber URI zurückgegeben → downgrade zu teilweise (nur wenn Label erwartet)
    if correct and has_uri and isinstance(expected_answer, str) and not expected_answer.startswith("http"):
        correct = False
        partial = True

    return {
        "id": qid,
        "klasse": klasse,
        "correct": correct,
        "partial": partial,
        "has_uri": has_uri,
        "model_status": model_status,
        "sparql_ok": sparql_ok if sparql_queries else None,
        "sparql_n": sparql_n,
        "sparql_n_correct": sparql_n_correct,
        "sparql_error": sparql_error,
        "turns": turns,
        # neue Felder
        "n_sparql_calls": n_sparql_calls,
        "first_query_ok": first_query_ok,
        "final_query_ok": final_query_ok,
        "repaired_after_error": repaired_after_error,
        "evidence_n_bindings": evidence_n_bindings,
        "tokens_prompt": tokens_prompt,
        "tokens_completion": tokens_completion,
        "tokens_total": tokens_total,
        "latency_ms": latency_ms,
        "confidence": confidence,
        "model_answer_raw": raw_answer,
        "expected_answer": expected_answer,
        "expected_n": expected_n,
        "expected_rows": expected_rows,
        "expected_value": expected_value,
    }


# Datei laden

def load_results_file(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def latest_file(prefix: str) -> Path | None:
    files = sorted(RESULTS_DIR.glob(f"{prefix}*.json"))
    return files[-1] if files else None


# Formatierungshelfer

def status_symbol(ev: dict) -> str:
    if ev["correct"]:
        return "✅"
    if ev["partial"]:
        return "~"
    return "❌"


def fmt(val, width: int, align: str = "<") -> str:
    s = str(val) if val is not None else "–"
    return f"{s:{align}{width}}"


def _pct(x: int, n: int) -> str:
    return f"{100 * x / n:.1f} %" if n else "–"


def _avg(values: list) -> str:
    vals = [v for v in values if v is not None]
    return f"{sum(vals) / len(vals):.2f}" if vals else "–"


# Bericht drucken

def print_report(run_data: dict, evals: list[dict], label: str):
    model = run_data.get("model", "?")
    system = "System B" if any(e["sparql_ok"] is not None for e in evals) else "System A"
    ts = run_data.get("timestamp", "")

    print(f"\n{'='*72}")
    print(f"  {label}  |  Modell: {model}  |  {system}  |  {ts}")
    print(f"{'='*72}")

    # Header
    col = ["ID", "K", "Stat", "Status", "Konfidenz", "Lat (ms)", "Tokens", "SPARQL-n"]
    print(f"  {fmt(col[0],6)} {fmt(col[1],2)} {fmt(col[2],3)} "
          f"{fmt(col[3],16)} {fmt(col[4],10)} {fmt(col[5],9)} "
          f"{fmt(col[6],7)} {fmt(col[7],8)}")
    print(f"  {'-'*6} {'-'*2} {'-'*3} {'-'*16} {'-'*10} {'-'*9} {'-'*7} {'-'*8}")

    for ev in evals:
        sparql_n_str = str(ev["sparql_n"]) if ev["sparql_n"] is not None else "–"
        conf_str = f"{ev['confidence']:.2f}" if ev["confidence"] is not None else "–"
        status_str = ev.get("model_status") or "–"
        uri_flag = " [URI]" if ev["has_uri"] else ""
        print(f"  {fmt(ev['id'],6)} {fmt(ev['klasse'],2)} {status_symbol(ev):<3} "
              f"{fmt(status_str,16)} {fmt(conf_str,10)} {fmt(ev['latency_ms'],9)} "
              f"{fmt(ev['tokens_total'],7)} {fmt(sparql_n_str,8)}{uri_flag}")

    # --- Zusammenfassung je Klasse ---
    print(f"\n  Zusammenfassung je Klasse:")
    print(f"  {'Klasse':<8} {'Korrekt':>8} {'Teilweise':>10} {'Fehler':>7} {'n':>4}")
    print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*7} {'-'*4}")

    for klasse in ["K1", "K2", "K3", "N"]:
        sub = [e for e in evals if e["klasse"] == klasse]
        if not sub:
            continue
        n = len(sub)
        ok = sum(1 for e in sub if e["correct"])
        pa = sum(1 for e in sub if e["partial"])
        err = n - ok - pa
        print(f"  {klasse:<8} {ok:>7}/{n}  {pa:>9}/{n}  {err:>6}/{n}  {n:>4}")

    # --- Gesamtmetriken ---
    n_total = len(evals)
    n_correct = sum(1 for e in evals if e["correct"])
    n_partial = sum(1 for e in evals if e["partial"])
    n_error = n_total - n_correct - n_partial

    avg_latency = sum(e["latency_ms"] for e in evals) / n_total if n_total else 0
    avg_tokens = sum(e["tokens_total"] for e in evals) / n_total if n_total else 0
    avg_conf = (
        sum(e["confidence"] for e in evals if e["confidence"] is not None) /
        sum(1 for e in evals if e["confidence"] is not None)
    ) if any(e["confidence"] is not None for e in evals) else None

    sparql_evals = [e for e in evals if e["sparql_ok"] is not None]
    sparql_ok_rate = (
        sum(1 for e in sparql_evals if e["sparql_ok"]) / len(sparql_evals)
        if sparql_evals else None
    )
    http400_rate = (
        sum(1 for e in sparql_evals if e["sparql_error"] == "HTTP 400") / len(sparql_evals)
        if sparql_evals else None
    )
    uri_count = sum(1 for e in evals if e["has_uri"])

    print(f"\n  Gesamt:")
    print(f"    Korrekt:       {n_correct}/{n_total}  ({_pct(n_correct, n_total)})")
    print(f"    Teilweise:     {n_partial}/{n_total}  ({_pct(n_partial, n_total)})")
    print(f"    Fehler:        {n_error}/{n_total}  ({_pct(n_error, n_total)})")
    print(f"    Avg Latenz:    {avg_latency:.0f} ms")
    print(f"    Avg Tokens:    {avg_tokens:.0f}")
    if avg_conf is not None:
        print(f"    Avg Konfidenz: {avg_conf:.2f}")
    if sparql_ok_rate is not None:
        print(f"    SPARQL-OK:     {sparql_ok_rate*100:.1f} %")
    if http400_rate is not None:
        print(f"    HTTP-400-Rate: {http400_rate*100:.1f} %")
    if uri_count:
        print(f"    URI-Antworten: {uri_count}  (Granularitätsproblem)")

    # --- C3: Neue Metriken (System B v0.2+) ---
    sparql_b2 = [e for e in evals if e.get("n_sparql_calls") is not None]
    if sparql_b2:
        avg_calls = sum(e["n_sparql_calls"] for e in sparql_b2) / len(sparql_b2)
        n_repaired = sum(1 for e in sparql_b2 if e.get("repaired_after_error"))
        first_ok = [e for e in sparql_b2 if e.get("first_query_ok") is not None]
        final_ok = [e for e in sparql_b2 if e.get("final_query_ok") is not None]
        first_ok_rate = sum(1 for e in first_ok if e["first_query_ok"]) / len(first_ok) if first_ok else None
        final_ok_rate = sum(1 for e in final_ok if e["final_query_ok"]) / len(final_ok) if final_ok else None
        ev_vals = [e["evidence_n_bindings"] for e in sparql_b2 if e.get("evidence_n_bindings") is not None]
        avg_bindings = sum(ev_vals) / len(ev_vals) if ev_vals else None

        print(f"\n  SPARQL-Verlauf (System B v0.2+):")
        print(f"    Avg SPARQL-Calls:     {avg_calls:.2f}")
        print(f"    Self-Corrections:     {n_repaired}/{len(sparql_b2)}  ({_pct(n_repaired, len(sparql_b2))})")
        if first_ok_rate is not None:
            print(f"    1. Query OK:          {first_ok_rate*100:.1f} %")
        if final_ok_rate is not None:
            print(f"    Letzte Query OK:      {final_ok_rate*100:.1f} %")
        if avg_bindings is not None:
            print(f"    Avg Evidence-Bindings:{avg_bindings:.1f}")

    print()


# Vergleichstabelle (mehrere Runs)

def print_comparison(runs: list[tuple[str, list[dict]]]):
    """Druckt eine kompakte Vergleichstabelle für mehrere Runs nebeneinander."""
    if len(runs) < 2:
        return
    all_ids = [e["id"] for e in runs[0][1]]
    labels = [label for label, _ in runs]

    print(f"\n{'='*72}")
    print("  VERGLEICHSTABELLE")
    print(f"{'='*72}")

    header = f"  {'ID':<8}"
    for label in labels:
        header += f"  {label[:12]:<12}"
    print(header)
    print(f"  {'-'*8}" + "".join(f"  {'-'*12}" for _ in labels))

    for qid in all_ids:
        row = f"  {qid:<8}"
        for _, evals in runs:
            ev = next((e for e in evals if e["id"] == qid), None)
            if ev is None:
                row += f"  {'–':<12}"
            else:
                sym = status_symbol(ev)
                conf = f"{ev['confidence']:.2f}" if ev["confidence"] is not None else "–"
                row += f"  {sym} conf={conf:<6}"
        print(row)

    print(f"\n  {'Korrekt':<8}", end="")
    for _, evals in runs:
        n = len(evals)
        ok = sum(1 for e in evals if e["correct"])
        print(f"  {ok}/{n} ({100*ok/n:.0f}%)    ", end="")
    print()
    print(f"  {'Teilweise':<8}", end="")
    for _, evals in runs:
        n = len(evals)
        pa = sum(1 for e in evals if e["partial"])
        print(f"  {pa}/{n} ({100*pa/n:.0f}%)    ", end="")
    print()
    print()


# Eval-Run: eine Datei

def run_eval(path: Path) -> tuple[dict, list[dict]]:
    data = load_results_file(path)
    evals = [evaluate_item(item) for item in data["results"]]
    return data, evals


# Ergebnis-JSON speichern

def save_eval_results(path: Path, run_data: dict, evals: list[dict],
                      output_dir: Path | None = None):
    n_total = len(evals)
    n_correct = sum(1 for e in evals if e["correct"])
    n_partial = sum(1 for e in evals if e["partial"])

    # aggregierte neue Metriken
    sparql_b2 = [e for e in evals if e.get("n_sparql_calls") is not None]
    c3_summary: dict = {}
    if sparql_b2:
        first_ok = [e for e in sparql_b2 if e.get("first_query_ok") is not None]
        final_ok = [e for e in sparql_b2 if e.get("final_query_ok") is not None]
        ev_vals = [e["evidence_n_bindings"] for e in sparql_b2 if e.get("evidence_n_bindings") is not None]
        c3_summary = {
            "avg_sparql_calls": round(
                sum(e["n_sparql_calls"] for e in sparql_b2) / len(sparql_b2), 2
            ),
            "n_repaired_after_error": sum(1 for e in sparql_b2 if e.get("repaired_after_error")),
            "first_query_ok_rate": round(
                sum(1 for e in first_ok if e["first_query_ok"]) / len(first_ok), 4
            ) if first_ok else None,
            "final_query_ok_rate": round(
                sum(1 for e in final_ok if e["final_query_ok"]) / len(final_ok), 4
            ) if final_ok else None,
            "avg_evidence_n_bindings": round(
                sum(ev_vals) / len(ev_vals), 2
            ) if ev_vals else None,
        }

    out = {
        "source_file": path.name,
        "model": run_data.get("model", "?"),
        "timestamp_eval": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "n_questions": n_total,
            "n_correct": n_correct,
            "n_partial": n_partial,
            "n_error": n_total - n_correct - n_partial,
            "accuracy_strict": round(n_correct / n_total, 4) if n_total else 0,
            "accuracy_lenient": round((n_correct + n_partial) / n_total, 4) if n_total else 0,
            "avg_latency_ms": round(sum(e["latency_ms"] for e in evals) / n_total, 1) if n_total else 0,
            "avg_tokens_total": round(sum(e["tokens_total"] for e in evals) / n_total, 1) if n_total else 0,
            **c3_summary,  # neue Felder eingebettet
        },
        "per_question": evals,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = path.stem
    save_dir = output_dir if output_dir is not None else RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"eval_{stem}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    return out_path


# CLI

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark-Evaluierung für System A / B")
    p.add_argument("--all", action="store_true", help="Alle JSON-Dateien in results/ auswerten")
    p.add_argument("--dir", type=Path, default=None,
                   help="Verzeichnis mit Ergebnis-JSONs (z.B. results/main_benchmark/)")
    p.add_argument("--file", type=Path, help="Bestimmte JSON-Ergebnisdatei auswerten")
    p.add_argument("--no-save", action="store_true", help="Kein eval_*.json schreiben")
    p.add_argument("--compare", action="store_true",
                   help="Neueste Datei je System nebeneinander vergleichen")
    return p.parse_args()


def main():
    args = parse_args()

    # Ziel-Verzeichnis fuer eval-JSON bestimmen
    eval_out_dir = args.dir if args.dir else RESULTS_DIR

    if args.file:
        files = [args.file]
    elif args.dir:
        d = args.dir
        files = sorted(d.glob("*.json"))
        files = [f for f in files if not f.name.startswith("eval_")]
    elif args.all:
        files = sorted(RESULTS_DIR.glob("*.json"))
        files = [f for f in files if not f.name.startswith("eval_")]
    else:
        files = []
        for prefix in ("system_a", "system_b"):
            f = latest_file(prefix)
            if f:
                files.append(f)
        if not files:
            print("Keine results/*.json gefunden. Bitte zuerst system_a.py / system_b.py ausführen.")
            sys.exit(1)

    all_runs: list[tuple[str, list[dict]]] = []

    for path in files:
        print(f"\nLade: {path.name}")
        try:
            run_data, evals = run_eval(path)
        except Exception as exc:
            print(f"  Fehler beim Laden: {exc}")
            continue

        label = path.stem
        print_report(run_data, evals, label)
        all_runs.append((label, evals))

        if not args.no_save:
            out_path = save_eval_results(path, run_data, evals, eval_out_dir)
            print(f"  Eval gespeichert: {out_path.name}")

    if args.compare and len(all_runs) >= 2:
        print_comparison(all_runs)


if __name__ == "__main__":
    main()
