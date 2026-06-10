"""Aggregiert Eval-Ergebnisse zu results_summary.yaml.

Aufruf: python make_results_summary.py --dir RESULTS_DIR [--a-dir DIR]
        python make_results_summary.py --compare DIR1 DIR2 [...]
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).parent
KLASSEN = ["K1", "K2", "K3", "N"]


def load_evals(results_dir: str, system: str) -> list[dict]:
    """Neueste Eval-Datei je System-Quelldatei laden."""
    by_src: dict[str, dict] = {}
    pattern = str(ROOT / results_dir / f"eval_{system}_*.json")
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f, encoding="utf-8"))
        by_src[d["source_file"]] = d  # sortiert -> letzte Eval gewinnt
    runs = [
        {q["id"]: q for q in d["per_question"]}
        for _, d in sorted(by_src.items())
    ]
    return runs


def score(q: dict) -> float:
    return 1.0 if q["correct"] else (0.5 if q["partial"] else 0.0)


def overall_stats(runs: list[dict]) -> dict:
    n = len(next(iter(runs)).keys()) if runs else 0
    strict_runs = [round(sum(q["correct"] for q in r.values()) / n, 4) for r in runs]
    lenient_runs = [
        round(sum(q["correct"] or q["partial"] for q in r.values()) / n, 4)
        for r in runs
    ]
    total = len(runs) * n
    out = {
        "strict_mean": round(sum(strict_runs) / len(strict_runs), 4),
        "strict_per_run": strict_runs,
        "lenient_mean": round(sum(lenient_runs) / len(lenient_runs), 4),
        "lenient_per_run": lenient_runs,
        "n_correct_strict_per_run": [
            sum(q["correct"] for q in r.values()) for r in runs
        ],
        "avg_latency_s": round(
            sum(q["latency_ms"] for r in runs for q in r.values()) / total / 1000, 1
        ),
        "avg_tokens_total": round(
            sum(q["tokens_total"] for r in runs for q in r.values()) / total
        ),
    }
    return out


def sparql_stats(runs: list[dict]) -> dict:
    calls = [q["n_sparql_calls"] for r in runs for q in r.values()
             if q.get("n_sparql_calls") is not None]
    with_calls = [q for r in runs for q in r.values()
                  if (q.get("n_sparql_calls") or 0) > 0]
    fco = [q["first_query_ok"] for q in with_calls]
    rep = sum(bool(q.get("repaired_after_error")) for r in runs for q in r.values())
    rej = sum(q.get("n_rejected_sparql") or 0 for r in runs for q in r.values())
    return {
        "avg_sparql_calls": round(sum(calls) / len(calls), 2) if calls else None,
        "sparql_first_call_success_rate": (
            round(sum(fco) / len(fco), 3) if fco else None
        ),
        "sparql_first_call_success_abs": f"{sum(fco)}/{len(fco)}" if fco else None,
        "repaired_after_error_total": rep,
        "rejected_after_limit_total": rej,
        "max_sparql_calls_single_question": max(calls) if calls else None,
        "runs_without_sparql_call": [
            (i + 1, q["id"]) for i, r in enumerate(runs) for q in r.values()
            if q.get("n_sparql_calls") == 0
        ],
    }


def confident_wrong(runs: list[dict], threshold: float = 0.7) -> dict:
    r = runs[0]  # System A deterministisch; Run 1 repraesentativ
    n = len(r)
    wrong = [q for q in r.values() if not q["correct"]]
    cw = [q for q in wrong if (q.get("confidence") or 0) >= threshold]
    return {
        "confident_wrong_rate_of_all": round(len(cw) / n, 3),
        "confident_wrong_rate_of_wrong": round(len(cw) / len(wrong), 3) if wrong else 0,
        "note": (
            f"{len(cw)} von {n} Antworten sind falsch UND confidence>={threshold}; "
            f"bezogen auf die {len(wrong)} falschen Antworten: "
            f"{len(cw) / len(wrong):.1%}" if wrong else ""
        ),
        "examples": [
            {"id": q["id"], "conf": q.get("confidence"),
             "ans": str(q.get("model_answer_raw"))[:60]}
            for q in sorted(cw, key=lambda x: x["id"])[:15]
        ],
    }


def by_klasse(a_runs: list[dict], b_runs: list[dict]) -> dict:
    def agg(runs):
        acc = defaultdict(lambda: [0, 0, 0])
        for r in runs:
            for q in r.values():
                k = q["klasse"]
                acc[k][0] += q["correct"]
                acc[k][1] += q["correct"] or q["partial"]
                acc[k][2] += 1
        return {k: (round(v[0] / v[2], 3), round(v[1] / v[2], 3))
                for k, v in acc.items()}

    pa, pb = agg(a_runs), agg(b_runs)
    return {
        k: {
            "n": sum(1 for q in a_runs[0].values() if q["klasse"] == k),
            "a_strict": pa[k][0], "a_lenient": pa[k][1],
            "b_strict": pb[k][0], "b_lenient": pb[k][1],
        }
        for k in KLASSEN
    }


def per_question(a_runs: list[dict], b_runs: list[dict], bench: dict) -> dict:
    out = {}
    for qid in sorted(b_runs[0], key=lambda x: (x.split(".")[0], x)):
        a_vals = [score(r[qid]) for r in a_runs]
        b_vals = [score(r[qid]) for r in b_runs]
        out[qid] = {
            "klasse": b_runs[0][qid]["klasse"],
            "frage": bench[qid]["frage"],
            "a_mean": round(sum(a_vals) / len(a_vals), 4),
            "b_mean": round(sum(b_vals) / len(b_vals), 4),
            "delta": round((sum(b_vals) - sum(a_vals)) / len(a_vals), 4),
            "a_vals": a_vals,
            "b_vals": b_vals,
        }
    return out


def build_summary(b_dir: str, a_dir: str | None, benchmark: str,
                  prompt_version: str | None) -> dict:
    a_runs = load_evals(a_dir or b_dir, "system_a")
    b_runs = load_evals(b_dir, "system_b")
    if not a_runs:
        raise SystemExit(f"Keine system_a-Evals in {a_dir or b_dir} gefunden.")
    if not b_runs:
        raise SystemExit(f"Keine system_b-Evals in {b_dir} gefunden.")
    bench = {q["id"]: q for q in yaml.safe_load(
        open(ROOT / "data" / benchmark, encoding="utf-8"))}

    summary = {
        "meta": {
            "created": str(date.today()),
            "generator": "make_results_summary.py",
            "benchmark": benchmark,
            "results_dir_b": b_dir,
            "results_dir_a": a_dir or b_dir,
            "n_questions": len(b_runs[0]),
            "n_runs_a": len(a_runs),
            "n_runs_b": len(b_runs),
            "prompt_version_b": prompt_version,
        },
        "overall": {
            "system_a": {**overall_stats(a_runs), **confident_wrong(a_runs)},
            "system_b": {**overall_stats(b_runs), **sparql_stats(b_runs)},
        },
        "by_klasse": by_klasse(a_runs, b_runs),
        "per_question": per_question(a_runs, b_runs, bench),
    }
    # Instabile Fragen ausweisen
    unstable = {
        qid: e["b_vals"] for qid, e in summary["per_question"].items()
        if len(set(e["b_vals"])) > 1
    }
    summary["instability"] = {
        "n_unstable_questions_b": len(unstable),
        "unstable_b_vals": unstable,
    }
    return summary


def compare(dirs: list[str], benchmark: str) -> None:
    """Ablationsvergleich: gleiche Fragen, verschiedene Prompt-Versionen."""
    bench = {q["id"]: q for q in yaml.safe_load(
        open(ROOT / "data" / benchmark, encoding="utf-8"))}
    cols = []
    for d in dirs:
        runs = load_evals(d, "system_b")
        if not runs:
            print(f"WARNUNG: keine B-Evals in {d}, uebersprungen")
            continue
        pv = None
        for f in sorted(glob.glob(str(ROOT / d / "system_b_*.json"))):
            pv = json.load(open(f, encoding="utf-8")).get("prompt_version")
            break
        cols.append((d, pv, runs))

    print(f"{'':8s}" + "".join(f"{pv or d[-12:]:>14s}" for d, pv, _ in cols))
    # Gesamt
    for label, fn in [
        ("strict", lambda r: sum(q["correct"] for q in r.values()) / len(r)),
        ("lenient", lambda r: sum(q["correct"] or q["partial"]
                                  for q in r.values()) / len(r)),
    ]:
        vals = [sum(fn(r) for r in runs) / len(runs) for _, _, runs in cols]
        print(f"{label:8s}" + "".join(f"{v:14.4f}" for v in vals))
    # SPARQL-Statistiken
    for d, pv, runs in cols:
        s = sparql_stats(runs)
        print(f"\n[{pv or d}] avg_calls={s['avg_sparql_calls']} "
              f"first_ok={s['sparql_first_call_success_rate']} "
              f"repaired={s['repaired_after_error_total']} "
              f"rejected={s['rejected_after_limit_total']} "
              f"max={s['max_sparql_calls_single_question']}")
    # Per-Frage-Deltas zwischen aufeinanderfolgenden Versionen
    for (d1, pv1, r1), (d2, pv2, r2) in zip(cols, cols[1:]):
        print(f"\n=== Veraenderte Fragen {pv1 or d1} -> {pv2 or d2} ===")
        for qid in sorted(r1[0], key=lambda x: (x.split('.')[0], x)):
            m1 = sum(score(r[qid]) for r in r1) / len(r1)
            m2 = sum(score(r[qid]) for r in r2) / len(r2)
            if abs(m1 - m2) > 0.01:
                print(f"  {qid:6s} {m1:.2f} -> {m2:.2f} "
                      f"({'+' if m2 > m1 else ''}{m2 - m1:.2f})  "
                      f"| {bench[qid]['frage'][:60]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", help="Results-Ordner mit System-B-Evals")
    ap.add_argument("--a-dir", help="Ordner mit System-A-Evals (Default: --dir)")
    ap.add_argument("--benchmark", default="benchmark_v1.yaml")
    ap.add_argument("--out", default="data/results_summary.yaml")
    ap.add_argument("--prompt-version", default=None,
                    help="Wird in meta geschrieben (sonst aus Result-JSON)")
    ap.add_argument("--compare", nargs="+",
                    help="2+ Results-Ordner fuer Ablationsvergleich")
    args = ap.parse_args()

    if args.compare:
        compare(args.compare, args.benchmark)
        return 0

    if not args.dir:
        ap.error("--dir oder --compare erforderlich")

    pv = args.prompt_version
    if pv is None:
        for f in sorted(glob.glob(str(ROOT / args.dir / "system_b_*.json"))):
            pv = json.load(open(f, encoding="utf-8")).get("prompt_version")
            break

    summary = build_summary(args.dir, args.a_dir, args.benchmark, pv)
    out_path = ROOT / args.out
    out_path.write_text(
        yaml.safe_dump(summary, allow_unicode=True, sort_keys=False,
                       default_flow_style=False, width=100),
        encoding="utf-8",
    )
    o = summary["overall"]
    print(f"Geschrieben: {out_path}")
    print(f"A strict={o['system_a']['strict_mean']}  "
          f"B strict={o['system_b']['strict_mean']} "
          f"(Runs: {o['system_b']['strict_per_run']})  "
          f"B lenient={o['system_b']['lenient_mean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
